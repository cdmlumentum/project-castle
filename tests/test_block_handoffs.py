"""Block handoff and command integration tests; all optical modes are synthetic."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from test_python_pso import ROOT, SyntheticBackend
from vlac import cli, blocks
from vlac.runtime import resolve_published
from vlac.block_io import digest, load_block, write_json, physics_signature
from vlac.config import load_config
from vlac.modes import extract_coupling
from vlac.solver import LumapiBackend
from vlac.materials import build_material_bank


class InspectionBackend(SyntheticBackend):
    """Fake MODE session: file contents are explicitly synthetic test data."""
    save_endpoint_inspection = LumapiBackend.save_endpoint_inspection
    instances = []

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.bank = build_material_bank(config)
        self.closed = False
        self.scene_calls = []
        self.session = SimpleNamespace(save=self.save)
        type(self).instances.append(self)

    def _mode_candidates(self, geometry, wavelength, **kwargs):
        self.scene_calls.append((geometry, wavelength, kwargs))
        return [object()]

    def save(self, path):
        if self.closed:
            raise AssertionError('Inspection attempted after closing MODE')
        Path(path).write_text('SYNTHETIC TEST FILE ONLY - not a MODE project')

    def __exit__(self, *args):
        self.closed = True
        return False


class RecordingBackend(SyntheticBackend):
    instances = []

    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.coefficient_calls = []
        type(self).instances.append(self)

    def evaluate_coefficients(self,geometry,wavelength,**kwargs):
        self.coefficient_calls.append((geometry,wavelength))
        delta = 1000.
        kappa = 100. if geometry.state=='cross' else 60.
        q = np.sqrt((delta/2)**2+kappa**2)
        return extract_coupling(1e7+delta/2,1e7-delta/2,1e7+q,1e7-q),None


class BlockHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = load_config(ROOT/'config/python_joint.json')
        self.config = replace(base,project_root=self.root,
            materials=replace(base.materials,silicon_data_csv=str(ROOT/'data/green_si_2008_1260_1340.csv')),
            endpoint_search=replace(base.endpoint_search,adaptive_max_widths=13),
            device=replace(base.device,z_samples_initial=3,wavelength_samples_initial=3,z_samples_max=9,wavelength_samples_max=9),
            optimizer=replace(base.optimizer,max_function_evaluations_per_grid=1,max_grid_refinements=0),
            propagation=replace(base.propagation,rk4_steps_initial=16,rk4_steps_max=128,power_convergence_rtol=1e-4))
        self.config_path = self.root/'config/test.json'
        self.config_path.parent.mkdir()
        write_json(self.config_path,self.config.as_dict())
        self.b1,self.b2,self.b3,self.b4 = (self.root/f'block{i}' for i in range(1,5))
        RecordingBackend.instances = []

    def tearDown(self):
        self.temp.cleanup()

    def command(self,*args,backend=RecordingBackend):
        with patch.object(cli,'LumapiBackend',backend),contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            return cli.main([*map(str,args),'--config',str(self.config_path)])

    def endpoint(self):
        self.assertEqual(self.command('block1','--out',self.b1),0)

    def test_endpoint_inspection_files_and_complex_field_roundtrip(self):
        self.assertEqual(self.command('block1','--out',self.b1,backend=InspectionBackend),0)
        report = json.loads((self.b1/'endpoint_diagnostics.json').read_text())
        backend = InspectionBackend.instances[-1]
        self.assertEqual(len([c for c in backend.calls if not c[3]]),10)
        self.assertEqual(len(backend.scene_calls),1)
        geometry,wavelength,flags = backend.scene_calls[0]
        self.assertEqual(flags,{'include_upper':True,'include_lower':True})
        endpoint = load_block(self.b1,1,self.config)[0]
        self.assertEqual(geometry.w1_m,endpoint.w1_input_m)
        self.assertEqual(geometry.d_c2c_m,self.config.geometry.d_c2c_input_m)
        self.assertEqual((geometry.state,wavelength),('cross',self.config.device.wavelength_min_m))
        self.assertEqual(len(list(self.b1.glob('*.lms'))),1)
        self.assertEqual(report['inspection_project']['status'],'saved')
        self.assertEqual(len(report['mode_profiles']['modes']),10)
        with np.load(self.b1/'endpoint_mode_profiles.npz',allow_pickle=False) as fields:
            self.assertEqual(len(fields.files),40)
            for item,packet in zip(report['mode_profiles']['modes'],backend._block1_mode_profiles):
                prefix = item['array_prefix']
                np.testing.assert_array_equal(fields[prefix+'_ex'],packet['mode']['ex'])
                np.testing.assert_array_equal(fields[prefix+'_ey'],packet['mode']['ey'])
                self.assertEqual(fields[prefix+'_ex'].shape,
                    (len(fields[prefix+'_x_m']),len(fields[prefix+'_y_m'])))
        self.assertTrue((self.b1/'endpoint_betas_linear.pdf').read_bytes().startswith(b'%PDF'))
        self.assertGreater((self.b1/'endpoint_betas_linear.png').stat().st_size,1000)
        self.assertGreater((self.b1/'endpoint_mode_profiles.png').stat().st_size,1000)

    def test_failed_rerun_preserves_successful_inspection_artifacts(self):
        from vlac.endpoint_diagnostics import FILES
        self.assertEqual(self.command('block1','--out',self.b1,backend=InspectionBackend),0)
        previous = load_block(self.b1,1,self.config)[3]['sha256']
        class Failure(InspectionBackend):
            def preflight(self, wavelength):
                raise RuntimeError('Synthetic startup failure')
        self.assertEqual(self.command('block1','--out',self.b1,backend=Failure),2)
        self.assertTrue(all((self.b1/name).exists() for name in (*FILES,'block1_result.json')))
        self.assertEqual(load_block(self.b1,1,self.config)[3]['sha256'],previous)

    def test_inspection_save_failure_does_not_publish_success(self):
        class Failure(InspectionBackend):
            def save(self,path):
                raise RuntimeError('Synthetic project save failure')
        self.assertEqual(self.command('block1','--out',self.b1,backend=Failure),2)
        self.assertFalse((self.b1/'block1_result.json').exists())
        self.assertIn('project save failure',(self.b1/'endpoint_failure.json').read_text())

    def profile(self):
        self.endpoint()
        self.assertEqual(self.command('block2','--endpoint-dir',self.b1,'--out',self.b2),0)

    def bar(self):
        self.profile()
        self.assertEqual(self.command('block3','--profile-dir',self.b2,'--out',self.b3),0)

    def test_four_individual_commands_and_block4_never_launches_mode(self):
        self.endpoint()
        endpoint = load_block(self.b1,1,self.config)[0]
        before = len(RecordingBackend.instances)
        self.assertEqual(self.command('validate-block2','--endpoint-dir',self.b1,'--out',self.b2),3)
        self.assertEqual(len(RecordingBackend.instances),before+1)
        profile_backend = RecordingBackend.instances[-1]
        self.assertFalse(profile_backend.calls)  # No endpoint isolated-mode search.
        self.assertTrue(profile_backend.coefficient_calls)
        self.assertEqual({g.state for g,w in profile_backend.coefficient_calls},{'cross'})
        self.assertEqual({w for g,w in profile_backend.coefficient_calls},set(endpoint.wavelengths_m))
        saved_endpoint,solution,_,artifact = load_block(self.b2,2,self.config)
        self.assertEqual(saved_endpoint.w1_input_m,endpoint.w1_input_m)
        self.assertEqual(solution.w1_control_m[0],endpoint.w1_input_m)
        self.assertEqual(solution.w1_control_m[-1],endpoint.w1_output_m)
        self.assertEqual(artifact['validation']['status'],'review')
        self.assertFalse(artifact['validation']['optimization_converged'])
        self.assertTrue((self.b2/'block2_validation_report.json').is_file())
        with patch('vlac.blocks.optimize_profiles',side_effect=AssertionError('Block 2 repeated')):
            self.assertEqual(self.command('validate-block3','--profile-dir',self.b2,'--out',self.b3),3)
        self.assertEqual({g.state for g,w in RecordingBackend.instances[-1].coefficient_calls},{'bar'})
        def forbidden(*args,**kwargs): raise AssertionError('MODE launched for Block 4')
        self.assertEqual(self.command('validate-block4','--bar-dir',self.b3,'--out',self.b4,backend=forbidden),3)
        with (self.b4/'performance_by_launch.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),8)
        self.assertEqual({(r['state'],r['launch_layer']) for r in rows},{(s,l) for s in ('cross','bar') for l in ('upper','lower')})
        self.assertTrue(all(r['integration_converged']=='True' for r in rows))
        self.assertEqual(json.loads((self.b4/'block4_validation_report.json').read_text())['mode_required_for_block4'],False)
        self.assertTrue((self.b2/'cross_coefficients.csv').exists())
        self.assertFalse((self.b2/'bar_coefficients.csv').exists())
        self.assertFalse((self.b2/'performance_by_launch.csv').exists())
        self.assertTrue((self.b3/'bar_coefficients.csv').exists())
        self.assertFalse((self.b3/'performance_by_launch.csv').exists())

    def test_validate_block2_can_pass_when_optimizer_and_grid_converge(self):
        self.endpoint()
        self.assertEqual(self.command('validate-block2','--endpoint-dir',self.b1,'--out',self.b2,
                                      '--max-evaluations','120','--max-refinements','0'),0)
        report = json.loads((self.b2/'block2_validation_report.json').read_text())
        self.assertTrue(report['optimization_converged'])
        self.assertTrue(report['grid_converged'])
        self.assertGreater(report['objective_achieved'],.01)  # Target fit is distinct from convergence.

    def test_legacy_v21_endpoint_directory_import_without_search(self):
        self.endpoint()
        (self.b1/'latest.json').unlink()
        (self.b1/'block1_result.json').unlink()
        endpoint,_,_,source = load_block(self.b1,1,self.config)
        self.assertTrue(source['legacy_import'])
        with patch('vlac.cli.search_endpoints',side_effect=AssertionError('endpoint repeated')):
            self.assertEqual(self.command('block2','--endpoint-dir',self.b1,'--out',self.b2),0)
        loaded = load_block(self.b2,2,self.config)[0]
        self.assertEqual(loaded.w1_input_m,endpoint.w1_input_m)

    def test_legacy_import_rejects_disagreeing_verification_and_finite_slabs(self):
        self.endpoint()
        (self.b1/'latest.json').unlink()
        (self.b1/'block1_result.json').unlink()
        original = (self.b1/'configuration.json').read_text()
        raw = json.loads(original);raw['geometry']['rib_slab_overhang_m']=2e-6
        write_json(self.b1/'configuration.json',raw)
        with self.assertRaisesRegex(ValueError,'Finite-slab'):load_block(self.b1,1,self.config)
        (self.b1/'configuration.json').write_text(original)
        report = json.loads((self.b1/'endpoint_adaptive_verification.json').read_text())
        report['endpoint_records'][0]['delta_beta_real_rad_per_m']+=10.
        write_json(self.b1/'endpoint_adaptive_verification.json',report)
        with self.assertRaisesRegex(ValueError,'disagrees'):load_block(self.b1,1,self.config)

    def test_configuration_compatibility_and_runtime_changes(self):
        self.endpoint()
        runtime = replace(self.config,solver=replace(self.config.solver,api_path='different-machine',threads=8,hide_gui=True),
                          optimizer=replace(self.config.optimizer,max_function_evaluations_per_grid=20),
                          device=replace(self.config.device,length_m=2e-3))
        load_block(self.b1,1,runtime)  # Endpoint physics is length-independent.
        for changed in (replace(self.config,materials=replace(self.config.materials,liquid_cross_n=1.56)),
                        replace(self.config,geometry=replace(self.config.geometry,w1_lower_m=1.9e-6)),
                        replace(self.config,solver=replace(self.config.solver,mesh_cells_x=600))):
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'incompatible'):load_block(self.b1,1,changed)
        self.assertNotEqual(physics_signature(self.config,2),physics_signature(runtime,2))

    def test_corrupt_input_leaves_previous_target_intact(self):
        self.endpoint()
        raw = json.loads((self.b1/'block1_result.json').read_text())
        raw['payload']['endpoint']['w1_input_m']+=1e-8
        write_json(resolve_published(self.b1)/'block1_result.json',raw)
        self.b2.mkdir();(self.b2/'block2_result.json').write_text('{"stale":true}')
        self.assertEqual(self.command('block2','--endpoint-dir',self.b1,'--out',self.b2,
                                      backend=lambda *a,**k: self.fail('MODE launched before input checks')),2)
        self.assertEqual((self.b2/'block2_result.json').read_text(),'{"stale":true}')
        self.assertTrue((self.b2/'block2_failure.json').exists())
        with self.assertRaisesRegex(ValueError,'checksum'):load_block(self.b1,1,self.config)

    def test_structural_grid_target_and_control_validation(self):
        self.profile()
        original = json.loads((self.b2/'block2_result.json').read_text())
        mutations = [lambda s:s['cross_coefficients'][0].pop(),
                     lambda s:s['w1_control_m'].__setitem__(0,1.5e-6),
                     lambda s:s['delta_target_rad_per_m'].__setitem__(0,123.),
                     lambda s:s['z_m'].__setitem__(1,s['z_m'][0]),
                     lambda s:s['cross_coefficients'][0][0]['delta_beta'].__setitem__('real',9999.)]
        for change in mutations:
            raw = deepcopy(original);change(raw['payload']['solution']);raw.pop('sha256');raw['sha256']=digest(raw)
            raw['content_id']=digest({'physics':raw['physics_signature'],'payload':raw['payload']})
            raw.pop('sha256');raw['sha256']=digest(raw)
            write_json(resolve_published(self.b2)/'block2_result.json',raw)
            with self.subTest(change=change),self.assertRaises(ValueError):load_block(self.b2,2,self.config)
        write_json(resolve_published(self.b2)/'block2_result.json',original)
        changed_length = replace(self.config,device=replace(self.config.device,length_m=2e-3))
        with self.assertRaisesRegex(ValueError,'length_m'):load_block(self.b2,2,changed_length)

    def test_material_table_content_changes_invalidate_results(self):
        self.endpoint()
        table = self.root/'liquid.csv'
        table.write_text('wavelength_um,n\n1.26,1.55\n1.34,1.55\n')
        config = replace(self.config,materials=replace(self.config.materials,liquid_dispersion_csv=str(table)))
        endpoint = load_block(self.b1,1,self.config)[0]
        from vlac.block_io import write_artifact
        target = self.root/'table_endpoint'
        write_artifact(target,1,config,endpoint)
        table.write_text('wavelength_um,n\n1.26,1.56\n1.34,1.56\n')
        with self.assertRaisesRegex(ValueError,'materials'):load_block(target,1,config)

    def test_moved_block3_is_self_contained_and_run_resume_is_offline(self):
        self.bar()
        moved = self.root/'moved_coefficients'
        shutil.move(self.b3,moved)
        shutil.rmtree(self.b1);shutil.rmtree(self.b2)
        self.assertEqual(self.command('run','--bar-dir',moved,'--out',self.b4,
                                     backend=lambda *a,**k:self.fail('MODE used for saved-coefficient continuation')),0)
        self.assertTrue((self.b4/'block4_result.json').is_file())
        self.assertTrue((self.b4/'block3_result.json').is_file())

    def test_full_run_checkpoints_survive_later_failure_and_resume_skips_earlier_blocks(self):
        out = self.root/'full'
        with patch('vlac.blocks.run_bar',side_effect=RuntimeError('bar fixture failure')):
            self.assertEqual(self.command('run','--out',out),2)
        partial = out/json.loads((out/'last_attempt.json').read_text())['attempt']
        self.assertTrue((partial/'block1_result.json').is_file())
        self.assertTrue((partial/'block2_result.json').is_file())
        self.assertFalse((out/'block3_result.json').exists())
        with patch('vlac.cli.search_endpoints',side_effect=AssertionError('endpoint repeated')),patch('vlac.blocks.optimize_profiles',side_effect=AssertionError('profile repeated')):
            self.assertEqual(self.command('run','--out',out),0)
        self.assertEqual({g.state for g,w in RecordingBackend.instances[-1].coefficient_calls},{'bar'})
        self.assertTrue((out/'block4_result.json').is_file())

    def test_run_can_resume_directly_from_endpoint(self):
        self.endpoint()
        with patch('vlac.cli.search_endpoints',side_effect=AssertionError('endpoint repeated')):
            self.assertEqual(self.command('run','--endpoint-dir',self.b1,'--out',self.root/'continued'),0)
        self.assertFalse(RecordingBackend.instances[-1].calls)

    def test_validate_block1_exports_usable_handoff(self):
        self.assertEqual(self.command('validate-block1','--out',self.b1),0)
        _,_,_,artifact=load_block(self.b1,1,self.config)
        self.assertEqual(artifact['validation']['status'],'passed')
        self.assertTrue((self.b1/'endpoint_detunings.csv').exists())
        self.assertTrue((self.b1/'block1_validation_report.json').exists())

    def test_bar_validation_flags_underresolved_coefficients(self):
        self.profile()
        class Rapid(RecordingBackend):
            def evaluate_coefficients(self,g,wavelength,**kwargs):
                delta = 1000.+2e6*np.cos(2*np.pi*(g.d_c2c_m-self.config.geometry.d_c2c_input_m)/(self.config.geometry.d_c2c_output_m-self.config.geometry.d_c2c_input_m))
                q=np.sqrt((delta/2)**2+100.**2)
                return extract_coupling(1e7+delta/2,1e7-delta/2,1e7+q,1e7-q),None
        self.assertEqual(self.command('validate-block3','--profile-dir',self.b2,'--out',self.b3,backend=Rapid),3)
        report = json.loads((self.b3/'block3_validation_report.json').read_text())
        self.assertGreater(report['grid_validation']['z_coefficient_interpolation_error'],self.config.device.grid_convergence_rtol)
        self.assertIsNone(report['grid_validation']['wavelength_coefficient_interpolation_error'])
        self.assertTrue((self.b3/'block3_result.json').exists())

    def test_block4_reports_step_limit_without_mode_and_preserves_review_export(self):
        self.bar()
        self.assertEqual(self.command('validate-block4','--bar-dir',self.b3,'--out',self.b4,
                                     '--rk4-steps-initial','16','--rk4-steps-max','16',
                                     backend=lambda *a,**k:self.fail('MODE used for propagation')),3)
        self.assertFalse(json.loads((self.b4/'block4_validation_report.json').read_text())['integration_converged'])
        self.assertTrue((self.b4/'performance_by_launch.csv').is_file())


if __name__=='__main__':unittest.main()
