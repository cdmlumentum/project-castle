"""Workflow regression tests; every optical fixture here is synthetic."""
from dataclasses import replace, asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sqlite3
from contextlib import closing
import numpy as np

from test_python_pso import ROOT, SyntheticBackend
from test_block_handoffs import RecordingBackend
from test_efficiency import fixture_modes, geometry
from vlac.config import load_config
from vlac.materials import build_material_bank
from vlac.adaptive_endpoint import search_endpoints_adaptive
from vlac.optimize import optimize_profiles
from vlac.characterize import characterize_profile
from vlac.block_io import write_artifact, load_block
from vlac.checkpoints import Checkpoint
from vlac.cache import SQLiteCache
from vlac.runtime import RunBudget, BudgetExceeded, output_attempt, resolve_published
from vlac.solver import LumapiBackend
from vlac.native_backend import NativeLumapiBackend
from vlac.propagate import propagate_both, propagate_converged


class StorageTests(unittest.TestCase):
    def test_numeric_cache_roundtrip_corruption_and_pruning(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = SQLiteCache(Path(folder)/'data.sqlite')
            cache.put('mode', fixture_modes(), kind='raw')
            np.testing.assert_array_equal(cache.get('mode')[0].ex,fixture_modes()[0].ex)
            self.assertEqual(cache.inspect()['entries'],1)
            with closing(sqlite3.connect(cache.path)) as db, db:
                db.execute("UPDATE cache_v2 SET payload=? WHERE cache_key='mode'",(b'corrupt',))
            self.assertIsNone(cache.get('mode'))
            cache.put('old',np.ones(100)); cache.put('new',np.zeros(100))
            self.assertEqual(cache.prune(0),2)
            self.assertEqual(cache.count(),0)

    def test_checkpoint_atomic_save_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Checkpoint(folder,'optimizer','identity')
            checkpoint.save({'controls':np.arange(4), 'complex':1+2j})
            with patch('vlac.checkpoints.encode',side_effect=ValueError('bad data')):
                with self.assertRaises(ValueError):checkpoint.save({})
            self.assertEqual(checkpoint.load()['complex'],1+2j)
            checkpoint.path.write_bytes(checkpoint.path.read_bytes()+b'broken')
            with self.assertRaisesRegex(ValueError,'checksum'):checkpoint.load()

    def test_failed_attempt_preserves_previous_published_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with output_attempt(folder,'first') as attempt:(attempt/'result.txt').write_text('good')
            previous=resolve_published(folder)
            with self.assertRaises(BudgetExceeded):
                with output_attempt(folder,'second') as attempt:
                    (attempt/'result.txt').write_text('partial')
                    raise BudgetExceeded('test budget')
            self.assertEqual(resolve_published(folder),previous)
            self.assertEqual((Path(folder)/'result.txt').read_text(),'good')

    def test_budget_reserves_retries(self):
        config=load_config(ROOT/'config/default.json')
        budget=RunBudget(replace(config.execution,max_solves=3))
        budget.check(3);budget.consume(2)
        with self.assertRaises(BudgetExceeded):budget.check(2)
        budget.check(1)

    def test_cache_only_backend_never_imports_or_opens_mode(self):
        config=load_config(ROOT/'config/default.json')
        with tempfile.TemporaryDirectory() as folder, patch.object(LumapiBackend,'_import_lumapi',side_effect=AssertionError('license acquired')):
            with NativeLumapiBackend(config,cache_path=Path(folder)/'cache.sqlite') as backend:
                backend.preflight(1.3e-6)
                key=backend._cache_identity(geometry(config),1.3e-6,reference_fingerprints=(),previous_kappa=None)
                backend.cache.put(key, ('cached','reference'))
                self.assertEqual(backend.evaluate_coefficients(geometry(config),1.3e-6),('cached','reference'))

    def test_native_raw_reuses_lower_guide_and_reselects(self):
        config=load_config(ROOT/'config/default.json')
        raw={'ok':1.,'error':'','attempts':1.,'n':2.,'items':[asdict(m) for m in fixture_modes()]}
        with tempfile.TemporaryDirectory() as folder, NativeLumapiBackend(config,cache_path=Path(folder)/'cache.sqlite') as backend:
            requests=[backend._raw_request(geometry(config,width=w,offset=d),1.3e-6,False,True)
                      for w,d in ((1e-6,0.),(1.1e-6,1e-6))]
            self.assertEqual(backend._raw_key(requests[0]),backend._raw_key(requests[1]))
            with patch.object(backend,'_call_native',return_value={'n':1.,'items':[raw]}) as call:
                backend._raw_packets(requests)
                backend._native_memory.clear()
                backend._raw_packets(requests)
                self.assertEqual(call.call_count,1)
            compact=dict(requests[0],compact=1.)
            self.assertNotEqual(backend._raw_key(compact),backend._raw_key(requests[0]))


class CharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        base=load_config(ROOT/'config/default.json')
        self.config=replace(base,project_root=self.root,
            materials=replace(base.materials,silicon_data_csv=str(ROOT/'data/green_si_2008_1260_1340.csv')),
            device=replace(base.device,z_samples_initial=3,wavelength_samples_initial=3,z_samples_max=9,wavelength_samples_max=9),
            optimizer=replace(base.optimizer,max_function_evaluations_per_grid=1,max_grid_refinements=0),
            endpoint_search=replace(base.endpoint_search,adaptive_max_widths=13))
        self.bank=build_material_bank(self.config)
        self.endpoint=search_endpoints_adaptive(SyntheticBackend(self.config),self.config,self.bank,output_dir=self.root/'ep')
        self.solution=optimize_profiles(RecordingBackend(self.config),self.config,self.bank,self.endpoint)

    def test_broadband_retains_checked_joint_grid_and_reuses_checkpoint(self):
        backend=RecordingBackend(self.config)
        result=characterize_profile(backend,self.config,self.bank,self.endpoint,self.solution)
        self.assertEqual(result['validation']['status'],'passed')
        self.assertEqual((len(result['wavelengths_m']),len(result['z_m'])),(5,5))
        self.assertEqual(len(self.solution.wavelengths_m),2)
        self.assertFalse(result['validation']['profile_optimized'])
        with patch.object(backend,'evaluate_coefficients',side_effect=AssertionError('completed characterization repeated')):
            repeated=characterize_profile(backend,self.config,self.bank,self.endpoint,self.solution)
        self.assertEqual(result['sha256'],repeated['sha256'])
        bar=self.solution.cross_coefficients
        write_artifact(self.root,3,self.config,self.endpoint,solution=self.solution,bar=bar,characterization=result,
                       validation={'status':'passed','upstream_validation':{'status':'review'}})
        loaded=load_block(self.root,3,self.config)[3]
        self.assertEqual(loaded['validation']['overall_status'],'review')
        self.assertEqual(loaded['payload']['characterization']['sha256'],result['sha256'])

    def test_axis_limit_cannot_claim_sampling_convergence(self):
        limited=replace(self.config,device=replace(self.config.device,z_samples_max=3,wavelength_samples_max=3))
        result=characterize_profile(RecordingBackend(limited),limited,self.bank,self.endpoint,self.solution)
        self.assertEqual(result['validation']['status'],'review')
        self.assertFalse(result['validation']['sampling_converged'])

    def test_width_search_resume_reuses_table_but_freshly_verifies(self):
        directory=self.root/'resume'
        first=SyntheticBackend(self.config)
        search_endpoints_adaptive(first,self.config,self.bank,output_dir=self.root/'one',checkpoint_dir=directory)
        second=SyntheticBackend(self.config)
        search_endpoints_adaptive(second,self.config,self.bank,output_dir=self.root/'two',checkpoint_dir=directory)
        self.assertFalse(any(call[3] for call in second.calls))
        self.assertEqual(len(second.calls),10)

    def test_powell_warm_restart_preserves_endpoint_controls(self):
        backend=RecordingBackend(self.config)
        optimize_profiles(backend,self.config,self.bank,self.endpoint,checkpoint_dir=self.root/'powell')
        solution=optimize_profiles(backend,self.config,self.bank,self.endpoint,checkpoint_dir=self.root/'powell')
        self.assertEqual(solution.w1_control_m[0],self.endpoint.w1_input_m)
        self.assertEqual(solution.w1_control_m[-1],self.endpoint.w1_output_m)

    def test_characterization_propagates_dense_dataset_offline(self):
        from vlac.blocks import run_propagation
        from vlac.block_io import write_json
        result=characterize_profile(RecordingBackend(self.config),self.config,self.bank,self.endpoint,self.solution)
        def fake_export(output,*args,**kwargs):
            write_json(Path(output)/'metadata.json',{})
        with patch('vlac.blocks.export_run',side_effect=fake_export):
            artifact=run_propagation(self.config,self.bank,self.endpoint,self.solution,self.solution.cross_coefficients,
                self.root/'performance',characterization=result,source={'validation':{'status':'review'}})
        self.assertEqual(len(artifact['payload']['performance']),4*len(result['wavelengths_m']))
        self.assertEqual(len(artifact['payload']['solution']['wavelengths_m']),2)
        self.assertEqual(artifact['validation']['overall_status'],'review')

    def test_process_sensitivity_preserves_nominal_controls_and_resumes(self):
        from vlac.sensitivity import run, load_cases
        from vlac.block_io import write_json
        write_artifact(self.root/'profile',2,self.config,self.endpoint,solution=self.solution,validation={'status':'review'})
        cases={'source':'SYNTHETIC TEST tolerance source; not a real process',
               'cases':[{'name':'nominal'},{'name':'width_case','upper_width_bias_m':1e-9}]}
        write_json(self.root/'cases.json',cases)
        cases=load_cases(self.root/'cases.json')
        controls=self.solution.w1_control_m.copy()
        result=run(self.config,self.root/'profile',cases,self.root/'study',RecordingBackend)
        np.testing.assert_array_equal(self.solution.w1_control_m,controls)
        self.assertEqual(len(result['cases']),2)
        self.assertTrue(all(not c['reoptimized'] for c in result['cases']))
        with patch.object(RecordingBackend,'evaluate_coefficients',side_effect=AssertionError('completed case repeated')):
            repeated=run(self.config,self.root/'profile',cases,self.root/'study',RecordingBackend)
        self.assertEqual(result['sha256'],repeated['sha256'])

    def test_shared_propagation_matches_independent_launch_stopping(self):
        z=np.linspace(0,1e-3,7)
        upper=1e7+100*np.cos(z*10000)+3j
        lower=1e7-80*np.cos(z*10000)+5j
        coupling=300+20*np.sin(z*10000)
        settings=replace(self.config.propagation,rk4_steps_initial=8,rk4_steps_max=512)
        both=propagate_both(z,upper,lower,coupling,settings)
        for launch in (0,1):
            separate=propagate_converged(z,upper,lower,coupling,launch_layer=launch,settings=settings)
            self.assertEqual(both[launch].steps,separate.steps)
            np.testing.assert_allclose(both[launch].output_powers,separate.output_powers,rtol=1e-12,atol=1e-14)


if __name__=='__main__':unittest.main()
