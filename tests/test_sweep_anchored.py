"""Sweep-anchor tests use SYNTHETIC optics only; no physical validation claims."""
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from vlac import cli
from vlac.config import load_config
from vlac.geometry import AnchoredProfile, solution_profiles
from vlac.materials import build_material_bank
from vlac.modes import extract_coupling
from vlac.profiles import evaluate_design_profiles
from vlac.optimize import optimize_profiles, _check_fabrication_constraints
from vlac.sweep_anchored import monotone_path, interpolate_samples, coupling_row
from vlac.block_io import write_artifact, load_block, validate_solution, geometry_data, digest
from vlac.blocks import profile_report, run_bar
from vlac.characterize import characterize_profile
from vlac.tracking import CandidateRejected
import test_sequential_profiles as sequential_fixtures

ROOT=Path(__file__).resolve().parents[1]


class SweepBackend:
    """Analytic linear detuning and width-dependent tent coupling, SYNTHETIC."""
    backend_name='SYNTHETIC sweep backend'
    def __init__(self):
        self.require_endpoint_consistency=Mock()
        self.preflight=Mock()
        self.bare=sequential_fixtures.SyntheticBackend()
        self.calls=[]
        self.imaginary=0.
        self.reject=False
        self.curved=False
    def solve_isolated_mode(self,*args,**kwargs):
        return self.bare.solve_isolated_mode(*args,**kwargs)
    def evaluate_coefficients(self,g,wavelength,**kwargs):
        self.calls.append((g,wavelength,kwargs))
        if self.reject and abs(g.d_c2c_m)<1e-7:
            raise CandidateRejected('SYNTHETIC missing sample')
        delta=(g.w1_m-1.3e-6)/.5e-6*2000 if g.state=='cross' else 4000.
        shape=max(0.,1-abs(g.d_c2c_m)/8e-6)
        if self.curved: shape=shape**2
        coupling=(1000+200*(g.w1_m-1.3e-6)/.5e-6)*shape+1j*self.imaginary*shape
        half=np.sqrt((delta/2)**2+coupling**2+0j)
        c=extract_coupling(1e7+delta,1e7,1e7+delta/2+half,1e7+delta/2-half)
        return c,SimpleNamespace(valid=True)


class SweepTests(unittest.TestCase):
    def setUp(self):
        fixture=sequential_fixtures.SequentialTests();fixture.setUp()
        self.endpoint=fixture.endpoint
        self.config=load_config(ROOT/'config/native_sweep.json')
        self.config.optimizer.sweep_width_samples=5
        self.config.optimizer.sweep_offset_samples=5
        self.config.optimizer.sweep_samples_max=17
        self.config.optimizer.sweep_interpolation_subdivisions=8
        self.config.optimizer.max_grid_refinements=0
        self.config.device.grid_convergence_rtol=.3
        self.bank=build_material_bank(self.config)
        self.backend=SweepBackend()
    def run_sweep(self,**kwargs):
        with patch('vlac.optimize.minimize',side_effect=AssertionError('Powell called')):
            return optimize_profiles(self.backend,self.config,self.bank,self.endpoint,**kwargs)
    def test_native_cache_distinguishes_anchor_coordinates(self):
        from vlac.native_backend import NativeLumapiBackend
        with tempfile.TemporaryDirectory() as folder:
            with NativeLumapiBackend(self.config,cache_path=Path(folder)/'cache.sqlite') as backend:
                with patch('vlac.profiles._profile_coefficients',return_value=[]) as evaluate:
                    args=(self.config,self.bank,self.endpoint,np.linspace(.8e-6,1.8e-6,4),
                          np.linspace(-8e-6,8e-6,4),np.linspace(0,.0015,5),[1.26e-6,1.34e-6])
                    for z in ([0,.0002,.001,.0015],[0,.0002,.001,.0015],[0,.0003,.001,.0015]):
                        backend.evaluate_profile_grid(*args,state='cross',anchor_z_m=z)
                    self.assertEqual(evaluate.call_count,2)
    def test_interrupted_sweep_reuses_completed_rows(self):
        original=self.backend.evaluate_coefficients
        counter=0
        def interrupted(*args,**kwargs):
            nonlocal counter
            counter+=1
            if counter==11: raise RuntimeError('SYNTHETIC interruption')
            return original(*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(self.backend,'evaluate_coefficients',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    self.run_sweep(checkpoint_dir=folder)
            before=len(self.backend.calls)
            resumed=self.run_sweep(checkpoint_dir=folder)
            resumed_calls=len(self.backend.calls)-before
            self.backend=SweepBackend()
            cold=self.run_sweep()
            self.assertLess(resumed_calls,len(self.backend.calls))
            np.testing.assert_array_equal(resumed.d_c2c_control_m,cold.d_c2c_control_m)
    def test_alignment_diagnostic_uses_pchip_positions_in_metres(self):
        from vlac.lower_state_check import alignment_roots
        solution=self.run_sweep()
        roots=alignment_roots(self.config,solution)
        self.assertGreater(len(roots),0)
        offset=solution_profiles(self.config,solution)[1]
        np.testing.assert_allclose(offset(roots),0,atol=1e-18)
    def test_legacy_schema_two_spline_handoff_still_loads(self):
        fixture=sequential_fixtures.SequentialTests();fixture.setUp()
        solution=fixture.optimize_once()
        with tempfile.TemporaryDirectory() as folder:
            artifact=write_artifact(folder,2,fixture.config,fixture.endpoint,solution=solution)
            for name in ('profile_kind','anchor_z_m','sweep_report'):
                artifact['payload']['solution'].pop(name)
            artifact['schema_version']=2
            artifact['content_id']=digest({'physics':artifact['physics_signature'],'payload':artifact['payload']})
            artifact.pop('sha256');artifact['sha256']=digest(artifact)
            Path(folder,'block2_result.json').write_text(json.dumps(artifact))
            _,loaded,_,_=load_block(folder,2,fixture.config)
            self.assertEqual(loaded.profile_kind,'clamped_cubic_v1')
            self.assertIsNone(loaded.anchor_z_m)
    def test_default_and_explicit_config_select_sweep(self):
        for name in ('default.json','native.json','native_sweep.json'):
            self.assertEqual(load_config(ROOT/'config'/name).optimizer.method,'sweep_anchored_v1')
    def test_offset_cap_includes_extra_aligned_sample(self):
        self.config.optimizer.sweep_width_samples=3
        self.config.optimizer.sweep_offset_samples=4
        self.config.optimizer.sweep_samples_max=4
        with self.assertRaisesRegex(ValueError,'aligned sample'):
            self.config.validate()
    def test_coefficient_stability_tool_retains_saved_anchor_coordinates(self):
        import runpy
        import sys
        solution=self.run_sweep()
        with patch.object(sys,'path',[str(ROOT/'tools'),*sys.path]):
            module=runpy.run_path(str(ROOT/'tools/check_block2_coefficient_stability.py'))
        with tempfile.TemporaryDirectory() as folder:
            write_artifact(folder,2,self.config,self.endpoint,solution=solution)
            w,d,z,source=module['frozen_profiles'](self.config,self.endpoint,Path(folder))
        np.testing.assert_array_equal(z,solution.anchor_z_m)
        np.testing.assert_array_equal(w,solution.w1_control_m)
        np.testing.assert_array_equal(d,solution.d_c2c_control_m)
    def test_index_sensitivity_uses_anchors_and_propagates_coefficients(self):
        solution=self.run_sweep()
        self.config.materials.sin_index_sensitivity_offsets=[0.]
        with tempfile.TemporaryDirectory() as folder:
            with patch('vlac.cli._backend',return_value=contextlib.nullcontext(self.backend)):
                cli._run_index_sensitivity(self.config,self.endpoint,solution,Path(folder),None)
            self.assertTrue(Path(folder,'sin_index_sensitivity.csv').is_file())
    def test_linear_sweeps_anchor_geometry_without_powell_and_keep_two_wavelengths(self):
        solution=self.run_sweep()
        self.assertEqual(solution.profile_kind,'pchip_anchors_v1')
        w,d=solution_profiles(self.config,solution)
        np.testing.assert_array_equal(w(solution.anchor_z_m),solution.w1_control_m)
        np.testing.assert_array_equal(d(solution.anchor_z_m),solution.d_c2c_control_m)
        np.testing.assert_array_equal(solution.w1_control_m[[0,-1]],[.8e-6,1.8e-6])
        np.testing.assert_array_equal(solution.d_c2c_control_m[[0,-1]],[-8e-6,8e-6])
        self.assertTrue(np.all(np.diff(solution.w1_control_m)>=0))
        self.assertTrue(np.all(np.diff(solution.d_c2c_control_m)>=0))
        self.assertEqual({x[1] for x in self.backend.calls},{1.26e-6,1.34e-6})
        first=self.backend.bare.isolated_calls[:6]
        widths=[x[0].w1_m for x in first if x[2]=='upper']
        np.testing.assert_allclose(widths,np.linspace(.8e-6,1.8e-6,5))
        validate_solution(solution,self.endpoint,self.config)
        self.assertTrue(solution.optimizer_history[-1]['width_frozen_during_offset_fit'])
    def test_dp_minimax_matches_exhaustive_paths_and_fixed_endpoints(self):
        rng=np.random.default_rng(35)
        cost=rng.uniform(size=(5,6));candidates=np.arange(6.)
        indices=monotone_path(cost,candidates,np.arange(5.))
        actual=max(cost[i,j] for i,j in enumerate(indices))
        exact=min(max(cost[i,j] for i,j in enumerate((0,*path,5)))
                  for path in itertools.combinations_with_replacement(range(6),3))
        self.assertAlmostEqual(actual,exact)
        cost[2,:]=np.inf
        with self.assertRaisesRegex(ValueError,'No valid'):
            monotone_path(cost,candidates,np.arange(5.))
    def test_complex_residual_is_not_magnitude_fit(self):
        self.backend.imaginary=400.
        solution=self.run_sweep()
        expected=max(abs(c.kappa-solution.kappa_target_rad_per_m[j])/1000
                     for row in solution.cross_coefficients for j,c in enumerate(row))
        self.assertAlmostEqual(solution.optimizer_history[-1]['offset_target_error'],expected)
        self.assertGreater(expected,.1)
    def test_invalid_samples_are_holes_not_zero_or_bridged(self):
        values=np.array([[0.,np.nan,2.]],complex)
        actual=interpolate_samples([0.,1.,2.],values,[0.,.5,1.,1.5,2.])
        self.assertTrue(np.isnan(actual[0,1:4]).all())
        self.backend.reject=True
        row,rejected=coupling_row(self.backend,self.config,self.bank,self.endpoint,1.3e-6,
                                 np.linspace(-8e-6,8e-6,5),[1.26e-6,1.34e-6])
        self.assertEqual(rejected,2)
        self.assertTrue(np.isnan(row[:,2]).all())
    def test_independent_z_validation_can_request_refinement(self):
        self.config.device.grid_convergence_rtol=1e-7
        self.config.optimizer.max_grid_refinements=1
        solution=self.run_sweep()
        self.assertEqual(len(solution.grid_convergence_history),2)
        self.assertGreater(len(solution.anchor_z_m),5)
        self.assertFalse(solution.grid_converged)
    def test_coupling_table_disagreement_is_reported_and_refined(self):
        self.backend.curved=True
        self.config.device.grid_convergence_rtol=1e-5
        self.config.optimizer.max_grid_refinements=1
        solution=self.run_sweep()
        self.assertGreater(solution.optimizer_history[0]['coupling_table_error'],1e-5)
        self.assertGreater(solution.optimizer_history[-1]['offset_sweep_samples'],5)
    def test_resume_reuses_completed_sweep_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            first=self.run_sweep(checkpoint_dir=folder)
            before=len(self.backend.calls)
            second=self.run_sweep(checkpoint_dir=folder)
            # A converged checkpoint does no optical work after mandatory preflight.
            if first.grid_converged:
                self.assertEqual(len(self.backend.calls),before)
            else:
                self.assertLess(len(self.backend.calls)-before,before)
            np.testing.assert_array_equal(first.w1_control_m,second.w1_control_m)
        self.assertEqual(self.backend.require_endpoint_consistency.call_count,2)
    def test_handoff_roundtrip_preserves_anchor_coordinates_and_rejects_relabeling(self):
        solution=self.run_sweep()
        with tempfile.TemporaryDirectory() as folder:
            artifact=write_artifact(folder,2,self.config,self.endpoint,solution=solution,
                                    validation=profile_report(solution,self.endpoint,self.config))
            self.assertEqual(artifact['schema_version'],4)
            _,loaded,_,_=load_block(folder,2,self.config)
            np.testing.assert_array_equal(loaded.anchor_z_m,solution.anchor_z_m)
            self.assertEqual(geometry_data(self.config,self.endpoint,solution),geometry_data(self.config,self.endpoint,loaded))
            loaded.profile_kind='clamped_cubic_v1'
            with self.assertRaisesRegex(ValueError,'representation'):
                validate_solution(loaded,self.endpoint,self.config)
            artifact['schema_version']=2
            artifact.pop('sha256');artifact['sha256']=digest(artifact)
            Path(folder,'block2_result.json').write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError,'schema 4'):
                load_block(folder,2,self.config)
    def test_anchor_coordinates_are_independent_of_characterization_grid(self):
        solution=self.run_sweep()
        original=solution.anchor_z_m.copy()
        self.config.characterization.max_refinements=0
        with tempfile.TemporaryDirectory() as folder:
            self.config.output.cache_sqlite=str(Path(folder)/'cache.sqlite')
            result=characterize_profile(self.backend,self.config,self.bank,self.endpoint,solution,
                                        endpoint_wavelengths_only=True)
        self.assertGreater(len(result['z_m']),len(original))
        np.testing.assert_array_equal(solution.anchor_z_m,original)
    def test_bar_exports_actual_anchor_geometry(self):
        solution=self.run_sweep()
        self.config.characterization.enabled=False
        with tempfile.TemporaryDirectory() as folder:
            _,artifact=run_bar(self.backend,self.config,self.bank,self.endpoint,solution,folder)
            geometry=json.loads(Path(folder,'block2_geometry.json').read_text())
            self.assertEqual(geometry['anchor_z_m'],solution.anchor_z_m.tolist())
            self.assertNotIn('upper_width_controls_m',geometry)
            _,loaded,_,_=load_block(folder,3,self.config)
            np.testing.assert_array_equal(loaded.anchor_z_m,solution.anchor_z_m)
    def test_pchip_constraints_check_between_samples_and_both_sides_of_knots(self):
        profile=AnchoredProfile(1.,[0,.01,.5,1.],[0,.1,.1,.2])
        dense=profile.derivative(np.linspace(0,1,10001))
        self.assertGreaterEqual(profile.max_abs_derivative(1)+1e-12,np.max(abs(dense)))
        self.config.optimizer.maximum_slope_m_per_m=5.
        self.assertGreater(_check_fabrication_constraints(self.config,profile,profile),0)
        with self.assertRaises(ValueError): AnchoredProfile(1.,[0,0,1],[0,1,2])
    def test_cli_runs_and_validates_sweep_with_explicit_sample_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=root/'config.json'
            raw=self.config.as_dict();raw['project_root']=str(ROOT)
            config.write_text(json.dumps(raw))
            write_artifact(root/'endpoint',1,self.config,self.endpoint)
            for command in ('block2','validate-block2'):
                out=root/command
                with patch('vlac.cli._backend',return_value=contextlib.nullcontext(self.backend)),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    code=cli.main([command,'--config',str(config),'--endpoint-dir',str(root/'endpoint'),
                        '--out',str(out),'--sweep-width-samples','5','--sweep-offset-samples','5','--max-refinements','0'])
                self.assertIn(code,(0,3))
                summary=json.loads((out/'block2_summary.json').read_text())
                self.assertEqual(summary['profile_kind'],'pchip_anchors_v1')
                self.assertEqual((out/'block2_validation_report.json').exists(),command.startswith('validate'))


if __name__=='__main__':unittest.main()
