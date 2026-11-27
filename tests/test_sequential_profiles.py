"""Sequential profile regression tests. All optical data are SYNTHETIC."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vlac.config import load_config
from vlac.endpoint import EndpointSearchResult
from vlac.geometry import ClampedCubicProfile
from vlac.materials import build_material_bank
from vlac.modes import ModeSnapshot, extract_coupling
from vlac.monotone import linear_coordinates
from vlac.optimize import optimize_profiles
from vlac.profiles import minimax_objective
from vlac.profile_planning import ProfilePlanner
from vlac.sequential import fit_width_profile, isolated_detuning_grid
from vlac.tracking import CandidateRejected, EndpointConsistencyError
from vlac.block_io import physics_signature, write_artifact, load_block, validate_solution
from vlac.blocks import profile_report, run_bar, check_bar_grid
from vlac.native_mode import prepare_bundle

ROOT = Path(__file__).resolve().parents[1]


class SyntheticBackend:
    """Analytic test modes only; no MODE import, license or physical claims."""
    def __init__(self):
        self.require_endpoint_consistency = Mock()
        self.isolated_calls = []
        self.profile_calls = []
        self.bad_overlap = False
        self.interior_wavelength_error = False
        self.imaginary_kappa = 0.

    def solve_isolated_mode(self, geometry, wavelength, *, guide, reference, compact_x):
        assert not compact_x
        self.isolated_calls.append((geometry, wavelength, guide, reference))
        beta = 1e7 + ((geometry.w1_m-1.3e-6)/.5e-6*2000 if guide == 'upper' else 0.)
        mode = ModeSnapshot(beta*wavelength/(2*np.pi), np.ones((2,2)), np.zeros((2,2)),
                            1., .9, 'SYNTHETIC '+guide)
        return mode, .1 if self.bad_overlap else 1.

    def evaluate_profile_grid(self, config, bank, endpoint, w, dc, z, wavelengths, **kwargs):
        self.profile_calls.append((w.copy(), dc.copy(), np.asarray(z), np.asarray(wavelengths)))
        width = ClampedCubicProfile(config.device.length_m, w)(z)
        offset = ClampedCubicProfile(config.device.length_m, dc)(z)
        delta = (width-1.3e-6)/.5e-6*2000
        result = []
        for wavelength in wavelengths:
            kappa = 1000*np.exp(-(offset/2e-6)**2)+1j*self.imaginary_kappa
            if self.interior_wavelength_error and config.device.wavelength_min_m < wavelength < config.device.wavelength_max_m:
                kappa = kappa+500
            row = []
            for detuning, coupling in zip(delta, kappa):
                half_split = np.sqrt((detuning/2)**2+coupling**2+0j)
                row.append(extract_coupling(1e7+detuning,1e7,
                    1e7+detuning/2+half_split,1e7+detuning/2-half_split))
            result.append(row)
        return result


def one_evaluation(objective, initial, **kwargs):
    objective(initial)
    return SimpleNamespace(x=np.asarray(initial), success=True, message='SYNTHETIC one-evaluation optimizer')


class SequentialTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'config/native_sequential.json')
        self.config.optimizer.max_function_evaluations_per_grid = 6
        self.config.optimizer.max_grid_refinements = 0
        self.bank = build_material_bank(self.config)
        self.endpoint = EndpointSearchResult(w1_input_m=.8e-6,w1_output_m=1.8e-6,
            delta_beta_max_rad_per_m=2000.,h_gap_m=1e-6,h_rib_m=175e-9,
            wavelengths_m=np.array([1.26e-6,1.34e-6]),endpoint_records=[],candidate_records=[],
            converged=False,refinement_history=[],isolated_mode_domain={})
        for end, width, offset, delta in (('input',.8e-6,-8e-6,-2000.),('output',1.8e-6,8e-6,2000.)):
            for state in ('cross','bar'):
                for wavelength in self.endpoint.wavelengths_m:
                    detuning = delta if state == 'cross' else 4000.
                    self.endpoint.endpoint_records.append({'end':end,'state':state,'wavelength_m':float(wavelength),
                        'w1_m':width,'d_c2c_m':offset,'h_rib_m':175e-9,'h_gap_m':1e-6,
                        'beta_upper_real_rad_per_m':1e7+detuning,'beta_lower_real_rad_per_m':1e7,
                        'delta_beta_real_rad_per_m':detuning,'beta_upper_imag_rad_per_m':0.,
                        'beta_lower_imag_rad_per_m':0.,'delta_beta_imag_rad_per_m':0.,
                        'upper_tracking_overlap':1.,'lower_tracking_overlap':1.,
                        'upper_te_fraction':1.,'lower_te_fraction':1.})
        self.backend = SyntheticBackend()

    def optimize_once(self, **kwargs):
        with patch('vlac.sequential.minimize', side_effect=one_evaluation), \
             patch('vlac.optimize.minimize', side_effect=one_evaluation):
            return optimize_profiles(self.backend,self.config,self.bank,self.endpoint,**kwargs)

    def test_width_stage_uses_only_isolated_full_domain_modes_and_reduces_error(self):
        z = np.linspace(0,self.config.device.length_m,9)
        waves = np.array([1.26e-6,1.34e-6])
        initial = linear_coordinates(6)
        kwargs = dict(planner=ProfilePlanner(None),validation_samples=50)
        _, baseline = fit_width_profile(self.backend,self.config,self.bank,self.endpoint,z,waves,
            initial,max_evaluations=1,**kwargs)
        width, fitted = fit_width_profile(self.backend,self.config,self.bank,self.endpoint,z,waves,
            initial,max_evaluations=120,**kwargs)
        self.assertLess(fitted['objective'],baseline['objective'])
        self.assertLessEqual(fitted['evaluations'],120)
        self.assertEqual(self.backend.profile_calls,[])
        self.assertTrue(all(call[0].d_c2c_m == 0. for call in self.backend.isolated_calls))
        self.assertTrue(np.all(np.diff(width)>=0))
        np.testing.assert_array_equal(width[[0,-1]],[.8e-6,1.8e-6])

    def test_stages_freeze_width_share_budget_and_check_z_at_band_edges(self):
        limits = []
        def step(objective, initial, **kwargs):
            limits.append(kwargs['options']['maxfev'])
            return one_evaluation(objective, initial, **kwargs)
        with patch('vlac.sequential.minimize',side_effect=step),patch('vlac.optimize.minimize',side_effect=step):
            result = optimize_profiles(self.backend,self.config,self.bank,self.endpoint)
        self.assertEqual(limits,[3,3])
        for width, offset, _, _ in self.backend.profile_calls:
            np.testing.assert_array_equal(width,result.w1_control_m)
            self.assertTrue(np.all(np.diff(offset)>=0))
            np.testing.assert_array_equal(offset[[0,-1]],[-8e-6,8e-6])
        self.assertEqual([(len(z),len(w)) for _,_,z,w in self.backend.profile_calls],[(5,2),(9,2)])
        history = result.optimizer_history[-1]
        self.assertEqual(history['evaluations'],2)
        self.assertTrue(history['width_frozen_during_offset_fit'])
        expected = minimax_objective(result.cross_coefficients,result.z_m,self.config.device.length_m,2000)[0]
        self.assertAlmostEqual(result.objective,expected)
        self.assertEqual(result.grid_convergence_history[-1]['wavelength_check'],'band_edges_only_no_interior_check')

    def test_standard_configs_execute_width_then_offset_by_default(self):
        from vlac.cli import DEFAULT_CONFIG
        self.assertEqual(DEFAULT_CONFIG.resolve(), (ROOT/'config/default.json').resolve())
        for name in ('default.json', 'native.json'):
            with self.subTest(config=name):
                self.config = load_config(ROOT/'config'/name)
                self.config.optimizer.max_function_evaluations_per_grid = 6
                self.config.optimizer.max_grid_refinements = 0
                self.bank = build_material_bank(self.config)
                self.backend = SyntheticBackend()
                result = self.optimize_once()
                self.assertTrue(self.backend.isolated_calls)
                self.assertTrue(self.backend.profile_calls)
                self.backend.require_endpoint_consistency.assert_called_once()
                history = result.optimizer_history[-1]
                self.assertEqual(history['method'], 'sequential_width_offset_v1')
                self.assertTrue(history['width_frozen_during_offset_fit'])
                self.assertEqual(history['width_stage']['coupled_solves_for_width_fit'], 0)

    def test_explicit_joint_configs_preserve_standalone_and_legacy_loading(self):
        from vlac.config import config_from_dict
        for name, engine in (('python_joint.json', 'python'), ('native_joint.json', 'native')):
            with self.subTest(config=name), tempfile.TemporaryDirectory() as folder:
                config = load_config(ROOT/'config'/name)
                self.assertEqual(config.optimizer.method, 'bounded_powell')
                self.assertEqual(config.solver.engine, engine)
                prepare_bundle(config, folder)
        legacy = config_from_dict({}, project_root=ROOT)
        self.assertEqual(legacy.optimizer.method, 'bounded_powell')
        self.assertEqual(legacy.solver.engine, 'python')

    def test_full_complex_kappa_residual_is_preserved(self):
        self.backend.imaginary_kappa = 200.
        result = self.optimize_once()
        target = result.kappa_target_rad_per_m
        expected = max(abs(c.kappa-target[j])/1000 for row in result.cross_coefficients for j,c in enumerate(row))
        self.assertAlmostEqual(result.optimizer_history[-1]['offset_objective'],expected)

    def test_no_interior_wavelengths_are_evaluated(self):
        self.backend.interior_wavelength_error = True
        result = self.optimize_once()
        expected = [self.config.device.wavelength_min_m,self.config.device.wavelength_max_m]
        for _,_,_,waves in self.backend.profile_calls:
            np.testing.assert_array_equal(waves,expected)
        self.assertIsNone(result.grid_convergence_history[-1]['wavelength_coefficient_interpolation_error'])

    def test_bad_isolated_modes_cannot_start_offset_fit(self):
        self.backend.bad_overlap = True
        with self.assertRaisesRegex(RuntimeError,'No valid isolated-mode width candidate'):
            self.optimize_once()
        self.assertEqual(self.backend.profile_calls,[])

    def test_fixed_endpoint_gate_remains_mandatory(self):
        self.backend.require_endpoint_consistency.side_effect = EndpointConsistencyError('SYNTHETIC negative_kappa_squared')
        with self.assertRaises(EndpointConsistencyError):
            self.optimize_once()
        self.assertEqual(self.backend.isolated_calls,[])
        self.assertEqual(self.backend.profile_calls,[])

    def test_tracking_recovery_never_advances_rejected_isolated_mode(self):
        original = self.backend.solve_isolated_mode
        rejected, references = [], []
        def limited(geometry, wavelength, **kwargs):
            mode, overlap = original(geometry,wavelength,**kwargs)
            reference = kwargs['reference']
            if kwargs['guide'] == 'upper':
                references.append(reference)
                if reference is not None and abs(mode.neff-reference.neff)*2*np.pi/wavelength>1100:
                    rejected.append(mode)
                    return mode,.1
            return mode,overlap
        self.backend.solve_isolated_mode = limited
        delta = isolated_detuning_grid(self.backend,self.config,self.bank,self.endpoint,
            np.linspace(.8e-6,1.8e-6,6),np.linspace(0,self.config.device.length_m,3),[1.26e-6])
        self.assertTrue(rejected)
        self.assertFalse(any(ref is bad for ref in references for bad in rejected))
        np.testing.assert_allclose(delta.real,[[-2000.,0.,2000.]],atol=1e-8)

    def test_summary_and_versioned_handoff_roundtrip_enforce_monotonicity(self):
        result = self.optimize_once()
        report = profile_report(result,self.endpoint,self.config)
        self.assertEqual(report['optimizer_method'],'sequential_width_offset_v1')
        self.assertEqual(len(report['sequential_stages']),1)
        self.assertIn('interior wavelengths not checked',report['wavelength_validation'])
        with tempfile.TemporaryDirectory() as folder:
            write_artifact(folder,2,self.config,self.endpoint,solution=result,validation=report)
            _, restored, _, _ = load_block(folder,2,self.config)
        np.testing.assert_array_equal(restored.w1_control_m,result.w1_control_m)
        self.assertEqual(restored.optimizer_history[0]['width_stage'],result.optimizer_history[0]['width_stage'])
        restored.w1_control_m[1] = restored.w1_control_m[-1]
        with self.assertRaisesRegex(ValueError,'monotone'):
            validate_solution(restored,self.endpoint,self.config)

    def test_resume_reuses_completed_width_stage(self):
        self.config.execution.resume = True
        with tempfile.TemporaryDirectory() as folder:
            first = self.optimize_once(checkpoint_dir=folder)
            before = len(self.backend.isolated_calls)
            second = self.optimize_once(checkpoint_dir=folder)
        self.assertEqual(len(self.backend.isolated_calls),before)
        np.testing.assert_array_equal(first.w1_control_m,second.w1_control_m)

    def test_sequential_identity_and_supported_execution_boundary(self):
        signature = physics_signature(self.config,2)
        self.config.optimizer.method = 'monotone_powell'
        self.assertNotEqual(signature,physics_signature(self.config,2))
        self.config.optimizer.method = 'sequential_width_offset_v1'
        with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):
            prepare_bundle(self.config,folder)
        self.config.solver.tracking_method = 'legacy'
        with self.assertRaisesRegex(ValueError,'guarded native'):
            self.config.validate()

    def test_pilot_preparer_defaults_to_sequential_and_preserves_optics(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'new_pilot'
            argv = ['prepare_tracking_pilot.py','--baseline',str(ROOT/'config/native.json'),'--out',str(out)]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT/'tools/prepare_tracking_pilot.py'),run_name='__main__')
            cfg = load_config(out/'pilot_config.json')
            plan = json.loads((out/'pilot_plan.json').read_text())
            self.assertEqual(cfg.optimizer.method,'sequential_width_offset_v1')
            self.assertTrue(plan['optical_settings_preserved'])
            self.assertEqual(cfg.optimizer.max_function_evaluations_per_grid,6)
            self.assertEqual(cfg.execution.max_solves,120)
            self.assertEqual(cfg.execution.max_seconds,2700)

    def test_block3_characterization_stays_at_endpoints_with_longitudinal_checks(self):
        solution = self.optimize_once()
        self.backend.profile_calls.clear()
        self.config.characterization.enabled = True
        self.config.characterization.max_refinements = 0
        with tempfile.TemporaryDirectory() as folder:
            self.config.output.cache_sqlite = str(Path(folder)/'cache/fde.sqlite')
            with patch('vlac.blocks.export_profiles',return_value={}):
                _,artifact = run_bar(self.backend,self.config,self.bank,self.endpoint,solution,folder)
            retained = artifact['payload']['characterization']
        self.assertEqual(retained['method'],'direct_nested_z_band_edges_v1')
        self.assertEqual(retained['wavelengths_m'],[1.26e-6,1.34e-6])
        self.assertGreater(len(retained['z_m']),len(solution.z_m))
        for _,_,_,wavelengths in self.backend.profile_calls:
            np.testing.assert_array_equal(wavelengths,[1.26e-6,1.34e-6])

    def test_block3_rejects_interior_wavelength_input_before_optical_work(self):
        solution = self.optimize_once()
        solution.wavelengths_m = np.array([1.26e-6,1.3e-6,1.34e-6])
        self.backend.profile_calls.clear()
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'exactly the two'):
            run_bar(self.backend,self.config,self.bank,self.endpoint,solution,folder)
        self.assertEqual(self.backend.profile_calls,[])

    def test_block3_z_validation_does_not_add_wavelengths(self):
        solution = self.optimize_once()
        bar = self.backend.evaluate_profile_grid(self.config,self.bank,self.endpoint,
            solution.w1_control_m,solution.d_c2c_control_m,solution.z_m,solution.wavelengths_m)
        self.backend.profile_calls.clear()
        report = check_bar_grid(self.backend,self.config,self.bank,self.endpoint,solution,bar)
        self.assertIsNone(report['wavelength_coefficient_interpolation_error'])
        self.assertEqual([(len(z),len(w)) for _,_,z,w in self.backend.profile_calls],[(9,2)])

    def test_automatic_block0_verification_requests_endpoint_only_scope(self):
        from vlac.block0 import require_design_verification
        solution = self.optimize_once()
        self.config.solver.calibration = {'certificate_id':'SYNTHETIC_TEST_CERTIFICATE'}
        with tempfile.TemporaryDirectory() as folder:
            self.config.output.cache_sqlite = str(Path(folder)/'cache/fde.sqlite')
            with patch('vlac.block0.verify_design',return_value={'status':'passed','checks':[]}) as verify:
                report = require_design_verification(self.config,folder,self.endpoint,solution)
        self.assertTrue(verify.call_args.kwargs['endpoint_wavelengths_only'])
        self.assertEqual(report['wavelength_policy'],'endpoints_only')

    def test_native_block3_contract_disables_wavelength_midpoints(self):
        source = (ROOT/'src/vlac/native/vlac_blocks.lsf').read_text(encoding='utf-8')
        block3 = source[source.index('function vlac_block3'):]
        self.assertIn('checks=vlac_grid_checks(cfg,request,grid,0)',block3)
        self.assertIn('block3_requires_endpoint_wavelengths_only',block3)

    def test_equal_best_width_coordinates_do_not_falsely_clear_termination_flag(self):
        # Synthetic constant-width problem: all coordinates describe the same
        # physical geometry and objective, so Powell may return another tie.
        self.endpoint.w1_input_m = self.endpoint.w1_output_m = 1.3e-6
        def tied(objective,initial,**kwargs):
            objective(initial)
            final = np.clip(initial+.01,0,1)
            return SimpleNamespace(x=final,fun=objective(final),success=True,message='SYNTHETIC tie')
        with patch('vlac.sequential.minimize',side_effect=tied):
            _,report = fit_width_profile(self.backend,self.config,self.bank,self.endpoint,
                np.linspace(0,self.config.device.length_m,5),[1.26e-6,1.34e-6],linear_coordinates(6),
                max_evaluations=2,planner=ProfilePlanner(None),validation_samples=18)
        self.assertTrue(report['success'])



if __name__ == '__main__':
    unittest.main()
