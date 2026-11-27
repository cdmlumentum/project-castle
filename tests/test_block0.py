"""Block 0 numerical/contract tests. ALL optical data here are synthetic.

No test launches MODE or establishes optical convergence of the real device.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import loadmat

from vlac import cli
from vlac import block0 as b0
from vlac.block_io import digest, physics_signature, write_json
from vlac.config import load_config, config_from_dict
from vlac.materials import build_material_bank
from vlac.native_backend import NativeLumapiBackend, native_script
from vlac.native_mode import config_payload, prepare_bundle
from vlac.solver import LumapiBackend

ROOT = Path(__file__).resolve().parents[1]


def synthetic_sample(kind, case, profile):
    # Orthonormal transverse fields; deliberately no physical solver model.
    one = np.zeros((2, 17, 17), dtype=complex)
    two = one.copy()
    one[0, 8, 6] = 1
    two[0, 8, 10] = 1
    fields = [one, two]
    neff = [1.7+(case['w1_m']-1.15e-6)*1e4, 1.7]
    if kind == 'coupled':
        fields += [(one+two)/np.sqrt(2), (one-two)/np.sqrt(2)]
        neff += [1.71, 1.69]
    value = {'neff': np.array(neff, dtype=complex), 'fields': np.array(fields),
             'te': np.ones(len(neff)), 'wavelength_m': case['wavelength_m'],
             'elapsed_seconds': float(profile.nx*profile.ny/1e5),
             'solves': 2 if kind == 'isolated' else 3, 'flag': 'ok',
             'delta_beta': complex((neff[0]-neff[1])*2*np.pi/case['wavelength_m']),
             'actual_mesh': [{'SYNTHETIC_TEST_ONLY': True}]}
    if kind == 'coupled':
        value['kappa'] = 100.+0j
    return value


class SyntheticEvaluator:
    synthetic = True
    instances = []

    def __init__(self, config):
        self.calls, self.closed = [], False
        self.instances.append(self)

    def __call__(self, kind, case, profile):
        self.calls.append((kind, case, profile))
        return synthetic_sample(kind, case, profile)

    def close(self):
        self.closed = True


def certificate_fixture(config):
    """Construct a temporary protocol fixture, never a production artifact."""
    profiles = {kind: asdict(replace(p, **b0.comparison_grid(config, p, kind)))
                for kind, p in b0.base_profiles(config).items()}
    value = {'schema': b0.SCHEMA, 'method': b0.METHOD, 'status': 'passed', 'trial': False,
             'synthetic': False, 'execution_completed': True, 'input_id': b0.input_identity(config),
             'implementation_id': b0.implementation_identity(), 'installation_id': b0.installation_identity(config),
             'profiles': profiles, 'reference_profiles': deepcopy(profiles),
             'assessments': {kind: {'status': 'passed'} for kind in b0.KINDS},
             'fixture_notice': 'SYNTHETIC TEST PROTOCOL FIXTURE; NOT CONVERGENCE EVIDENCE'}
    value['certificate_id'] = digest(value)
    return value


class BlockZeroTests(unittest.TestCase):
    def test_domain_reference_keeps_the_uniform_mesh_lattice_fixed(self):
        config = load_config(ROOT/'config/native.json')
        for kind, anchor in b0.base_profiles(config).items():
            dx = anchor.x_span_m/anchor.nx
            dy = (anchor.y_max_m-anchor.y_min_m)/anchor.ny
            for factor in config.convergence.reference_domain_factors:
                with self.subTest(kind=kind, factor=factor):
                    lateral = b0.domain_reference_profile(config, anchor, kind, 'domain_x', factor)
                    target = b0.margins(config, anchor, kind, factor)
                    self.assertEqual(lateral.nx % 2, anchor.nx % 2)
                    self.assertAlmostEqual(lateral.x_span_m/lateral.nx/dx, 1., places=13)
                    self.assertGreaterEqual(lateral.x_span_m+1e-18, target.x_span_m)
                    self.assertLess(lateral.x_span_m-target.x_span_m, 2*dx+1e-18)
                    for axis in ('domain_top', 'domain_bottom'):
                        profile = b0.domain_reference_profile(config, anchor, kind, axis, factor)
                        self.assertAlmostEqual((profile.y_max_m-profile.y_min_m)/profile.ny/dy, 1., places=13)
                        self.assertAlmostEqual((profile.y_min_m-anchor.y_min_m)/dy,
                                               round((profile.y_min_m-anchor.y_min_m)/dy), places=12)
                        self.assertGreaterEqual(profile.y_max_m+1e-18, target.y_max_m if axis=='domain_top' else anchor.y_max_m)
                        self.assertLessEqual(profile.y_min_m-1e-18, target.y_min_m if axis=='domain_bottom' else anchor.y_min_m)

    def test_integer_cell_counts_do_not_gain_roundoff_cells(self):
        profile = b0.SolverProfile(np.nextafter(204., np.inf), 0., np.nextafter(208., np.inf), 1., 1.)
        self.assertEqual((profile.nx, profile.ny), (204, 208))
        self.assertEqual(replace(profile, x_span_m=204.001).nx, 205)

    def test_native_script_error_is_reported_and_failed_attempts_are_budgeted(self):
        config = load_config(ROOT/'config/native.json')
        backend = Mock()
        backend.session.getv.return_value = 'unsupported mesh property at native line 103'
        evaluator = b0.NativeEvaluator(config, backend=backend)
        evaluator.functions_loaded = True
        profile = b0.base_profiles(config)['isolated']
        profile = replace(profile, **b0.comparison_grid(config, profile, 'isolated'))
        case = b0.test_cases(config)['isolated'][0]
        with self.assertRaisesRegex(RuntimeError, 'unsupported mesh property at native line 103'):
            evaluator('isolated', case, profile)
        backend._budget.return_value.check.assert_called_once_with(4)
        backend._budget.return_value.consume.assert_called_once_with(4)
        backend.session.getv.assert_called_once_with('vlac_sample_error')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        base = load_config(ROOT/'config/default.json')
        self.config = replace(base, convergence=replace(base.convergence, max_solves=5000, max_seconds=120,
            comparison_dx_m=2e-6, comparison_dy_m=1e-6, timing_repeats=1, reference_sentinels=2,
            reference_domain_factors=[1., 1.25, 1.5], reference_mesh_factors=[1., .75, .5],
            candidate_margin_factors=[1., .5], candidate_mesh_factors=[1., 1.5]))
        SyntheticEvaluator.instances = []
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temp.cleanup()

    def sample(self, kind='coupled'):
        case = b0.test_cases(self.config, trial=True)[kind][0]
        return synthetic_sample(kind, case, b0.base_profiles(self.config)[kind])

    def run_fake(self, *, config=None, resume=False, trial=True):
        return b0.run_calibration(config or self.config, self.out, trial=trial, resume=resume,
                                  evaluator_factory=SyntheticEvaluator)

    def test_default_bounds_and_distinct_profiles(self):
        cases = b0.test_cases(self.config)
        self.assertEqual(len(cases['isolated']), 18)
        self.assertEqual(len(cases['coupled']), 54)
        self.assertEqual({c['state'] for c in cases['coupled']}, {'cross', 'bar'})
        self.assertEqual(len({c['wavelength_m'] for c in cases['coupled']}), 3)
        profiles = b0.base_profiles(self.config)
        self.assertAlmostEqual(profiles['coupled'].x_span_m, 26e-6)
        self.assertAlmostEqual(profiles['isolated'].x_span_m, 10e-6)
        self.assertEqual(profiles['coupled'].dx_m, profiles['isolated'].dx_m)
        expanded = b0.margins(self.config, profiles['coupled'], 'coupled', 1.5)
        self.assertEqual(expanded.dx_m, profiles['coupled'].dx_m)
        self.assertGreater(expanded.nx, profiles['coupled'].nx)
        self.assertLess(expanded.y_min_m, -self.config.geometry.box_thickness_m)

    def test_invalid_budgets_and_reference_sequences_are_rejected(self):
        for updates in ({'max_solves': 0}, {'max_seconds': float('nan')},
                        {'reference_domain_factors': [1., .5, 2.]}, {'reference_mesh_factors': [1., .5]},
                        {'minimum_power_overlap': 1.}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(self.config, convergence=replace(self.config.convergence, **updates)).validate()

    def test_metrics_detect_phase_error_despite_small_neff_error(self):
        reference, actual = self.sample(), self.sample()
        actual['neff'][0] += 3e-6
        actual['delta_beta'] += 30.
        result = b0.compare_samples(self.config, actual, reference)
        self.assertLess(result['maximum_real_neff_error'], self.config.convergence.neff_atol)
        self.assertGreater(result['delta_beta_phase_error_rad'], self.config.convergence.phase_error_rad)
        self.assertFalse(result['passed'])

    def test_near_zero_coupling_uses_absolute_and_phase_limits(self):
        reference = self.sample(); reference['kappa'] = 0j
        actual = deepcopy(reference); actual['kappa'] = 1.9
        self.assertTrue(b0.compare_samples(self.config, actual, reference)['passed'])
        actual['kappa'] = 3.
        self.assertFalse(b0.compare_samples(self.config, actual, reference)['passed'])

    def test_imaginary_beta_and_field_shape_are_checked(self):
        reference, actual = self.sample(), self.sample()
        actual['neff'][0] += 1e-5j
        self.assertFalse(b0.compare_samples(self.config, actual, reference)['passed'])
        actual = deepcopy(reference); actual['fields'][0] = actual['fields'][1]
        self.assertFalse(b0.compare_samples(self.config, actual, reference)['passed'])
        actual = deepcopy(reference); actual['flag'] = 'negative_kappa_squared'
        self.assertFalse(b0.compare_samples(self.config, actual, reference)['passed'])

    def test_degenerate_pair_accepts_basis_rotation_but_not_wrong_subspace(self):
        reference = self.sample(); reference['neff'][2:] = [1.71, 1.71+1e-10]
        actual = deepcopy(reference)
        actual['fields'][2:] = reference['fields'][2:][::-1]
        self.assertTrue(b0.compare_samples(self.config, actual, reference)['passed'])
        actual['fields'][2:] = 0
        self.assertFalse(b0.compare_samples(self.config, actual, reference)['passed'])

    def test_trial_reports_both_states_without_publishing_profiles(self):
        report = self.run_fake()
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['execution_completed'])
        self.assertTrue(report['trial'])
        self.assertTrue(report['synthetic'])
        self.assertEqual(report['session_solves'], 30)
        self.assertEqual(len(SyntheticEvaluator.instances[0].calls), 12)
        self.assertTrue(SyntheticEvaluator.instances[0].closed)
        for name in ('certificate.json', 'report.html', 'convergence_results.csv', 'checkpoint.json', 'pilot.json', 'plan.json'):
            self.assertTrue((self.out/name).is_file())
        self.assertFalse((self.out/'solver_profiles.json').exists())
        with self.assertRaisesRegex(ValueError, 'passed, full, real'):
            b0.apply_certificate(self.config, self.out)

    def test_budget_stops_between_samples_and_resume_uses_checked_fields(self):
        small = replace(self.config, convergence=replace(self.config.convergence, max_solves=4))
        partial = self.run_fake(config=small)
        self.assertEqual(partial['status'], 'incomplete')
        self.assertFalse(partial['execution_completed'])
        self.assertEqual(partial['session_solves'], 2)
        complete = self.run_fake(resume=True)
        self.assertTrue(complete['execution_completed'])
        self.assertEqual(complete['session_solves'], 28)
        self.assertEqual(len(SyntheticEvaluator.instances[-1].calls), 11)

    def test_resume_rejects_changed_inputs_and_corrupted_sample(self):
        self.run_fake()
        changed = replace(self.config, geometry=replace(self.config.geometry, w1_max_m=2.1e-6))
        with self.assertRaisesRegex(ValueError, 'resume inputs'):
            self.run_fake(config=changed, resume=True)
        self.assertEqual(SyntheticEvaluator.instances[-1].calls, [])
        payload = next((self.out/'samples').glob('*.npz'))
        payload.write_bytes(b'corrupted checkpoint')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_fake(resume=True)

    def test_no_overwrite_and_lazy_license_launch(self):
        self.run_fake()
        with patch.object(b0.NativeEvaluator, '_open') as opened:
            with self.assertRaises(FileExistsError):
                b0.run_calibration(self.config, self.out, trial=True)
            opened.assert_not_called()

    def test_full_search_exercises_gates_but_synthetic_cannot_certify(self):
        cases = b0.test_cases(self.config, trial=True)
        with patch.object(b0, 'test_cases', return_value=cases):
            report = self.run_fake(trial=False)
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['execution_completed'])
        self.assertEqual({a['status'] for a in report['assessments'].values()}, {'passed'})
        stages = {row['stage'] for row in json.loads((self.out/'convergence_results.json').read_text())}
        self.assertTrue({'reference_domain_x', 'reference_domain_top', 'reference_domain_bottom',
                         'reference_mesh_x', 'reference_mesh_y', 'reference_joint', 'reference_pml',
                         'candidate_screen', 'candidate_combined', 'full_envelope_gate', 'full_reference_gate'} <= stages)
        self.assertFalse((self.out/'solver_profiles.mat').exists())

    def test_unstable_reference_does_not_select_production_profile(self):
        evaluator = SyntheticEvaluator(self.config)
        run = b0.CalibrationRun(self.config, self.out, evaluator)
        with patch.object(run, 'compare', return_value=(False, 1.)):
            result = run.run()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Reference domain', result['reason'])
        self.assertEqual(result['profiles'], {})

    def test_anisotropic_errors_cannot_cancel_in_joint_reference_refinement(self):
        config = self.config
        class CancellingErrors(SyntheticEvaluator):
            def __call__(self, kind, case, profile):
                value = super().__call__(kind, case, profile)
                base = b0.base_profiles(config)[kind]
                error = (profile.dx_m/base.dx_m-profile.dy_m/base.dy_m)*.01
                value['neff'][0] += error
                value['delta_beta'] += error*2*np.pi/case['wavelength_m']
                return value
        with patch.object(b0, 'test_cases', return_value=b0.test_cases(self.config, trial=True)):
            result = b0.run_calibration(self.config, self.out, evaluator_factory=CancellingErrors)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('mesh_x', result['reason'])

    def test_stored_coefficient_check_detects_a_wrong_saved_branch(self):
        reference = self.sample()
        factor = 2*np.pi/reference['wavelength_m']
        saved = {key: factor*reference['neff'][i] for i, key in enumerate(('beta_upper', 'beta_lower', 'beta_plus', 'beta_minus'))}
        saved.update(delta_beta=reference['delta_beta'], kappa=reference['kappa'])
        self.assertTrue(b0.compare_saved_coefficients(self.config, saved, reference)['passed'])
        saved['beta_upper'] += 1e4
        self.assertFalse(b0.compare_saved_coefficients(self.config, saved, reference)['passed'])

    def test_publication_protocol_and_generated_config_keep_project_root(self):
        # Impersonate a real evaluator ONLY to exercise the publication protocol
        # in a deleted temporary directory. Every optical datum remains synthetic.
        class ProtocolFixture(SyntheticEvaluator):
            synthetic = False
        cases = b0.test_cases(self.config, trial=True)
        with patch.object(b0, 'test_cases', return_value=cases):
            result = b0.run_calibration(self.config, self.out, evaluator_factory=ProtocolFixture)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue((self.out/'solver_profiles.mat').is_file())
        loaded = load_config(self.out/'calibrated_configuration.json')
        self.assertEqual(loaded.project_root, self.config.project_root)
        self.assertEqual(loaded.solver.calibration['certificate_id'], result['certificate_id'])

    def test_certificate_integrity_scope_and_staleness(self):
        certificate = certificate_fixture(self.config)
        config = b0.apply_certificate(self.config, certificate)
        config.validate()
        restored = config_from_dict(config.as_dict(), project_root=config.project_root)
        self.assertEqual(restored.solver.calibration['certificate_id'], certificate['certificate_id'])
        for key, value in (('status', 'incomplete'), ('trial', True), ('synthetic', True)):
            bad = deepcopy(certificate); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                b0.apply_certificate(self.config, bad)
        bad = deepcopy(certificate); bad['profiles']['coupled']['dx_m'] *= 2
        with self.assertRaisesRegex(ValueError, 'checksum'):
            b0.apply_certificate(self.config, bad)
        changed = replace(self.config, materials=replace(self.config.materials, liquid_cross_n=1.56))
        with self.assertRaisesRegex(ValueError, 'does not match geometry'):
            b0.apply_certificate(changed, certificate)
        with patch.object(b0, 'implementation_identity', return_value='new_solver'):
            with self.assertRaisesRegex(ValueError, 'implementation'):
                b0.apply_certificate(self.config, certificate)

    def test_calibrated_profiles_reach_both_adapters_and_native_bundle(self):
        certificate = certificate_fixture(self.config)
        config = b0.apply_certificate(self.config, certificate)
        backend = object.__new__(LumapiBackend)
        backend.config, backend.mesh_accuracy = config, config.solver.mesh_accuracy
        backend._build_domain()
        compact = backend.endpoint_isolated_domain_info()
        isolated = b0.certified_profile(config, 'isolated')
        self.assertEqual(compact['mesh_cells_x'], isolated.nx)
        self.assertEqual(compact['y_min_m'], isolated.y_min_m)
        self.assertEqual(len(backend.track_x), b0.certified_profile(config, 'coupled').tracking_nx)
        payload = config_payload(config)
        self.assertEqual(payload['block0_certificate_id'], certificate['certificate_id'])
        self.assertEqual(payload['compact_profile']['mesh_cells_y'], isolated.ny)
        with patch.object(NativeLumapiBackend, '_open_session', side_effect=AssertionError('Unexpected MODE launch')):
            bundle = prepare_bundle(config, self.out/'bundle')
        mat = loadmat(bundle/'native_config.mat', simplify_cells=True)['vlac_cfg']
        self.assertEqual(mat['full_profile']['x_span_m'], certificate['profiles']['coupled']['x_span_m'])
        manifest = json.loads((bundle/'manifest.json').read_text())
        self.assertEqual(manifest['block0_certificate_id'], certificate['certificate_id'])
        self.assertIn('block0_certificate.json', manifest['files'])

    def test_handoff_and_cache_identity_include_certificate(self):
        config = b0.apply_certificate(self.config, certificate_fixture(self.config))
        self.assertNotIn('block0_certificate_id', physics_signature(self.config, 1))
        self.assertIn('block0_certificate_id', physics_signature(config, 1))

    def test_selected_design_gate_stops_failed_or_incomplete_results(self):
        config = b0.apply_certificate(self.config, certificate_fixture(self.config))
        for status in ('failed', 'incomplete'):
            with patch.object(b0, 'verify_design', return_value={'status': status}), self.assertRaisesRegex(b0.BudgetExceeded if status=='incomplete' else ValueError, status):
                b0.require_design_verification(config, self.out, config.geometry)

    def test_cli_uses_budget_and_trial_options_without_mode(self):
        with patch.object(b0, 'run_calibration', return_value={'status': 'incomplete'}) as run:
            code = cli.main(['block0', '--config', str(ROOT/'config/native.json'), '--out', str(self.out),
                             '--trial', '--max-solves', '12', '--max-seconds', '180', '--hide-gui'])
        self.assertEqual(code, 3)
        self.assertTrue(run.call_args.kwargs['trial'])
        self.assertEqual(run.call_args.args[0].convergence.max_solves, 12)
        self.assertEqual(run.call_args.args[0].convergence.max_seconds, 180)
        self.assertTrue(run.call_args.args[0].solver.hide_gui)

    def test_native_convergence_entry_and_physical_override_placement(self):
        self.assertIn('function vlac_convergence_sample(', native_script('vlac_convergence.lsf'))
        from test_efficiency import geometry
        profile = replace(b0.base_profiles(self.config)['coupled'], local_dx_m=20e-9, local_dy_m=10e-9)
        script = '\n'.join(b0.mesh_override_script(profile, geometry(self.config, offset=2e-6), True, False))
        self.assertIn('vlac_local_2', script)
        self.assertNotIn('vlac_local_1', script)
        self.assertIn('set("x",2e-06)', script)
        self.assertIn('set("dx",2e-08)', script)


if __name__ == '__main__':
    unittest.main()
