"""All optical fields in these tests are SYNTHETIC; no MODE license is used."""
from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vlac import cli
from vlac.config import load_config
from vlac.geometry import make_geometry
from vlac.modes import ModeSnapshot
from vlac import lower_state_check as check

ROOT = Path(__file__).resolve().parents[1]


class SyntheticProbe:
    synthetic = True
    instances = []
    state_shift = 0.
    mesh_shift = 0.
    coupled_healthy = True

    def __init__(self, config):
        self.config, self.calls, self.closed = config, [], False
        self.instances.append(self)

    def __call__(self, geometry, wavelength, kind, profile):
        self.calls.append((geometry, wavelength, kind, profile))
        shape = (profile.tracking_nx, profile.tracking_ny)
        a, b = np.zeros(shape, complex), np.zeros(shape, complex)
        a[shape[0]//2, shape[1]//2] = 1
        b[shape[0]//2+1, shape[1]//2] = 1
        lower_n = 1.7 + (self.state_shift if geometry.state == 'bar' else 0.)
        if profile.dx_m < 4e-8:
            lower_n += self.mesh_shift
        def mode(n, e, name):
            return ModeSnapshot(n, e, np.zeros_like(e), 1., .9, 'SYNTHETIC-'+name)
        if kind == 'upper':
            modes = [mode(1.7, a, 'upper')]
        elif kind == 'lower':
            modes = [mode(lower_n, b, 'lower')]
        else:
            modes = [mode(1.70002, (a+b)/np.sqrt(2), 'plus'),
                     mode(1.69998 if self.coupled_healthy else 1.69999999, (a-b)/np.sqrt(2), 'minus')]
        # Order changes deliberately; tests never rely on MODE mode numbers.
        return list(reversed(modes)), 1

    def close(self):
        self.closed = True


class LowerStateCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)/'check'
        self.config = load_config(ROOT/'config/native.json')
        self.config = replace(self.config, solver=replace(self.config.solver, tracking_grid_nx=17, tracking_grid_ny=17),
            propagation=replace(self.config.propagation, rk4_steps_initial=10, rk4_steps_max=80))
        SyntheticProbe.instances = []
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temp.cleanup()

    def run_check(self, **kwargs):
        return check.run_check(self.config, self.out, probe_factory=SyntheticProbe, **kwargs)

    def fixture(self):
        g = self.config.geometry
        endpoint = SimpleNamespace(w1_input_m=1.1e-6, w1_output_m=1.8e-6,
            h_gap_m=1e-6, h_rib_m=175e-9, delta_beta_max_rad_per_m=5e4)
        solution = SimpleNamespace(w1_control_m=np.linspace(1.1e-6, 1.8e-6, 4),
            d_c2c_control_m=np.linspace(g.d_c2c_input_m, g.d_c2c_output_m, 4))
        return endpoint, solution, None, {'block': 2, 'sha256': 'SYNTHETIC', 'validation': {'status': 'review'}}

    def test_plan_reduces_work_and_never_opens_mode(self):
        plan, _, _, profiles = check.prepare(self.config, 'screen')
        self.assertEqual(plan['nominal_fde_calls'], 20)
        np.testing.assert_allclose(plan['wavelengths_m'], [1.26e-6, 1.30e-6, 1.34e-6])
        self.assertEqual(SyntheticProbe.instances, [])
        self.assertGreater(profiles['domain'].nx, profiles['production'].nx)
        self.assertEqual(profiles['domain'].dx_m, profiles['production'].dx_m)
        np.testing.assert_array_equal(profiles['domain'].grid()[0], profiles['production'].grid()[0])

    def test_full_requires_compatible_saved_handoff(self):
        with self.assertRaisesRegex(ValueError, 'profile-dir'):
            check.prepare(self.config, 'full')
        with self.assertRaises(ValueError):
            check.prepare(self.config, 'full', self.out)

    def test_screen_is_twenty_fresh_solves_and_never_qualifies_synthetic(self):
        report = self.run_check(max_solves=40)
        self.assertEqual(report['fresh_fde_calls'], 20)
        self.assertEqual(len(report['screen']), 9)
        self.assertEqual(len(report['numerical_checks']), 14)
        self.assertTrue(all(v['passed'] for v in report['screen']))
        self.assertFalse(report['eligible_for_shared_lower_mode'])
        self.assertEqual(report['status'], 'inconclusive')
        self.assertTrue(SyntheticProbe.instances[-1].closed)
        self.assertTrue(all(c[2] == 'lower' for c in SyntheticProbe.instances[-1].calls))

    def test_state_sensitivity_fails_gate(self):
        class Sensitive(SyntheticProbe):
            state_shift = 1e-4
        report = check.run_check(self.config, self.out, probe_factory=Sensitive)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['eligible_for_shared_lower_mode'])

    def test_mesh_uncertainty_is_inconclusive(self):
        class Unresolved(SyntheticProbe):
            mesh_shift = 1e-4
        report = check.run_check(self.config, self.out, probe_factory=Unresolved)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertIn('uncertainty', report['reason'])

    def test_budget_reserves_retries_and_resume_reuses_fields(self):
        first = self.run_check(max_solves=3)
        self.assertEqual(first['status'], 'incomplete')
        self.assertEqual(first['fresh_fde_calls'], 2)
        second = self.run_check(max_solves=40, resume=True)
        self.assertEqual(second['fresh_fde_calls'], 18)
        self.assertEqual(second['reused_sample_packets'], 2)
        self.assertFalse(second['eligible_for_shared_lower_mode'])

    def test_resume_rejects_changed_inputs_and_corrupt_fields(self):
        self.run_check(max_solves=3)
        with self.assertRaisesRegex(ValueError, 'changed inputs'):
            check.run_check(replace(self.config, device=replace(self.config.device, length_m=.002)),
                self.out, resume=True, probe_factory=SyntheticProbe)
        next((self.out/'samples').glob('*.npz')).write_bytes(b'CORRUPTED SYNTHETIC DATA')
        report = self.run_check(resume=True)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('checksum', report['reason'])

    def test_existing_production_directory_is_not_modified(self):
        self.out.mkdir()
        sentinel = self.out/'block2_result.json'
        sentinel.write_text('SYNTHETIC existing workflow result')
        with self.assertRaises(FileExistsError):
            self.run_check()
        self.assertEqual(sentinel.read_text(), 'SYNTHETIC existing workflow result')
        self.assertFalse((self.out/'report.json').exists())

    def test_nonfinite_raw_sample_fails_and_closes(self):
        class Nonfinite(SyntheticProbe):
            def __call__(self, *args):
                modes, count = super().__call__(*args)
                modes[0].ex[0, 0] = np.nan
                return modes, count
        report = check.run_check(self.config, self.out, probe_factory=Nonfinite)
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(SyntheticProbe.instances[-1].closed)

    def test_probe_failure_accounts_unknown_attempts_without_claiming_completed_solves(self):
        class Unavailable(SyntheticProbe):
            def __call__(self, *args):
                raise RuntimeError('SYNTHETIC solver startup failure')
        report = check.run_check(self.config, self.out, probe_factory=Unavailable)
        self.assertEqual(report['fresh_fde_calls'], 0)
        self.assertEqual(report['unconfirmed_attempts_reserved'], 2)
        self.assertEqual(report['accounted_fde_attempts_upper_bound'], 2)
        self.assertEqual(report['status'], 'failed')

    def test_invalid_reference_stops_full_check_before_propagation(self):
        class InvalidReference(SyntheticProbe):
            def __call__(self, geometry, wavelength, kind, profile):
                modes, count = super().__call__(geometry, wavelength, kind, profile)
                if kind == 'upper':
                    modes[0].neff += .01
                return modes, count
        with patch.object(check, 'load_block', return_value=self.fixture()):
            report = check.run_check(self.config, self.out, stage='full', profile_dir='SYNTHETIC', probe_factory=InvalidReference)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertEqual(report['fresh_fde_calls'], 22)
        self.assertEqual(report['propagation'], [])

    def test_full_comparison_preserves_complex_coefficients_and_both_launches(self):
        with patch.object(check, 'load_block', return_value=self.fixture()):
            report = self.run_check(stage='full', profile_dir='SYNTHETIC')
        self.assertEqual(report['fresh_fde_calls'], 80)
        self.assertEqual(len(report['coefficient_comparisons']), 30)
        self.assertEqual(len(report['propagation']), 12)
        self.assertEqual(len(report['midpoint_checks']), 24)
        self.assertTrue(all(v['passed'] for v in report['propagation']))
        self.assertTrue(all(v['passed'] for v in report['coefficient_comparisons']))
        self.assertFalse(report['eligible_for_shared_lower_mode'])
        self.assertTrue(all(v['shared_lower']['delta_beta'] == 0j for v in report['coefficient_comparisons']))

    def test_mode_metrics_include_loss_phase_and_global_phase_invariance(self):
        a = ModeSnapshot(1.7+1e-9j, np.ones((2, 2), complex), np.zeros((2, 2)), 1., 1., 'SYNTHETIC')
        b = replace(a, neff=a.neff+1e-6+2e-8j, ex=1j*a.ex)
        result = check.mode_difference(a, b, 1.3e-6, .0015)
        self.assertAlmostEqual(result['squared_field_overlap'], 1)
        self.assertGreater(result['phase_error_rad'], 0)
        self.assertGreater(result['attenuation_error_db'], 0)

    def test_cli_plan_only_and_invalid_budgets(self):
        args = ['check-lower-state', '--config', str(ROOT/'config/native.json'), '--out', str(self.out)]
        self.assertEqual(cli.main(args+['--plan-only']), 0)
        self.assertFalse(self.out.exists())
        self.assertEqual(cli.main(args+['--max-seconds', '3601']), 2)
        self.assertEqual(cli.main(args+['--max-solves', '0']), 2)

    def test_hard_timeout_only_terminates_worker_tree_and_marks_incomplete(self):
        self.out.mkdir()
        write = self.out/'report.json'
        write.write_text(json.dumps({'run_id': 'SYNTHETIC-RUN', 'status': 'running'}))
        worker = Mock(pid=12345)
        worker.communicate.side_effect = [subprocess.TimeoutExpired('SYNTHETIC', .01), (b'', None)]
        with patch.object(check.uuid, 'uuid4', return_value=SimpleNamespace(hex='SYNTHETIC-RUN')), \
             patch.object(check.subprocess, 'Popen', return_value=worker), \
             patch.object(check.subprocess, 'run') as terminate, patch.object(check.os, 'killpg', create=True) as killpg:
            self.assertEqual(check._supervise([], self.out, .01), 3)
        self.assertEqual(json.loads(write.read_text())['status'], 'incomplete')
        worker.kill.assert_called_once()
        if os.name == 'nt':
            self.assertEqual(terminate.call_args.args[0], ['taskkill', '/PID', '12345', '/T', '/F'])
        else:
            killpg.assert_called_once()


if __name__ == '__main__':
    unittest.main()
