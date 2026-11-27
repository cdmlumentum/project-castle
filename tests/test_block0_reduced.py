"""Reduced Block 0 contracts. ALL optical fixtures are SYNTHETIC TEST DATA.

No test launches MODE or establishes real optical convergence.
"""
from copy import deepcopy
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vlac import block0 as full
from vlac import block0_reduced as reduced
from vlac import cli
from vlac.config import load_config
from test_block0 import SyntheticEvaluator, synthetic_sample

ROOT = Path(__file__).resolve().parents[1]


class ReducedBlockZeroTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)/'reduced'
        config = load_config(ROOT/'config/native.json')
        self.config = replace(config, convergence=replace(config.convergence,
            max_solves=200, max_seconds=120, comparison_dx_m=2e-6, comparison_dy_m=1e-6))
        SyntheticEvaluator.instances = []

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(self, **kwargs):
        config = kwargs.pop('config', self.config)
        factory = kwargs.pop('evaluator_factory', SyntheticEvaluator)
        return reduced.run_reduced(config, self.out, evaluator_factory=factory, **kwargs)

    def test_nine_settings_preserve_domains_and_independent_spacing(self):
        plan = reduced.prepare_plan(self.config)
        self.assertEqual(plan['planned_fde_calls'], 180)
        self.assertEqual(plan['hard_lifetime_fde_call_limit'], 200)
        self.assertEqual(plan['reserve_fde_calls'], 20)
        self.assertIn('NOT a verified phase-matching', plan['case_selection'])
        settings = reduced.solver_settings(self.config)
        base = full.base_profiles(self.config)
        for kind, profiles in settings.items():
            self.assertEqual(list(profiles), list('ABCDEFGHI'))
            for profile in profiles.values():
                self.assertEqual(profile.y_min_m, base[kind].y_min_m)
                self.assertEqual(profile.y_max_m, base[kind].y_max_m)
                self.assertGreaterEqual(profile.tracking_x_span_m, profile.x_span_m)
                self.assertEqual(profile.tracking_nx % 2, 1)
                self.assertEqual(profile.tracking_ny % 2, 1)
            self.assertEqual(profiles['A'].dy_m, profiles['C'].dy_m)
            self.assertEqual(profiles['C'].dx_m, profiles['E'].dx_m)
            self.assertEqual(profiles['E'].dy_m, profiles['G'].dy_m)
            anchor = profiles['E']
            for label in ('F', 'G'):
                expanded = profiles[label]
                self.assertEqual(expanded.nx % 2, anchor.nx % 2)
                self.assertAlmostEqual(expanded.x_span_m/expanded.nx/(anchor.x_span_m/anchor.nx), 1., places=13)
                self.assertEqual(expanded.y_min_m, anchor.y_min_m)
                self.assertEqual(expanded.ny, anchor.ny)
            self.assertEqual(profiles['I'].pml_layers-profiles['H'].pml_layers, 4)
        self.assertAlmostEqual(settings['coupled']['A'].x_span_m, 22e-6)
        self.assertAlmostEqual(settings['isolated']['A'].x_span_m, 6e-6)
        self.assertEqual(len(plan['comparisons']), 9)

    def test_explicit_case_validation_before_evaluator_creation(self):
        cases = reduced.stress_cases(self.config)
        self.assertEqual(reduced.stress_cases(self.config, cases), cases)
        for change in ('wrong_length', 'wrong_band', 'out_of_bounds', 'nonfinite', 'wrong_state', 'extra_key'):
            bad = deepcopy(cases)
            if change == 'wrong_length': bad['isolated'].pop()
            if change == 'wrong_band': bad['coupled'][0]['wavelength_m'] = 1.3e-6
            if change == 'out_of_bounds': bad['coupled'][0]['w1_m'] = 100e-6
            if change == 'nonfinite': bad['coupled'][0]['w1_m'] = float('nan')
            if change == 'wrong_state': bad['coupled'][0]['state'] = 'invalid'
            if change == 'extra_key': bad['coupled'][0]['synthetic'] = True
            with self.subTest(change=change), patch.object(reduced.full, 'NativeEvaluator') as factory:
                with self.assertRaises(ValueError):
                    reduced.run_reduced(self.config, self.out, cases=bad, evaluator_factory=factory)
                factory.assert_not_called()

    def test_synthetic_run_exercises_every_gate_without_certification(self):
        report = self.run_fixture()
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['checks_passed'])
        self.assertTrue(report['execution_completed'])
        self.assertFalse(report['convergence_demonstrated'])
        self.assertFalse(report['production_certified'])
        self.assertEqual(report['total_solves'], 180)
        self.assertEqual(report['session_samples'], 72)
        self.assertEqual(len(report['checks']), 72)
        self.assertEqual({r['stage'] for r in report['checks']}, {r[0] for r in reduced.COMPARISONS})
        for name in ('plan.json', 'pilot.json', 'reduced_report.json', 'convergence_results.csv', 'report.html'):
            self.assertTrue((self.out/name).is_file())
        for name in ('certificate.json', 'solver_profiles.json', 'solver_profiles.mat', 'calibrated_configuration.json'):
            self.assertFalse((self.out/name).exists())
        resumed = self.run_fixture(resume=True)
        self.assertEqual(resumed['total_solves'], 180)
        self.assertEqual(resumed['session_solves'], 0)
        self.assertEqual(SyntheticEvaluator.instances[-1].calls, [])

    def test_scoped_pass_is_rejected_by_full_certificate_loader(self):
        class ProtocolFixture(SyntheticEvaluator):
            # Explicitly synthetic protocol fixture exercising the real-status
            # branch. Never export this fixture as production evidence.
            synthetic = False
        report = self.run_fixture(evaluator_factory=ProtocolFixture)
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['convergence_demonstrated'])
        self.assertFalse(report['production_certified'])
        with self.assertRaisesRegex(ValueError, 'full, real MODE certificate'):
            full.apply_certificate(self.config, report)
        self.assertFalse((self.out/'certificate.json').exists())

    def test_end_to_end_guard_rejects_accumulated_small_changes(self):
        class DriftingFixture(SyntheticEvaluator):
            def __call__(self, kind, case, profile):
                self.calls.append((kind, case, profile))
                value = synthetic_sample(kind, case, profile)
                if kind == 'coupled':
                    base = full.base_profiles(config)[kind]
                    value['kappa'] += 5*(1.25-profile.dx_m/base.dx_m)
                return value
        config = self.config
        report = self.run_fixture(evaluator_factory=DriftingFixture)
        self.assertEqual(report['status'], 'failed')
        failures = [r for r in report['checks'] if not r['passed']]
        self.assertEqual({r['stage'] for r in failures}, {'candidate_to_reference'})
        self.assertFalse(report['convergence_demonstrated'])

    def test_resume_uses_cached_samples_and_retains_lifetime_accounting(self):
        small = replace(self.config, convergence=replace(self.config.convergence, max_solves=20))
        partial = self.run_fixture(config=small)
        self.assertEqual(partial['status'], 'incomplete')
        self.assertFalse(partial['execution_completed'])
        self.assertLessEqual(partial['total_solves'], 20)
        complete = self.run_fixture(resume=True)
        self.assertTrue(complete['execution_completed'])
        self.assertEqual(complete['total_solves'], 180)
        self.assertEqual(partial['session_solves']+complete['session_solves'], 180)

    def test_200_cap_includes_retries_across_all_resumes(self):
        class RetryingFixture(SyntheticEvaluator):
            def __call__(self, kind, case, profile):
                value = super().__call__(kind, case, profile)
                value['solves'] *= 2
                return value
        small = replace(self.config, convergence=replace(self.config.convergence, max_solves=60))
        previous = -1
        for invocation in range(8):
            report = self.run_fixture(config=small, evaluator_factory=RetryingFixture, resume=invocation > 0)
            self.assertLessEqual(report['total_solves'], 200)
            self.assertEqual(report['status'], 'incomplete')
            if report['total_solves'] == previous:
                break
            previous = report['total_solves']
        self.assertGreaterEqual(report['total_solves'], 195)
        self.assertFalse(report['execution_completed'])
        self.assertEqual(SyntheticEvaluator.instances[-1].calls, [])

    def test_interrupted_batch_keeps_pre_dispatch_reservation(self):
        class InterruptedFixture(SyntheticEvaluator):
            def __call__(self, *args):
                raise KeyboardInterrupt('SYNTHETIC interruption')
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(evaluator_factory=InterruptedFixture)
        checkpoint = json.loads((self.out/'checkpoint.json').read_text())
        self.assertEqual(checkpoint['total_solves'], 4)
        self.assertEqual(checkpoint['pending_reservation']['reserved_fde_calls'], 4)
        report = self.run_fixture(resume=True)
        self.assertEqual(report['total_solves'], 184)
        self.assertTrue(report['execution_completed'])

    def test_resume_rejects_changed_inputs_and_corrupted_evidence(self):
        self.run_fixture()
        changed = replace(self.config, materials=replace(self.config.materials, liquid_cross_n=1.56))
        with self.assertRaisesRegex(ValueError, 'resume inputs'):
            self.run_fixture(config=changed, resume=True)
        sample = next((self.out/'samples').glob('*.npz'))
        sample.write_bytes(b'CORRUPTED SYNTHETIC EVIDENCE')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_fixture(resume=True)
        self.assertEqual(json.loads((self.out/'reduced_report.json').read_text())['status'], 'failed')

    def test_time_limit_stops_before_solver_dispatch(self):
        config = replace(self.config, convergence=replace(self.config.convergence, max_seconds=1.))
        with patch.object(reduced.time, 'monotonic', side_effect=[0., 2., 2.]):
            report = self.run_fixture(config=config)
        self.assertEqual(report['total_solves'], 0)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(SyntheticEvaluator.instances[-1].calls, [])

    def test_shared_backend_budget_exhaustion_is_incomplete(self):
        class BudgetStoppedFixture(SyntheticEvaluator):
            def __call__(self, *args):
                raise reduced.RuntimeBudgetExceeded('SYNTHETIC backend budget stop')
        report = self.run_fixture(evaluator_factory=BudgetStoppedFixture)
        self.assertEqual(report['status'], 'incomplete')
        self.assertFalse(report['execution_completed'])
        self.assertEqual(report['total_solves'], 4)

    def test_output_lock_prevents_concurrent_budget_journals(self):
        with reduced._output_lock(self.out):
            with self.assertRaisesRegex(FileExistsError, 'already in use'):
                self.run_fixture()
        self.assertEqual(SyntheticEvaluator.instances, [])

    def test_plan_only_cli_never_launches_mode_or_writes_outputs(self):
        stream = io.StringIO()
        with patch.object(reduced.full.NativeEvaluator, '_open', side_effect=AssertionError('Unexpected MODE launch')):
            with contextlib.redirect_stdout(stream):
                code = cli.main(['block0', '--method', 'reduced', '--plan-only',
                    '--config', str(ROOT/'config/native.json'), '--out', str(self.out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stream.getvalue())['planned_fde_calls'], 180)
        self.assertFalse(self.out.exists())

    def test_cli_routes_reduced_method_and_defaults_to_two_hour_budget(self):
        with patch.object(reduced, 'run_reduced', return_value={'status': 'incomplete'}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(['block0', '--method', 'reduced', '--config', str(ROOT/'config/native.json'),
                                 '--out', str(self.out), '--max-solves', '999'])
        self.assertEqual(code, 3)
        self.assertEqual(run.call_args.args[0].convergence.max_seconds, 7200.)
        # The runner independently clamps even a larger invocation allowance.
        self.assertEqual(reduced.HARD_MAX_SOLVES, 200)


if __name__ == '__main__':
    unittest.main()
