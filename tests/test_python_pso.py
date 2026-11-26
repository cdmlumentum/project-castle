"""Numerical/integration tests. Synthetic modes below are not device results."""

from dataclasses import replace
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from vlac import cli
from vlac.config import EndpointSearchConfig, load_config
from vlac.endpoint import search_endpoints, search_endpoints_grid
from vlac.endpoint_constraints import EndpointSignError, cross_state_sign_report, require_cross_state_signs
from vlac.geometry import make_geometry, rectangles_for_cross_section
from vlac.materials import build_material_bank
from vlac.modes import InvalidModeError, ModeSnapshot, select_mode, extract_coupling
from vlac.cache import SQLiteCache
from vlac.python_pso import EndpointEvaluator, run_particle_swarm, search_endpoints_python_pso
from vlac.solver import LumapiBackend

ROOT = Path(__file__).resolve().parents[1]


class SyntheticBackend:
    """Analytic fixture that records every call and rejects narrow upper modes."""
    backend_name = "synthetic_test_fixture_only"

    def __init__(self, config, **kwargs):
        self.config, self.calls = config, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def preflight(self, wavelength):
        pass

    def endpoint_isolated_domain_info(self, **kwargs):
        backend = object.__new__(LumapiBackend)
        backend.config, backend.mesh_accuracy = self.config, self.config.solver.mesh_accuracy
        backend._build_domain()
        return backend.endpoint_isolated_domain_info(**kwargs)

    def solve_isolated_mode(self, geometry, wavelength, *, guide, reference=None,
                            compact_x=None, use_cache=True, **kwargs):
        self.calls.append((guide, geometry, wavelength, use_cache, reference))
        if guide == "upper" and geometry.w1_m < 0.25e-6:
            raise InvalidModeError("Synthetic narrow-width rejection")
        neff = 1.80 if guide == "lower" else (
            1.80 + 0.10*(geometry.w1_m-(self.config.geometry.w1_min_m+self.config.geometry.w1_max_m)/2)/1e-6
            + 0.005*(geometry.liquid_index-self.config.materials.liquid_cross_n) + 0.001*(wavelength/1e-6-1.30)
        )
        shape = (self.config.solver.tracking_grid_nx, self.config.solver.tracking_grid_ny)
        mode = ModeSnapshot(complex(neff), np.ones(shape)/np.sqrt(np.prod(shape)), np.zeros(shape), 1.0, 0.8, "synthetic")
        return mode, 1.0


class SwarmTests(unittest.TestCase):
    def test_known_interior_optimum_and_reproducible_bounds(self):
        settings = replace(EndpointSearchConfig(), pso_generation_size=16, pso_max_generations=70, pso_random_seed=42)
        traces = []
        bounds = np.array([[0.3e-6, 1.2e-6], [0.3e-6, 1.2e-6]])
        target = np.array([0.57e-6, 1.03e-6])
        objective = lambda x: -float(np.sum(((x-target)/1e-6)**2))
        a = run_particle_swarm(objective, bounds, settings, on_generation=traces.append)
        b = run_particle_swarm(objective, bounds, settings)
        np.testing.assert_allclose(a.position, target, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(a.position, b.position)
        self.assertEqual(a.score, b.score)
        self.assertEqual(a.evaluations, 16*70)
        self.assertFalse(a.converged)
        self.assertTrue(all(np.all((np.array(s['positions_normalized']) >= 0) &
                                   (np.array(s['positions_normalized']) <= 1)) for s in traces))
        best = [s['best_score'] for s in traces]
        self.assertTrue(all(x <= y for x, y in zip(best, best[1:])))

    def test_all_invalid_has_no_winner(self):
        states = []
        with self.assertRaisesRegex(RuntimeError, "no candidate"):
            run_particle_swarm(lambda x: -np.inf, [[0, 1], [0, 1]],
                               replace(EndpointSearchConfig(), pso_max_generations=2), on_generation=states.append)
        self.assertEqual(len(states), 2)
        self.assertIsNone(states[-1]['best_score'])
        json.dumps(states, allow_nan=False)

    def test_unexpected_failure_and_nan_not_masked(self):
        def fail(x):
            raise RuntimeError("synthetic API license failure")
        with self.assertRaisesRegex(RuntimeError, "license"):
            run_particle_swarm(fail, [[0, 1], [0, 1]], EndpointSearchConfig())
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            run_particle_swarm(lambda x: float('nan'), [[0, 1], [0, 1]], EndpointSearchConfig())

    def test_early_stop_requires_spread_as_well_as_stall(self):
        settings = replace(EndpointSearchConfig(), pso_inertia=0, pso_cognitive=0, pso_social=0,
                           pso_tolerance=0.1, pso_max_generations=5)
        result = run_particle_swarm(lambda x: 1., [[0, 1], [0, 1]], settings)
        self.assertFalse(result.converged)  # corner seeds keep the spread at one
        self.assertEqual(result.generations, 5)
        result = run_particle_swarm(lambda x: 1., [[0, 1], [0, 1]], replace(settings, pso_tolerance=1))
        self.assertTrue(result.converged)
        self.assertEqual(result.generations, 4)


class SignRestrictionTests(unittest.TestCase):
    @staticmethod
    def records():
        return [dict(end=end, state=state, wavelength_m=wavelength,
                     delta_beta_real_rad_per_m=-100. if end == 'input' else 200.)
                for end in ('input', 'output') for state in ('cross', 'bar')
                for wavelength in (1.26e-6, 1.34e-6)]

    def test_each_band_edge_is_checked_independently(self):
        for end in ('input', 'output'):
            for wavelength in (1.26e-6, 1.34e-6):
                rows = self.records()
                row = next(r for r in rows if r['state'] == 'cross' and r['end'] == end and r['wavelength_m'] == wavelength)
                row['delta_beta_real_rad_per_m'] *= -1
                with self.subTest(end=end, wavelength=wavelength), self.assertRaises(EndpointSignError):
                    require_cross_state_signs(rows)
                report = cross_state_sign_report(rows)
                self.assertEqual(sum(not c['passed'] for c in report['checks']), 1)

    def test_zero_fails_and_bar_signs_are_unrestricted(self):
        for value in (0., -0., float('nan'), float('inf')):
            rows = self.records()
            rows[0]['delta_beta_real_rad_per_m'] = value
            with self.subTest(value=value), self.assertRaises(EndpointSignError):
                require_cross_state_signs(rows)
        for value in (-500., 500., 0.):
            rows = self.records()
            for row in rows:
                if row['state'] == 'bar':
                    row['delta_beta_real_rad_per_m'] = value
            self.assertTrue(require_cross_state_signs(rows)['passed'])

    def test_missing_or_duplicate_edge_is_an_error(self):
        with self.assertRaises(ValueError):
            require_cross_state_signs(self.records()[1:])
        rows = self.records()
        rows[1]['wavelength_m'] = rows[0]['wavelength_m']
        with self.assertRaises(ValueError):
            require_cross_state_signs(rows)

    def test_constraint_orientation_matches_blackman_target(self):
        from vlac.profiles import target_profiles
        delta, _ = target_profiles(np.array([0., 1.]), 1., 100.)
        rows = self.records()
        for row in rows:
            if row['state'] == 'cross':
                row['delta_beta_real_rad_per_m'] = float(delta[0 if row['end'] == 'input' else 1])
        self.assertTrue(require_cross_state_signs(rows)['passed'])


class EndpointTests(unittest.TestCase):
    def setUp(self):
        base = load_config(ROOT/'config/default.json')
        self.config = replace(base, endpoint_search=replace(base.endpoint_search, method='python_pso'))
        self.bank = build_material_bank(self.config)

    def test_eight_case_objective_and_lower_reuse(self):
        backend = SyntheticBackend(self.config)
        evaluator = EndpointEvaluator(backend, self.config, self.bank)
        evaluator.precompute_lower()
        score, rows = evaluator.evaluate([.6e-6, 1.7e-6])
        self.assertEqual(len(backend.calls), 10)
        self.assertEqual(sum(c[0] == 'lower' for c in backend.calls), 2)
        self.assertEqual(len(rows), 8)
        self.assertEqual({(r['end'], r['state'], r['wavelength_m']) for r in rows},
                         {(e, s, w) for e in ('input', 'output') for s in ('cross', 'bar')
                          for w in (1.26e-6, 1.34e-6)})
        manual = []
        for row in rows:
            n = 1.80 + .10*(row['w1_m']-(self.config.geometry.w1_min_m+self.config.geometry.w1_max_m)/2)/1e-6 + .005*((self.config.materials.liquid_cross_n if row['state'] == 'cross' else self.config.materials.liquid_bar_n)-self.config.materials.liquid_cross_n) + .001*(row['wavelength_m']/1e-6-1.30)
            expected = 2*np.pi*(n-1.80)/row['wavelength_m']
            self.assertAlmostEqual(row['delta_beta_real_rad_per_m'], expected, places=7)
            manual.append(abs(expected))
        self.assertAlmostEqual(score, min(manual), places=7)
        self.assertEqual(backend.calls[0][1].state, 'cross')
        self.assertIsNotNone(backend.calls[1][4])

    def test_smoke_outputs_penalties_and_ten_uncached_checks(self):
        backend = SyntheticBackend(self.config)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for name in ('endpoint_pso_setup.lsf', 'endpoint_pso.lms', 'endpoint_width_preflight.json'):
                (out/name).write_text('stale')
            result = search_endpoints_python_pso(backend, self.config, self.bank, validation_only=True, output_dir=out)
            self.assertEqual(result.search_method, 'python_pso')
            self.assertEqual(result.pso_summary['evaluations'], 3)
            self.assertEqual(len([c for c in backend.calls if not c[3]]), 10)
            self.assertEqual(sum(c[0] == 'lower' and c[3] for c in backend.calls), 2)
            particles = list(csv.DictReader(io.StringIO((out/'endpoint_pso_particles.csv').read_text())))
            self.assertEqual(len(particles), 3)
            self.assertEqual(particles[0]['valid'], 'True')
            self.assertEqual(particles[1]['valid'], 'False')
            self.assertLess(result.w1_input_m, result.w1_output_m)
            self.assertTrue(require_cross_state_signs(result.endpoint_records)['passed'])
            self.assertTrue(any(r['valid'] == 'False' and r['error'] for r in particles))
            self.assertTrue(all(r['bottleneck_rad_per_m'] == '' for r in particles if r['valid'] == 'False'))
            gen = list(csv.DictReader(io.StringIO((out/'endpoint_pso_generation_detunings.csv').read_text())))[0]
            self.assertIn('delta_beta_cross_in_lambda_min_rad_per_m', gen)
            self.assertIn('delta_beta_bar_out_worst_wavelength_m', gen)
            report = json.loads((out/'endpoint_pso_verification.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(len(report['endpoint_records']), 8)
            self.assertEqual(report['maximum_case_error_rad_per_m'], 0)
            self.assertTrue(report['cross_state_sign_verification']['passed'])
            self.assertEqual(len(report['cross_state_sign_verification']['checks']), 4)
            self.assertFalse((out/'endpoint_pso_setup.lsf').exists())
            self.assertFalse((out/'endpoint_width_preflight.json').exists())
            self.assertEqual(json.loads((out/'endpoint_pso_checkpoint.json').read_text())['swarm']['generation'], 1)
            self.assertEqual(self.config.geometry.w1_min_m, 0.3e-6)

    def test_verification_disagreement_does_not_publish_best(self):
        class Unstable(SyntheticBackend):
            def solve_isolated_mode(self, *args, **kwargs):
                mode, overlap = super().solve_isolated_mode(*args, **kwargs)
                if kwargs.get('use_cache') is False and kwargs['guide'] == 'upper':
                    mode.neff = 1.8 + (mode.neff-1.8)*1.0001
                return mode, overlap
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'disagree'):
                search_endpoints_python_pso(Unstable(self.config), self.config, self.bank,
                                           validation_only=True, output_dir=folder)
            out = Path(folder)
            self.assertFalse((out/'endpoint_pso_best.json').exists())
            self.assertEqual(json.loads((out/'endpoint_pso_verification.json').read_text())['status'], 'failed')

    def test_runtime_error_not_penalized(self):
        class Broken(SyntheticBackend):
            def solve_isolated_mode(self, *args, **kwargs):
                if kwargs['guide'] == 'upper':
                    raise RuntimeError('unexpected getdata error')
                return super().solve_isolated_mode(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'getdata'):
                search_endpoints_python_pso(Broken(self.config), self.config, self.bank,
                                           validation_only=True, output_dir=folder)

    def test_equal_and_reversed_widths_fail_cross_state_signs(self):
        evaluator = EndpointEvaluator(SyntheticBackend(self.config), self.config, self.bank)
        evaluator.precompute_lower()
        for widths in ([1.2e-6, 1.2e-6], [.6e-6, .6e-6], [1.7e-6, .6e-6]):
            with self.subTest(widths=widths), self.assertRaises(EndpointSignError) as caught:
                evaluator.evaluate(widths)
            self.assertEqual(len(caught.exception.records), 8)
            self.assertFalse(caught.exception.report['passed'])
        _, rows = evaluator.evaluate([.75e-6, 1.55e-6])
        self.assertTrue(require_cross_state_signs(rows)['passed'])

    def test_wrong_sign_is_logged_and_cannot_become_a_winner(self):
        class SameSignBackend(SyntheticBackend):
            def solve_isolated_mode(self, *args, **kwargs):
                mode, overlap = super().solve_isolated_mode(*args, **kwargs)
                if kwargs['guide'] == 'upper':
                    mode.neff += .2
                return mode, overlap
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'endpoint_pso_best.json').write_text('{"status":"stale"}')
            with self.assertRaisesRegex(RuntimeError, 'sign restriction'):
                search_endpoints_python_pso(SameSignBackend(self.config), self.config, self.bank,
                                           validation_only=True, output_dir=out)
            self.assertFalse((out/'endpoint_pso_best.json').exists())
            particles = list(csv.DictReader(io.StringIO((out/'endpoint_pso_particles.csv').read_text())))
            rejected = next(r for r in particles if r['rejection_reason'] == 'cross_state_endpoint_sign')
            self.assertEqual(rejected['cross_state_signs_passed'], 'False')
            self.assertEqual(rejected['mode_checks_passed'], 'True')
            self.assertEqual(rejected['bottleneck_rad_per_m'], '')
            self.assertIn('input', rejected['error'])
            self.assertTrue(rejected['delta_beta_cross_out_lambda_max_rad_per_m'])
            history = list(csv.DictReader(io.StringIO((out/'endpoint_pso_history.csv').read_text())))
            self.assertEqual(history[-1]['valid_particles'], '0')
            self.assertEqual(history[-1]['best_delta_beta_max_rad_per_m'], '')

    def test_fresh_verification_rechecks_cross_state_signs(self):
        class FreshWrongSign(SyntheticBackend):
            def solve_isolated_mode(self, *args, **kwargs):
                mode, overlap = super().solve_isolated_mode(*args, **kwargs)
                if kwargs.get('use_cache') is False and kwargs['guide'] == 'upper':
                    mode.neff += .2
                return mode, overlap
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(EndpointSignError, 'sign restriction'):
                search_endpoints_python_pso(FreshWrongSign(self.config), self.config, self.bank,
                                           validation_only=True, output_dir=folder)
            out = Path(folder)
            self.assertFalse((out/'endpoint_pso_best.json').exists())
            report = json.loads((out/'endpoint_pso_verification.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['cross_state_sign_verification']['passed'])
            self.assertEqual(len(report['endpoint_records']), 8)

    def test_grid_rejects_equal_widths_and_fails_without_signed_endpoints(self):
        config = replace(self.config, endpoint_search=replace(self.config.endpoint_search, method='python_grid'),
                         geometry=replace(self.config.geometry, w1_endpoint_candidates_m=[.6e-6, 1.15e-6, 1.7e-6]),
                         optimizer=replace(self.config.optimizer, max_grid_refinements=0))
        result = search_endpoints_grid(SyntheticBackend(config), config, self.bank)
        self.assertTrue(require_cross_state_signs(result.endpoint_records)['passed'])
        self.assertNotEqual(result.w1_input_m, result.w1_output_m)
        equal = [r for r in result.candidate_records if r['w1_input_m'] == r['w1_output_m']]
        self.assertTrue(equal)
        self.assertTrue(all(not r['valid'] for r in equal))
        self.assertTrue(all(r['bottleneck_rad_per_m'] is None for r in equal))
        config = replace(config, geometry=replace(config.geometry, w1_endpoint_candidates_m=[.6e-6, .75e-6]))
        with self.assertRaisesRegex(RuntimeError, 'sign restriction'):
            search_endpoints_grid(SyntheticBackend(config), config, self.bank)

    def test_candidate_te_and_overlap_failures(self):
        evaluator = EndpointEvaluator(SyntheticBackend(self.config), self.config, self.bank)
        evaluator.precompute_lower()
        geometry = evaluator.geometry(1e-6, 0, 'cross', 1.26e-6)
        mode = ModeSnapshot(1.9+0j, np.ones((4, 4)), np.zeros((4, 4)), .1, .8, 'fixture')
        with self.assertRaisesRegex(InvalidModeError, 'TE fraction'):
            evaluator.check_mode(mode, 1., None, geometry, 1.26e-6, 'upper')
        mode.te_fraction = 1.
        with self.assertRaisesRegex(InvalidModeError, 'overlap'):
            evaluator.check_mode(mode, .1, mode, geometry, 1.26e-6, 'upper')
        mode.neff = complex(geometry.liquid_index - .01)
        with self.assertRaisesRegex(InvalidModeError, 'n_clad'):
            evaluator.check_mode(mode, 1., None, geometry, 1.26e-6, 'upper')
        with self.assertRaises(InvalidModeError):
            select_mode([], guide='upper', minimum_te_fraction=.45)

    def test_substrate_free_domain_and_nominal_resolution(self):
        backend = SyntheticBackend(self.config)
        compact = backend.endpoint_isolated_domain_info()
        full = backend.endpoint_isolated_domain_info(compact_x=False)
        self.assertAlmostEqual(compact['x_span_m']/1e-6, 10.0)
        self.assertAlmostEqual(compact['y_span_m']/1e-6, 8.2)
        self.assertAlmostEqual(full['x_span_m']/1e-6, 26.0)
        self.assertEqual(compact['mesh_cells_x'], 203)
        self.assertEqual(compact['mesh_cells_y'], 208)
        geometry = make_geometry(self.config.geometry, w1_m=1e-6, h_rib_m=175e-9,
                                 h_gap_m=1e-6, d_c2c_m=0, state='cross', liquid_index=self.config.materials.liquid_cross_n)
        rects = rectangles_for_cross_section(geometry, x_span_m=compact['x_span_m'],
                                            y_min_m=compact['y_min_m'], y_max_m=compact['y_max_m'])
        self.assertEqual(set(r.material for r in rects), {'SiO2_Malitson_1965', 'Si3N4_Luke_2015_provisional', 'Liquid'})
        oxide = next(r for r in rects if r.name == 'buried_oxide_and_gap')
        self.assertAlmostEqual(oxide.y_center_m - oxide.y_span_m/2, compact['y_min_m'])

    def test_lower_guide_envelope_used_when_upper_bound_is_smaller(self):
        config = replace(self.config, geometry=replace(self.config.geometry, w1_max_m=.8e-6))
        info = SyntheticBackend(config).endpoint_isolated_domain_info()
        self.assertAlmostEqual(info['x_span_m']/1e-6, 10.0)

    def test_actual_adapter_cache_bypass_and_direct_script(self):
        backend = object.__new__(LumapiBackend)
        backend.config, backend.bank = self.config, self.bank
        backend.mesh_accuracy, backend.progress = self.config.solver.mesh_accuracy, None
        backend._build_domain()
        geometry = make_geometry(self.config.geometry, w1_m=1e-6, h_rib_m=175e-9,
                                 h_gap_m=1e-6, d_c2c_m=8e-6, state='cross', liquid_index=self.config.materials.liquid_cross_n)
        mode = ModeSnapshot(1.9+0j, np.ones((4, 4))/4, np.zeros((4, 4)), 1., .8, 'fixture')
        with tempfile.TemporaryDirectory() as folder:
            backend.cache = SQLiteCache(Path(folder)/'cache.sqlite')
            with patch.object(backend, '_mode_candidates', return_value=[mode]) as candidates:
                backend.solve_isolated_mode(geometry, 1.26e-6, guide='upper')
                backend.solve_isolated_mode(geometry, 1.26e-6, guide='upper')
                self.assertEqual(candidates.call_count, 1)
                backend.solve_isolated_mode(geometry, 1.26e-6, guide='upper', use_cache=False)
                self.assertEqual(candidates.call_count, 2)
                self.assertEqual(candidates.call_args.args[0].d_c2c_m, 0)
                self.assertEqual(candidates.call_args.kwargs['mesh_cells_override'], (203, 208))
            scripts = []
            class RecordingSession:
                def eval(self, script):
                    scripts.append(script)
                def nummodes(self):
                    return 0
                def getv(self, name):
                    return {'count': 0, 'cards': []}
            backend.session = RecordingSession()
            with patch.object(backend, '_install_materials'):
                result = backend._mode_candidates(geometry, 1.26e-6, include_upper=True, include_lower=False)
            self.assertEqual(result, [])
            combined = '\n'.join(scripts)
            self.assertIn('findmodes;', combined)
            self.assertIn('setanalysis("wavelength",', combined)
            self.assertIn('buried_oxide_and_gap', combined)
            self.assertIn('setnamed("lower_ridge","enabled",0)', combined)
            self.assertNotIn('Si_Green', combined)
            for forbidden in ('getanalysis(', 'runanalysis', 'runsweep', 'addsweep', 'endpoint_design', 'mode_metrics'):
                self.assertNotIn(forbidden, combined)

    def test_full_workflow_exports_python_result_schema(self):
        # All device coefficients are synthetic; this only exercises integration.
        class FullSyntheticBackend(SyntheticBackend):
            def evaluate_coefficients(self, geometry, wavelength, **kwargs):
                delta, kappa, mean = 1000., 100., 1.e7
                q = np.sqrt((delta/2)**2 + kappa**2)
                return extract_coupling(mean+delta/2, mean-delta/2, mean+q, mean-q), None
        with tempfile.TemporaryDirectory() as folder:
            config = replace(
                self.config,
                materials=replace(self.config.materials, silicon_data_csv=str(ROOT/'data/green_si_2008_1260_1340.csv')),
                endpoint_search=replace(self.config.endpoint_search, pso_generation_size=3, pso_max_generations=1),
                device=replace(self.config.device, z_samples_initial=3, wavelength_samples_initial=3),
                optimizer=replace(self.config.optimizer, max_function_evaluations_per_grid=1, max_grid_refinements=0),
                propagation=replace(self.config.propagation, rk4_steps_initial=16, rk4_steps_max=64),
            )
            p = Path(folder)/'config.json'
            p.write_text(json.dumps(config.as_dict()))
            with patch.object(cli, 'LumapiBackend', FullSyntheticBackend), contextlib.redirect_stderr(io.StringIO()):
                out = cli.run_workflow(p, Path(folder)/'run')
            meta = json.loads((out/'metadata.json').read_text())
            self.assertEqual(meta['endpoint_search_assumption']['search_method'], 'python_pso')
            self.assertIn(f'n_cross={config.materials.liquid_cross_n:g}', meta['material_assumptions']['liquid'])
            self.assertIn(f'n_bar={config.materials.liquid_bar_n:g}', meta['material_assumptions']['liquid'])
            self.assertEqual(meta['endpoint_search_assumption']['pso_summary']['evaluations'], 3)
            self.assertTrue((out/'wavelength_performance.png').is_file())
            self.assertTrue((out/'performance_summary.csv').is_file())
            self.assertIn('delta_beta_cross_in_lambda_min_rad_per_m', (out/'endpoint_candidates.csv').read_text())

    def test_legacy_config_migration_preserves_bounds(self):
        raw = self.config.as_dict()
        raw['endpoint_search'] = {'method': 'lumerical_pso', 'pso_reset_random_generator': True,
                                  'width_preflight_cutoff_iterations': 12, 'width_auto_expand_max_m': 2e-6}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'legacy.json'
            p.write_text(json.dumps(raw))
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                migrated = load_config(p)
            self.assertEqual(len(notices), 1)
            self.assertEqual(migrated.endpoint_search.method, 'python_pso')
            self.assertEqual(migrated.geometry.w1_min_m, self.config.geometry.w1_min_m)
            self.assertEqual(migrated.geometry.w1_max_m, self.config.geometry.w1_max_m)
            self.assertNotIn('width_auto_expand_max_m', migrated.as_dict()['endpoint_search'])

    def test_endpoint_cli_and_validate_pso_use_python(self):
        with tempfile.TemporaryDirectory() as folder:
            config = replace(self.config, endpoint_search=replace(self.config.endpoint_search,
                             pso_generation_size=3, pso_max_generations=1))
            p = Path(folder)/'config.json'
            p.write_text(json.dumps(config.as_dict()))
            with patch.object(cli, 'LumapiBackend', SyntheticBackend), patch.object(cli, 'build_material_bank', return_value=self.bank), contextlib.redirect_stderr(io.StringIO()):
                out = cli.run_endpoint_only(p, Path(folder)/'endpoint')
            summary = json.loads((out/'endpoint_summary.json').read_text())
            self.assertEqual(summary['search_method'], 'python_pso')
            self.assertTrue(summary['cross_state_sign_verification']['passed'])
            self.assertEqual(summary['width_bounds_m'], [.3e-6, 2e-6])
            self.assertEqual(len(list(csv.DictReader(io.StringIO((out/'endpoint_detunings.csv').read_text())))), 8)
            for path in summary['files'].values():
                self.assertTrue(Path(path).is_file())
            with patch.object(cli, 'LumapiBackend', SyntheticBackend), patch.object(cli, 'build_material_bank', return_value=self.bank), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                rc = cli.main(['validate-pso', '--config', str(p), '--out', str(Path(folder)/'smoke/report.json')])
            self.assertEqual(rc, 0)
            report = json.loads((Path(folder)/'smoke/report.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['pso_summary']['evaluations'], 3)
            self.assertTrue(report['cross_state_sign_verification']['passed'])

    def test_dispatch_preserves_grid_choice(self):
        with patch('vlac.endpoint.search_endpoints_grid', return_value='grid') as grid:
            config = replace(self.config, endpoint_search=replace(self.config.endpoint_search, method='python_grid'))
            self.assertEqual(search_endpoints(None, config, self.bank), 'grid')
            grid.assert_called_once()
        with patch('vlac.python_pso.search_endpoints_python_pso', return_value='pso') as pso:
            self.assertEqual(search_endpoints(None, self.config, self.bank), 'pso')
            pso.assert_called_once()


if __name__ == '__main__':
    unittest.main()
