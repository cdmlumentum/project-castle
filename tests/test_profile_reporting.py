"""Block 2 time planning/report contracts; all optical data are SYNTHETIC."""
from dataclasses import replace
import csv
import json
import unittest
from unittest.mock import patch
import numpy as np

import test_workflow_runtime as fixtures
from vlac.profile_diagnostics import export_profile_diagnostics, comparison_summary, FILES
from vlac.profile_planning import ProfilePlanner
from vlac.optimize import optimize_profiles
from vlac.profiles import minimax_objective
from test_block_handoffs import RecordingBackend


class ProfileReportingTests(unittest.TestCase):
    setUp = fixtures.CharacterizationTests.setUp
    def test_complex_residuals_reproduce_objective_and_geometry(self):
        # Imaginary kappa must contribute to errors, even when Re(kappa) fits.
        solution = self.solution
        solution.cross_coefficients[0][1].kappa += 500j
        solution.objective = minimax_objective(solution.cross_coefficients,solution.z_m,
            self.config.device.length_m,self.endpoint.delta_beta_max_rad_per_m)[0]
        diagnostics = export_profile_diagnostics(self.root,self.config,self.endpoint,solution)
        self.assertEqual(diagnostics['files'],list(FILES))
        self.assertTrue(all((self.root/f).is_file() for f in FILES))
        comparison = json.loads((self.root/FILES[2]).read_text())
        self.assertAlmostEqual(comparison['objective_achieved'],solution.objective)
        self.assertEqual(comparison['by_wavelength'][0]['kappa_max_abs_imag_rad_per_m'],500.)
        geometry = json.loads((self.root/FILES[0]).read_text())
        self.assertEqual(geometry['fixed_upper_endpoint_widths_m'],[self.endpoint.w1_input_m,self.endpoint.w1_output_m])
        with (self.root/FILES[3]).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows),len(solution.z_m)*len(solution.wavelengths_m))
        self.assertTrue((self.root/FILES[5]).read_bytes().startswith(b'%PDF'))
        with self.assertRaisesRegex(ValueError,'disagrees'):
            comparison_summary(self.config,self.endpoint,replace(solution,objective=solution.objective+1))

    def test_time_planned_finish_still_checks_grid_and_returns_solution(self):
        cfg = replace(self.config,optimizer=replace(self.config.optimizer,
            complete_within_seconds=1000.,max_function_evaluations_per_grid=120))
        backend = RecordingBackend(cfg)
        with patch.object(ProfilePlanner,'allows',return_value=False):
            solution = optimize_profiles(backend,cfg,self.bank,self.endpoint)
        self.assertTrue(solution.optimizer_history[-1]['planned_finalization'])
        self.assertFalse(solution.optimizer_history[-1]['success'])
        self.assertTrue(solution.grid_convergence_history)
        self.assertEqual(solution.w1_control_m[0],self.endpoint.w1_input_m)
        self.assertEqual(solution.w1_control_m[-1],self.endpoint.w1_output_m)
        self.assertAlmostEqual(solution.cross_coefficients[0][0].kappa.real,100.,places=7)
        self.assertGreater(len(backend.coefficient_calls),len(solution.z_m)*len(solution.wavelengths_m))


class PlannerTests(unittest.TestCase):
    def test_reserves_validation_and_exports_before_more_optimization(self):
        with patch('vlac.profile_planning.time.monotonic',return_value=0.):
            planner = ProfilePlanner(5000.)
            planner.observe(100.,10)
            self.assertTrue(planner.allows(10,18))
        with patch('vlac.profile_planning.time.monotonic',return_value=3700.):
            self.assertFalse(planner.allows(10,18))
            self.assertFalse(planner.allows(18,34))
        planner.observe(1.,10)
        self.assertEqual(planner.seconds_per_sample,10.)

