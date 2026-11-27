"""Regression checks for balanced increasing endpoints and full-width slabs.

All modes are synthetic. No licensed MODE result is asserted by these tests.
"""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
import warnings

import numpy as np

from test_python_pso import ROOT, SyntheticBackend
from vlac.adaptive_endpoint import search_endpoints_adaptive
from vlac.config import load_config
from vlac.endpoint_constraints import (EndpointConstraintError, endpoint_constraint_report,
                                       require_endpoint_constraints)
from vlac.geometry import make_geometry, rectangles_for_cross_section
from vlac.materials import build_material_bank
from vlac.modes import ModeSnapshot
from vlac.profiles import target_profiles
from vlac.python_pso import search_endpoints_python_pso
from vlac.solver import LumapiBackend


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'config/python_joint.json')
        self.bank = build_material_bank(self.config)

    @staticmethod
    def records(widths=(.6e-6,1.7e-6)):
        return [dict(end=end,state=state,w1_m=width,wavelength_m=wavelength,
                     delta_beta_real_rad_per_m=-100. if end=='input' else 100.)
                for end,width in zip(('input','output'),widths)
                for state in ('cross','bar') for wavelength in (1.26e-6,1.34e-6)]

    def test_balance_checks_each_wavelength_and_uses_mean_denominator(self):
        rows = self.records()
        self.assertTrue(require_endpoint_constraints(rows,self.config.endpoint_search)['passed'])
        changed = next(r for r in rows if r['end']=='output' and r['state']=='cross' and r['wavelength_m']==1.34e-6)
        changed['delta_beta_real_rad_per_m'] = 120.
        report = endpoint_constraint_report(rows,self.config.endpoint_search)
        self.assertTrue(report['cross_state_sign_verification']['passed'])
        self.assertAlmostEqual(report['magnitude_balance']['maximum_relative_mismatch'],40/220)
        self.assertEqual([r['passed'] for r in report['magnitude_balance']['checks']],[True,False])
        with self.assertRaises(EndpointConstraintError):
            require_endpoint_constraints(rows,self.config.endpoint_search)

    def test_width_order_is_checked_even_when_signs_and_balance_pass(self):
        for widths in ((1e-6,1e-6),(1.7e-6,.6e-6)):
            with self.subTest(widths=widths):
                report = endpoint_constraint_report(self.records(widths),self.config.endpoint_search)
                self.assertTrue(report['cross_state_sign_verification']['passed'])
                self.assertTrue(report['magnitude_balance']['passed'])
                self.assertFalse(report['width_order']['passed'])
                with self.assertRaisesRegex(EndpointConstraintError,'endpoint_width_order'):
                    require_endpoint_constraints(self.records(widths),self.config.endpoint_search)

    def test_pair_search_finds_balanced_widths_absent_from_initial_table(self):
        class Asymmetric(SyntheticBackend):
            def solve_isolated_mode(self,g,wavelength,*,guide,use_cache=True,reference=None,**kwargs):
                self.calls.append((guide,g,wavelength,use_cache,reference))
                u = (g.w1_m-self.config.geometry.w1_min_m)/(self.config.geometry.w1_max_m-self.config.geometry.w1_min_m)
                neff = 1.8 if guide=='lower' else 1.8+.02*(u-.3)
                return ModeSnapshot(complex(neff),np.ones((4,4))/4,np.zeros((4,4)),1.,.8,'asymmetric analytic'),1.
        config = replace(self.config,endpoint_search=replace(self.config.endpoint_search,
                         adaptive_initial_widths=5,adaptive_max_widths=33,cross_magnitude_balance_rtol=.01))
        with tempfile.TemporaryDirectory() as folder:
            result = search_endpoints_adaptive(Asymmetric(config),config,self.bank,output_dir=folder)
            report = require_endpoint_constraints(result.endpoint_records,config.endpoint_search)
            self.assertLessEqual(report['magnitude_balance']['maximum_relative_mismatch'],.01)
            theoretical = 2*np.pi*.006/config.device.wavelength_max_m
            self.assertGreaterEqual(result.delta_beta_max_rad_per_m,.97*theoretical)
            self.assertAlmostEqual(result.w1_input_m/1e-6,.3,delta=.01)
            self.assertAlmostEqual(result.w1_output_m/1e-6,1.32,delta=.015)
            pairs = (Path(folder)/'endpoint_pairs.csv').read_text()
            self.assertIn('maximum_relative_mismatch',pairs)

    def test_pair_failure_is_distinct_from_missing_sign_and_does_not_relax_limit(self):
        class Incompatible(SyntheticBackend):
            def solve_isolated_mode(self,g,wavelength,*,guide,**kwargs):
                mode,overlap = super().solve_isolated_mode(g,wavelength,guide=guide,**kwargs)
                if guide=='upper': mode.neff = 1.79 if g.w1_m<1.1e-6 else 1.83
                return mode,overlap
        config = replace(self.config,endpoint_search=replace(self.config.endpoint_search,adaptive_max_widths=13))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'endpoint_adaptive_best.json').write_text('{"stale":true}')
            with self.assertRaisesRegex(RuntimeError,'no sampled pair satisfying'):
                search_endpoints_adaptive(Incompatible(config),config,self.bank,output_dir=out)
            summary = json.loads((out/'endpoint_adaptive_summary.json').read_text())
            self.assertGreater(summary['input_feasible_samples'],0)
            self.assertGreater(summary['output_feasible_samples'],0)
            self.assertEqual(summary['feasible_sampled_pairs'],0)
            self.assertEqual(summary['cross_magnitude_balance_rtol'],.1)
            self.assertFalse((out/'endpoint_adaptive_best.json').exists())

    def test_fresh_balance_failure_cannot_publish_adaptive_or_pso_winner(self):
        class Changed(SyntheticBackend):
            def solve_isolated_mode(self,g,wavelength,*,guide,use_cache=True,**kwargs):
                mode,overlap = super().solve_isolated_mode(g,wavelength,guide=guide,use_cache=use_cache,**kwargs)
                midpoint = (self.config.geometry.w1_min_m+self.config.geometry.w1_max_m)/2
                if not use_cache and guide=='upper' and g.w1_m<midpoint:
                    mode.neff = 1.8+.5*(mode.neff-1.8)
                return mode,overlap
        for search,prefix in ((search_endpoints_adaptive,'endpoint_adaptive'),(search_endpoints_python_pso,'endpoint_pso')):
            with self.subTest(search=prefix),tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(EndpointConstraintError,'cross_state_magnitude_balance'):
                    search(Changed(self.config),self.config,self.bank,validation_only=True,output_dir=folder)
                report = json.loads((Path(folder)/(prefix+'_verification.json')).read_text())
                self.assertEqual(report['status'],'failed')
                self.assertTrue(report['cross_state_sign_verification']['passed'])
                self.assertFalse(report['endpoint_constraint_verification']['magnitude_balance']['passed'])
                self.assertFalse((Path(folder)/(prefix+'_best.json')).exists())

    def test_both_slabs_reach_domain_edges_at_every_offset_and_domain_size(self):
        for scale in (1.,1.5):
            backend = object.__new__(LumapiBackend)
            backend.config,backend.mesh_accuracy = self.config,self.config.solver.mesh_accuracy
            backend._build_domain(scale)
            for span in (10e-6,backend.x_span_m):
                for offset in (-8e-6,0.,8e-6):
                    geometry = make_geometry(self.config.geometry,w1_m=.3e-6,h_rib_m=175e-9,h_gap_m=1e-6,
                                             d_c2c_m=offset,state='cross',liquid_index=1.55)
                    rects = rectangles_for_cross_section(geometry,x_span_m=span,y_min_m=backend.y_min_m,y_max_m=backend.y_max_m)
                    for name in ('upper_slab','lower_slab'):
                        rect = next(r for r in rects if r.name==name)
                        self.assertAlmostEqual(rect.x_center_m-rect.x_span_m/2,-span/2)
                        self.assertAlmostEqual(rect.x_center_m+rect.x_span_m/2,span/2)
                    self.assertEqual(next(r for r in rects if r.name=='upper_ridge').x_center_m,offset)
                    no_slab = rectangles_for_cross_section(replace(geometry,h_rib_m=0),x_span_m=span,
                                                          y_min_m=backend.y_min_m,y_max_m=backend.y_max_m)
                    self.assertFalse(any(r.name.endswith('_slab') for r in no_slab))

    def test_legacy_overhang_migrates_to_margin_and_target_has_correct_orientation(self):
        raw = self.config.as_dict()
        raw['geometry'].update(domain_margin_x_m=2e-6,rib_slab_overhang_m=2e-6,rib_slab_overhang_is_provisional=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'legacy.json';path.write_text(json.dumps(raw))
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');config=load_config(path)
            self.assertEqual(len(notices),1)
            self.assertEqual(config.geometry.domain_margin_x_m,4e-6)
            self.assertNotIn('rib_slab_overhang_m',config.as_dict()['geometry'])
        delta,kappa = target_profiles(np.array([0.,.5,1.]),1.,100.)
        np.testing.assert_allclose(delta,[-100.,0.,100.],atol=1e-12)
        np.testing.assert_allclose(kappa,[0.,50.,0.],atol=1e-12)
        for bad in (0.,2.,float('nan')):
            config = replace(self.config,endpoint_search=replace(self.config.endpoint_search,cross_magnitude_balance_rtol=bad))
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'balance'):
                config.validate()


if __name__=='__main__':unittest.main()
