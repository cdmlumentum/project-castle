"""Offline grid/accounting contracts; any optical fixtures are synthetic."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vlac.block0 import base_profiles, domain_reference_profile
from vlac.config import load_config
from vlac.mesh_study import Study, StudyBackend, matching_grids, two_pass_choice
from vlac.materials import build_material_bank
from vlac.geometry import make_geometry

ROOT = Path(__file__).resolve().parents[1]


class MeshStudyTests(unittest.TestCase):
    def test_requested_counts_do_not_substitute_for_actual_registration(self):
        grid = {'x_m': np.arange(5)*1e-8,'y_m':np.arange(4)*2e-8}
        self.assertTrue(matching_grids(grid,grid)['passed'])
        shifted = {**grid,'x_m':grid['x_m']+1e-9}
        self.assertFalse(matching_grids(grid,shifted)['passed'])
        shorter = {**grid,'x_m':grid['x_m'][1:]}
        self.assertFalse(matching_grids(grid,shorter)['passed'])

    def test_domain_check_compares_common_optical_region(self):
        grid = {'x_m': np.arange(-3,4)*1e-8,'y_m':np.arange(-3,4)*1e-8}
        expanded = {'x_m': np.arange(-5,6)*1e-8,'y_m':np.arange(-5,6)*1e-8}
        self.assertFalse(matching_grids(grid,expanded)['passed'])
        self.assertTrue(matching_grids(grid,expanded,bounds={'x':(-2e-8,2e-8),'y':(-2e-8,2e-8)})['passed'])

    def test_two_successive_passes_required(self):
        self.assertIsNone(two_pass_choice([40,30,20],[{'passed':True},{'passed':False}]))
        self.assertEqual(two_pass_choice([40,30,20,15],
            [{'passed':False},{'passed':True},{'passed':True}]),(20,15))

    def test_removed_ridge_still_changes_shared_mesh_cache_identity(self):
        cfg = load_config(ROOT/'config/native_tabulated.json')
        backend = object.__new__(StudyBackend)
        backend.config,backend.bank = cfg,build_material_bank(cfg)
        backend.mesh_accuracy,backend.script_hash = cfg.solver.mesh_accuracy,'synthetic-contract'
        backend._build_domain()
        backend.study_profile = base_profiles(cfg)['coupled']
        geometry = make_geometry(cfg.geometry,w1_m=1.2e-6,d_c2c_m=-8e-6,
            h_gap_m=1e-6,h_rib_m=175e-9,state='cross',liquid_index=1.55)
        lower = lambda g: backend._raw_key(backend._raw_request(g,1.26e-6,False,True))
        self.assertNotEqual(lower(geometry),lower(replace(geometry,w1_m=1.3e-6)))
        self.assertNotEqual(lower(geometry),lower(replace(geometry,d_c2c_m=0.)))
        first = lower(geometry)
        backend.retain_slabs = False
        self.assertNotEqual(first,lower(geometry))

    def test_uniform_background_domain_expands_on_same_lattice(self):
        cfg = load_config(ROOT/'config/native_tabulated.json')
        base = base_profiles(cfg)['coupled']
        for axis in ('domain_x','domain_top','domain_bottom'):
            expanded = domain_reference_profile(cfg,base,'coupled',axis,1.5)
            self.assertAlmostEqual(expanded.x_span_m/expanded.nx,base.x_span_m/base.nx,places=18)
            self.assertAlmostEqual((expanded.y_max_m-expanded.y_min_m)/expanded.ny,
                                  (base.y_max_m-base.y_min_m)/base.ny,places=18)

    def orchestration_fixture(self):
        # Control-flow fixture only: no synthetic optical values or MODE session.
        cfg = load_config(ROOT/'config/native_tabulated.json')
        study = object.__new__(Study)
        study.config,study.base = cfg,base_profiles(cfg)['coupled']
        study.probes = [dict(name='probe')]
        study.cases = [dict(name=f'{s}_{n}_{w}') for s in ('cross','bar')
                       for n in ('input','center','shoulder','output') for w in (1260,1340)]
        study.report = dict(axes={},numerical_convergence_passed=False)
        study.backend,study.progress = Mock(),Mock()
        study.out = Path('synthetic_control_flow_only')
        for name in ('save','sample','compare'):
            setattr(study,name,Mock())
        study.group = Mock(return_value=['archived-sample-marker'])
        study.compare_group = Mock(return_value=dict(passed=True))
        return study

    def test_success_path_checks_both_states_then_domains_and_pml(self):
        study = self.orchestration_fixture()
        candidate = replace(study.base,local_dx_m=20e-9,local_dy_m=20e-9)
        reference = replace(candidate,local_dx_m=15e-9,local_dy_m=15e-9)
        study.axis = Mock(return_value=((candidate,reference),[],[]))
        with patch('vlac.mesh_study.render_report'):
            study.execute()
        self.assertTrue(study.report['numerical_convergence_passed'])
        self.assertEqual(study.report['status'],'scoped_numerical_pass')
        full_checks = [c for c in study.group.call_args_list
                       if c.args[0] in ('combined_candidate','combined_reference','final_reference')]
        self.assertEqual(len(full_checks),3)
        self.assertTrue(all(len(c.args[2])==16 for c in full_checks))
        for axis in ('domain_x','domain_top','domain_bottom'):
            self.assertTrue(study.report['axes'][axis]['passed'])
        study.backend.close.assert_called_once()

    def test_failed_axis_still_diagnoses_other_axis_but_blocks_downstream(self):
        study = self.orchestration_fixture()
        study.axis = Mock(return_value=(None,[],[]))
        with patch('vlac.mesh_study.render_report'):
            study.execute()
        self.assertEqual(study.axis.call_count,2)
        study.group.assert_not_called()
        self.assertEqual(study.report['status'],'mesh_gate_failed')
        self.assertFalse(study.report['numerical_convergence_passed'])
        study.backend.close.assert_called_once()


if __name__=='__main__': unittest.main()
