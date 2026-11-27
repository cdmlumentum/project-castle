"""Table/scaling/uncertainty regressions. All optical data are SYNTHETIC."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from vlac.config import load_config
from vlac.geometry import make_geometry, ClampedCubicProfile
from vlac.materials import build_material_bank
from vlac.modes import extract_coupling, ModeReference
from vlac.tabulated import select_coupling_scale, pchip, OpticalTables, optimize_tabulated, signed_offset_nodes
from vlac.coupling_uncertainty import apply_zero_compatibility, validate_zero_compatibility, evidence_identity, case_id
from vlac.block_io import write_json, write_artifact, load_block, validate_solution, decode_coefficients, coefficient_data, solution_kappa_max
from vlac.tracking import require_candidate, CandidateRejected
from vlac.profile_diagnostics import comparison_summary
import test_sequential_profiles as fixtures
from test_sequential_profiles import SyntheticBackend

ROOT = Path(__file__).resolve().parents[1]


class SyntheticTableBackend(SyntheticBackend):
    """Analytic synthetic beta/kappa, with deliberately limited peak coupling."""
    def __init__(self):
        super().__init__()
        self.coefficient_calls = []
        self.reject = False

    def evaluate_coefficients(self,geometry,wavelength,**kwargs):
        self.coefficient_calls.append((geometry,wavelength))
        if self.reject: raise CandidateRejected('SYNTHETIC rejected sample')
        detuning = (geometry.w1_m-1.3e-6)/.5e-6*2000
        coupling = 650*(1-.05*(wavelength-1.26e-6)/.08e-6)*np.exp(-(geometry.d_c2c_m/4e-6)**2)
        half = np.sqrt((detuning/2)**2+coupling**2)
        c = extract_coupling(1e7+detuning,1e7,1e7+detuning/2+half,1e7+detuning/2-half)
        mode,_ = self.solve_isolated_mode(geometry,wavelength,guide='upper',reference=None,compact_x=False)
        return c,ModeReference(mode,mode,(mode,mode),valid=True)

    def evaluate_profile_grid(self,config,bank,endpoint,w,dc,z,wavelengths,**kwargs):
        self.profile_calls.append((w.copy(),dc.copy(),np.array(z),np.array(wavelengths)))
        wp,dp = ClampedCubicProfile(config.device.length_m,w),ClampedCubicProfile(config.device.length_m,dc)
        return [[self.evaluate_coefficients(make_geometry(config.geometry,w1_m=float(wp(zi)),d_c2c_m=float(dp(zi)),
                    h_gap_m=endpoint.h_gap_m,h_rib_m=endpoint.h_rib_m,state='cross',liquid_index=1.55),l)[0]
                 for zi in z] for l in wavelengths]


class TableTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SequentialTests()
        fixture.setUp()
        self.cfg,self.endpoint = fixture.config,fixture.endpoint
        self.cfg.optimizer.method = 'tabulated_width_offset_v1'
        self.cfg.optimizer.table_initial_widths = 5
        self.cfg.optimizer.table_max_widths = 5
        self.cfg.optimizer.table_initial_offsets = 3
        self.cfg.optimizer.table_max_offsets = 3
        self.bank = build_material_bank(self.cfg)
        self.backend = SyntheticTableBackend()

    def test_scale_uses_both_wavelengths_and_whole_envelope(self):
        result = select_coupling_scale(2000,[[800,800,800],[800,600,800]],[0,1,0])
        self.assertEqual(result['kappa_max_rad_per_m'],600)
        self.assertAlmostEqual(result['delta_beta_max_over_kappa_max'],10/3)
        self.assertTrue(result['coupling_target_reduced'])
        result = select_coupling_scale(2000,[[800,1200,800],[800,1100,800]],[0,1,0])
        self.assertEqual(result['kappa_max_rad_per_m'],1000)
        with self.assertRaises(CandidateRejected): select_coupling_scale(2000,[[0,0,0]],[0,1,0])

    def test_configuration_and_asymmetric_offset_allowance(self):
        cfg = load_config(ROOT/'config/native_tabulated.json')
        self.assertEqual(cfg.optimizer.method,'tabulated_width_offset_v1')
        self.assertEqual(cfg.execution.max_seconds,28800)
        for count in (3,6,7):
            nodes = signed_offset_nodes(-3e-6,8e-6,count)
            self.assertEqual(len(nodes),count)
            np.testing.assert_array_equal(nodes[[0,-1]],[-3e-6,8e-6])
            self.assertIn(0.,nodes)
            self.assertTrue(np.all(np.diff(nodes)>0))

    def test_no_extrapolation_and_complex_part_preserved(self):
        np.testing.assert_allclose(pchip([0,1,2],np.array([0,1,2])*(1+2j),[.5,1.5]),[.5+1j,1.5+3j])
        with self.assertRaises(ValueError): pchip([0,1,2],[0,1,2],-1)

    def test_width_table_cached_and_rejects_nonmonotone_branch(self):
        tables = OpticalTables(self.backend,self.cfg,self.bank,self.endpoint,None)
        nodes,delta,ok = tables.width_table()
        self.assertTrue(ok)
        self.assertEqual(delta.shape,(2,5))
        count = len(self.backend.isolated_calls)
        tables.width_table()
        self.assertEqual(count,len(self.backend.isolated_calls))
        self.assertEqual(self.backend.coefficient_calls,[])
        original = tables.delta
        tables.delta = lambda w,l: original(w,l) if w!=nodes[2] else 1e9
        with self.assertRaisesRegex(CandidateRejected,'monotone branch'): tables.width_table()

    def test_invalid_coupling_never_saved(self):
        tables = OpticalTables(self.backend,self.cfg,self.bank,self.endpoint,None)
        self.backend.reject = True
        with self.assertRaises(CandidateRejected): tables.coupling(1.3e-6,0.,1.26e-6)
        self.assertFalse(tables.coupling_cache)
        self.assertFalse(tables.references)

    def test_complete_synthetic_pipeline_handoff_and_reporting(self):
        # One cheap optimizer evaluation suffices to exercise contracts here;
        # this test makes no numerical/physical convergence claim.
        def one(function,initial,**kwargs):
            value = function(initial)
            return SimpleNamespace(success=True,message='SYNTHETIC optimizer',nfev=1,fun=value,x=initial)
        with patch('vlac.tabulated.minimize',side_effect=one):
            result = optimize_tabulated(self.backend,self.cfg,self.bank,self.endpoint)
        self.assertTrue(result.target_metadata['coupling_target_reduced'])
        self.assertAlmostEqual(solution_kappa_max(result,self.endpoint,self.cfg),617.5,places=5)
        self.assertTrue(all(l in (1.26e-6,1.34e-6) for _,l in self.backend.coefficient_calls))
        self.assertTrue(np.all(np.diff(result.w1_control_m)>=0))
        self.assertTrue(np.all(np.diff(result.d_c2c_control_m)>=0))
        validate_solution(result,self.endpoint,self.cfg)
        report = comparison_summary(self.cfg,self.endpoint,result)
        self.assertEqual(report['K_rad_per_m'],result.target_metadata['kappa_max_rad_per_m'])
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_artifact(directory,2,self.cfg,self.endpoint,solution=result)
            self.assertEqual(artifact['schema_version'],3)
            _,loaded,_,_ = load_block(directory,2,self.cfg)
            self.assertEqual(loaded.target_metadata,result.target_metadata)
        result.kappa_target_rad_per_m *= .9
        with self.assertRaisesRegex(ValueError,'targets are inconsistent'): validate_solution(result,self.endpoint,self.cfg)

    def test_table_checkpoint_restores_optical_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            first = OpticalTables(self.backend,self.cfg,self.bank,self.endpoint,None,directory)
            value = first.delta(1.3e-6,1.26e-6)
            first.save(checkpoint=True)
            count = len(self.backend.isolated_calls)
            second = OpticalTables(self.backend,self.cfg,self.bank,self.endpoint,None,directory)
            self.assertEqual(second.delta(1.3e-6,1.26e-6),value)
            self.assertEqual(len(self.backend.isolated_calls),count)


class ZeroBoundTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT/'config/native_sequential.json')
        self.geometry = make_geometry(self.cfg.geometry,w1_m=1.5e-6,d_c2c_m=8e-6,
            h_gap_m=1e-6,h_rib_m=175e-9,state='cross',liquid_index=1.55)
        # SYNTHETIC q=-1, a much tighter bound than the measured project issue.
        self.coefficient = extract_coupling(102.,98.,100+np.sqrt(3),100-np.sqrt(3))
        self.assertEqual(self.coefficient.extraction_flag,'negative_kappa_squared')

    def evidence(self,path,bound=1.1):
        q = self.coefficient.delta_pm**2-self.coefficient.delta_12**2
        self.cfg.solver.coupling_zero_evidence = str(path)
        # Create file first: physics identity hashes any configured evidence.
        write_json(path,{})
        record = {'raw_q_real':q.real,'raw_q_imag':q.imag,'q_absolute_error_bound':bound,
            'matched_background_checked':True,'mesh_domain_checked':True,
            'review_rationale':'SYNTHETIC test bound, not optical evidence.'}
        evidence = {'schema_version':1,'status':'reviewed_bound','identity':evidence_identity(self.cfg),
            'cases':{case_id(self.geometry,1.26e-6):record}}
        write_json(path,evidence)

    def test_no_evidence_no_clipping(self):
        actual = apply_zero_compatibility(self.coefficient,self.cfg,self.geometry,1.26e-6)
        self.assertEqual(actual.kappa,self.coefficient.kappa)
        with self.assertRaises(CandidateRejected): require_candidate(actual,self.cfg)

    def test_bound_must_cover_zero_be_small_and_exactly_match_case(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'evidence.json'
            self.evidence(path)
            actual = apply_zero_compatibility(self.coefficient,self.cfg,self.geometry,1.26e-6)
            self.assertEqual(actual.kappa,0j)
            self.assertTrue(validate_zero_compatibility(actual,self.cfg))
            require_candidate(actual,self.cfg)
            legacy = coefficient_data([[self.coefficient]])
            legacy[0][0].pop('zero_compatibility')
            self.assertIsNone(decode_coefficients(legacy)[0][0].zero_compatibility)
            self.assertEqual(apply_zero_compatibility(self.coefficient,self.cfg,self.geometry,1.34e-6).kappa,self.coefficient.kappa)
            self.evidence(path,.5)
            self.assertEqual(apply_zero_compatibility(self.coefficient,self.cfg,self.geometry,1.26e-6).kappa,self.coefficient.kappa)
            self.evidence(path,1e9)
            self.assertEqual(apply_zero_compatibility(self.coefficient,self.cfg,self.geometry,1.26e-6).kappa,self.coefficient.kappa)

    def test_tampered_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'evidence.json'
            self.evidence(path)
            actual = apply_zero_compatibility(self.coefficient,self.cfg,self.geometry,1.26e-6)
            raw = json.loads(path.read_text(encoding='utf-8'))
            raw['status'] = 'draft'
            write_json(path,raw)
            with self.assertRaises(ValueError): validate_zero_compatibility(actual,self.cfg)


if __name__=='__main__': unittest.main()
