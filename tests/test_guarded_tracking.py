"""Synthetic optical fixtures only; no physical device claims."""
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

from vlac.config import load_config
from vlac.geometry import ClampedCubicProfile
from vlac.modes import ModeSnapshot, ModeReference, field_overlap, extract_coupling
from vlac.monotone import controls, linear_coordinates
from vlac.tracking import (CandidateRejected, EndpointConsistencyError, select_isolated, require_candidate,
                           endpoint_consistency, enforce_endpoint_report, require_anchor_match)
from vlac.profiles import _profile_coefficients
from vlac.materials import build_material_bank
from vlac.native_backend import NativeLumapiBackend
from vlac.native_mode import prepare_bundle
from vlac.block_io import physics_signature
from vlac.geometry import make_geometry
from vlac.optimize import optimize_profiles
from unittest.mock import patch
import tempfile
import json
from vlac import blocks
from vlac.progress import ProgressReporter

ROOT = Path(__file__).resolve().parents[1]


def mode(center=0., odd=False):
    x=np.linspace(-12e-6,12e-6,241)
    y=np.linspace(-1e-6,1e-6,41)
    xx,yy=np.meshgrid(x-center,y,indexing='ij')
    ex=np.exp(-((xx/.6e-6)**2+(yy/.3e-6)**2))
    if odd: ex=ex*xx/.6e-6
    return ModeSnapshot(1.75,ex,np.zeros_like(ex),1.,.8,'SYNTHETIC',x,y,center)


class GuardedTrackingTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config/native_monotone.json')

    def test_translated_fundamental_beats_laboratory_frame_distractor(self):
        old=mode(-8e-6)
        ref=ModeReference(old,old,(old,old))
        good=mode(1e-6)
        # A fixed laboratory-frame remnant would beat the moving ridge in v23.
        distractor=replace(old,center_x_m=1e-6,identity='SYNTHETIC_WRONG_BRANCH')
        self.assertLess(field_overlap(old,good),.01)
        selected,overlap=select_isolated([distractor,good],(ref,),'upper',self.config)
        self.assertIs(selected,good)
        self.assertGreater(overlap,.999)

    def test_rejected_reference_and_ineligible_modes_do_not_advance(self):
        m=mode()
        ref=ModeReference(m,m,(m,m),valid=False)
        with self.assertRaisesRegex(CandidateRejected,'rejected mode reference'):
            select_isolated([m],(ref,),'upper',self.config)
        with self.assertRaises(CandidateRejected):
            select_isolated([replace(m,te_fraction=.1)],(),'upper',self.config)
        with self.assertRaises(CandidateRejected):
            select_isolated([replace(m,ex=m.ex*0,ey=m.ey*0)],(),'upper',self.config)

    def test_domain_warning_does_not_relax_sign_or_field_identity(self):
        cfg=self.config
        cfg.solver.endpoint_domain_policy='warn_exploratory'
        enforce_endpoint_report({'passed':False,'detuning_signs_passed':True},cfg)
        with self.assertRaises(EndpointConsistencyError):
            enforce_endpoint_report({'passed':False,'detuning_signs_passed':False},cfg)
        with self.assertRaises(EndpointConsistencyError):
            require_anchor_match(mode(0,odd=True),mode(),cfg)
        cfg.solver.endpoint_domain_policy='strict'
        with self.assertRaises(EndpointConsistencyError):
            enforce_endpoint_report({'passed':False,'detuning_signs_passed':True},cfg)

    def test_domain_warning_is_only_permitted_with_explicit_short_budgets(self):
        cfg=self.config
        cfg.solver.endpoint_domain_policy='warn_exploratory'
        with self.assertRaisesRegex(ValueError,'warn_exploratory requires'):
            cfg.validate()
        cfg.optimizer.max_function_evaluations_per_grid=6
        cfg.optimizer.max_grid_refinements=0
        cfg.execution.max_solves=120
        cfg.execution.max_seconds=2700
        cfg.validate()
        cfg.optimizer.max_function_evaluations_per_grid=7
        with self.assertRaises(ValueError): cfg.validate()

    def test_fixed_endpoint_coupling_preflight_cannot_be_bypassed_by_warning(self):
        cfg=self.config
        cfg.geometry.d_c2c_input_m=cfg.geometry.d_c2c_output_m=0.
        cfg.solver.endpoint_domain_policy='warn_exploratory'
        cfg.optimizer.max_function_evaluations_per_grid=6
        cfg.optimizer.max_grid_refinements=0
        cfg.execution.max_solves=120
        cfg.execution.max_seconds=2700
        scale=2*np.pi/1.26e-6
        records=[{'state':'cross','end':end,'wavelength_m':1.26e-6,
            'beta_upper_real_rad_per_m':scale*1.75,'beta_upper_imag_rad_per_m':0.,
            'beta_lower_real_rad_per_m':scale*1.74,'beta_lower_imag_rad_per_m':0.,
            'delta_beta_real_rad_per_m':scale*.01} for end in ('input','output')]
        endpoint=SimpleNamespace(wavelengths_m=[1.26e-6],w1_input_m=1.1e-6,w1_output_m=1.8e-6,
            h_rib_m=175e-9,h_gap_m=1e-6,endpoint_records=records)
        def isolated(g,wavelength,*,guide,compact_x):
            center=g.d_c2c_m if guide=='upper' and not compact_x else 0.
            return replace(mode(center),neff=1.75 if guide=='upper' else 1.74),1.
        bad=extract_coupling(scale*1.75,scale*1.74,scale*1.7451,scale*1.7449)
        m=mode()
        with tempfile.TemporaryDirectory() as folder,NativeLumapiBackend(cfg) as backend:
            backend.solve_isolated_mode=Mock(side_effect=isolated)
            backend._guarded_coefficients=Mock(return_value=(bad,ModeReference(m,m,(m,m),valid=False)))
            progress=SimpleNamespace(output_dir=Path(folder),checkpoint=Mock(),note=Mock())
            with self.assertRaisesRegex(EndpointConsistencyError,'Fixed-endpoint coefficient extraction'):
                backend.require_endpoint_consistency(endpoint,progress=progress)
            report=json.loads((Path(folder)/'endpoint_tracking_preflight.json').read_text())
            self.assertTrue(report['field_identity_passed'])
            self.assertFalse(report['coefficient_extraction_passed'])
            self.assertEqual(len(report['coefficient_checks']),2)
            self.assertEqual({c['end'] for c in report['anchor_checks']},{'input','output'})
            self.assertEqual(backend.budget.solves,0)

    def test_low_overlap_is_not_cured_by_high_neff(self):
        m=mode()
        ref=ModeReference(m,m,(m,m))
        with self.assertRaises(CandidateRejected):
            select_isolated([replace(mode(0,odd=True),neff=2.)],(ref,),'upper',self.config)

    def test_full_complex_extraction_is_rejected_without_clipping(self):
        c=extract_coupling(4.,2.,3.2,2.8)
        self.assertNotEqual(c.kappa.imag,0)
        with self.assertRaises(CandidateRejected): require_candidate(c,self.config)
        self.assertNotEqual(c.kappa.imag,0)

    def test_guarded_raw_selection_preserves_complex_branch_convention(self):
        cfg=self.config
        backend=NativeLumapiBackend(cfg)
        a=mode(-2e-6)
        b=replace(mode(2e-6),neff=1.74)
        plus=replace(a,neff=1.76)
        minus=replace(b,neff=1.73)
        packets=[{'n':len(modes),'items':[asdict(m) for m in modes]} for modes in ([a],[b],[plus,minus])]
        backend._raw_packets=Mock(return_value=packets)
        geometry=make_geometry(cfg.geometry,w1_m=1.5e-6,h_rib_m=175e-9,h_gap_m=1e-6,
                               d_c2c_m=0.,state='cross',liquid_index=1.55)
        actual,reference=backend.evaluate_coefficients(geometry,1.26e-6,previous_kappa=-1.)
        scale=2*np.pi/1.26e-6
        expected=extract_coupling(*(scale*m.neff for m in (a,b,plus,minus)),previous_kappa=-1.)
        self.assertAlmostEqual(actual.kappa.real,expected.kappa.real)
        self.assertLess(actual.kappa.real,0)
        self.assertTrue(reference.valid)
        backend.close()

    def test_diagnostic_flagged_coefficients_cannot_be_used_as_references(self):
        cfg=self.config
        backend=NativeLumapiBackend(cfg)
        a=mode(-2e-6)
        b=replace(mode(2e-6),neff=1.74)
        pair=[replace(a,neff=1.7451),replace(b,neff=1.7449)]
        packets=[{'n':len(modes),'items':[asdict(m) for m in modes]} for modes in ([a],[b],pair)]
        backend._raw_packets=Mock(return_value=packets)
        geometry=make_geometry(cfg.geometry,w1_m=1.5e-6,h_rib_m=175e-9,h_gap_m=1e-6,
                               d_c2c_m=0.,state='cross',liquid_index=1.55)
        with self.assertRaises(CandidateRejected):
            backend.evaluate_coefficients(geometry,1.26e-6)
        coefficient,reference=backend._guarded_coefficients(geometry,1.26e-6,(),None,diagnostic=True)
        self.assertFalse(reference.valid)
        self.assertNotEqual(coefficient.kappa.imag,0.)
        with self.assertRaisesRegex(CandidateRejected,'rejected mode reference'):
            backend.evaluate_coefficients(geometry,1.26e-6,references=(reference,))
        backend.close()

    def test_named_method_is_handoff_distinct_and_standalone_does_not_substitute(self):
        legacy=load_config(ROOT/'config/python_joint.json')
        # Selection changes do not invalidate an unchanged Block 1 geometry.
        self.assertEqual(physics_signature(legacy,1),physics_signature(self.config,1))
        self.assertNotEqual(physics_signature(legacy,2),physics_signature(self.config,2))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'standalone coordinate_pattern'):
                prepare_bundle(self.config,folder)

    def test_endpoint_gate_checks_absolute_modes_even_when_detuning_sign_matches(self):
        scale=2*np.pi/1.26e-6
        endpoint=SimpleNamespace(endpoint_records=[{'state':'cross','end':end,'wavelength_m':1.26e-6,
            'beta_upper_real_rad_per_m':scale*1.75,'beta_upper_imag_rad_per_m':0.,
            'beta_lower_real_rad_per_m':scale*1.7,'beta_lower_imag_rad_per_m':0.,
            'delta_beta_real_rad_per_m':scale*.05} for end in ('input','output')])
        bad=SimpleNamespace(beta_upper=scale*1.85,beta_lower=scale*1.8,delta_beta=scale*.05)
        self.assertFalse(endpoint_consistency([[bad,bad]],[1.26e-6],endpoint,self.config)['passed'])

    def test_midpoint_recovery_uses_only_last_accepted_reference(self):
        cfg=self.config
        cfg.geometry.d_c2c_input_m=0.
        cfg.geometry.d_c2c_output_m=1e-6
        endpoint=SimpleNamespace(h_rib_m=175e-9,h_gap_m=1e-6,endpoint_records=[])
        accepted=[]
        class Backend:
            def evaluate_coefficients(self,g,w,*,references,previous_kappa):
                if references and abs(g.d_c2c_m-references[0].upper.center_x_m)>.26e-6:
                    raise CandidateRejected('SYNTHETIC coarse jump')
                m=mode(g.d_c2c_m)
                accepted.append(g.d_c2c_m)
                return extract_coupling(4.,2.,5.,1.),ModeReference(m,m,(m,m))
        # Isolate reference recovery from separately tested endpoint records.
        from unittest.mock import patch
        with patch('vlac.tracking.endpoint_consistency',return_value={'passed':True}):
            _profile_coefficients(Backend(),cfg,build_material_bank(cfg),endpoint,np.ones(4)*1e-6,
                np.linspace(0,1e-6,4),np.array([0.,cfg.device.length_m]),np.array([1.26e-6]),state='cross')
        np.testing.assert_allclose(accepted,np.arange(5)*.25e-6,atol=1e-18)

    def test_monotone_spline_derivative_for_both_directions_and_equal_ends(self):
        rng=np.random.default_rng(2007)
        for start,end in ((1.1e-6,1.8e-6),(1.8e-6,1.1e-6),(-8e-6,8e-6),(2.,2.)):
            for _ in range(20):
                values=controls(rng.random(4),start,end)
                self.assertEqual(values[0],start)
                self.assertEqual(values[-1],end)
                p=ClampedCubicProfile(1.5e-3,values)
                derivative=p.derivative(np.linspace(0,1.5e-3,1001))
                self.assertTrue(np.all(derivative*(1 if end>=start else -1)>=-1e-12))
        np.testing.assert_allclose(controls(linear_coordinates(6),0.,1.),np.linspace(0,1,6))

    def test_rejected_candidates_never_reach_the_coefficient_objective(self):
        cfg=self.config
        cfg.optimizer.max_function_evaluations_per_grid=1
        endpoint=SimpleNamespace(w1_input_m=1.1e-6,w1_output_m=1.8e-6,delta_beta_max_rad_per_m=1.)
        backend=SimpleNamespace(require_endpoint_consistency=Mock())
        def stop_after_one(objective, initial, **kwargs):
            objective(initial)
            return SimpleNamespace(x=initial,success=True,message='SYNTHETIC_TEST_ONLY')
        with patch('vlac.optimize.minimize',side_effect=stop_after_one), \
             patch('vlac.optimize.evaluate_design_profiles',side_effect=CandidateRejected('SYNTHETIC wrong branch')), \
             patch('vlac.optimize.minimax_objective') as objective:
            with self.assertRaisesRegex(RuntimeError,'No optically valid'):
                optimize_profiles(backend,cfg,build_material_bank(cfg),endpoint)
            objective.assert_not_called()

    def test_endpoint_gate_failure_retains_attempt_status_and_solve_counts(self):
        def rejected(*args,progress,**kwargs):
            progress.native_solves(3)
            raise CandidateRejected('SYNTHETIC endpoint mismatch')
        with tempfile.TemporaryDirectory() as folder:
            progress=ProgressReporter('synthetic gate test',folder)
            with patch('vlac.blocks.load_block'),patch('vlac.blocks._run_single_block',side_effect=rejected):
                with self.assertRaises(CandidateRejected):
                    blocks.run_single_block(2,self.config,folder,'SYNTHETIC',backend_factory=None,progress=progress)
            attempt=next((Path(folder)/'.attempts').iterdir())
            status=json.loads((attempt/'run_status.json').read_text())
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['findmodes_calls']['completed'],3)
            failure=json.loads((Path(folder)/'block2_attempt_failure.json').read_text())
            self.assertEqual(failure['runtime_diagnostics']['findmodes_calls']['completed'],3)

    def test_optimizer_uses_monotone_coordinates_and_validates_midpoints(self):
        cfg=self.config
        cfg.optimizer.max_grid_refinements=0
        endpoint=SimpleNamespace(w1_input_m=1.8e-6,w1_output_m=1.1e-6,delta_beta_max_rad_per_m=2.)
        backend=SimpleNamespace(require_endpoint_consistency=Mock())
        counts=[]
        def profiles(backend,config,bank,endpoint,w,dc,z,wavelengths,**kwargs):
            self.assertTrue(np.all(np.diff(w)<=0))
            self.assertTrue(np.all(np.diff(dc)>=0))
            np.testing.assert_array_equal(w[[0,-1]],[endpoint.w1_input_m,endpoint.w1_output_m])
            counts.append(len(z))
            return [[extract_coupling(4.,2.,5.,1.) for _ in z] for _ in wavelengths]
        def one_step(objective,initial,**kwargs):
            trial=np.linspace(.1,.9,len(initial))
            objective(trial)
            return SimpleNamespace(x=trial,success=True,message='SYNTHETIC_TEST_ONLY')
        with patch('vlac.optimize.minimize',side_effect=one_step), \
             patch('vlac.optimize.evaluate_design_profiles',side_effect=profiles):
            result=optimize_profiles(backend,cfg,build_material_bank(cfg),endpoint)
        self.assertEqual(counts,[5,9])
        self.assertTrue(result.grid_converged)
        self.assertEqual(result.optimizer_history[-1]['coordinate_system'],'monotone_stick_breaking_v1')

if __name__=='__main__': unittest.main()
