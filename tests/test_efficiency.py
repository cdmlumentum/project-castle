"""Efficiency and correctness checks using synthetic modes and recording API stubs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import csv
import io
import json
import tempfile
import unittest

import numpy as np

from test_python_pso import SyntheticBackend, ROOT
from vlac import cli
from vlac.adaptive_endpoint import WidthEvaluator, search_endpoints_adaptive
from vlac.cache import SQLiteCache
from vlac.config import load_config
from vlac.endpoint import EndpointSearchResult, search_endpoints
from vlac.endpoint_constraints import require_cross_state_signs, require_endpoint_constraints, EndpointSignError
from vlac.geometry import make_geometry
from vlac.materials import build_material_bank
from vlac.modes import BareModeReference, ModeReference, ModeSnapshot, extract_coupling
from vlac.optimize import optimize_profiles
from vlac.progress import ProgressReporter
from vlac.solver import LumapiBackend


def backend_without_mode(config, bank, folder=None):
    b = object.__new__(LumapiBackend)
    b.config,b.bank,b.mesh_accuracy,b.progress = config,bank,config.solver.mesh_accuracy,None
    b.cache = SQLiteCache(Path(folder)/'cache.sqlite') if folder else None
    b._build_domain()
    return b


def geometry(config, width=1e-6, offset=0., state='cross', liquid=None):
    if liquid is None:
        liquid = config.materials.liquid_cross_n if state == 'cross' else config.materials.liquid_bar_n
    return make_geometry(config.geometry,w1_m=width,h_rib_m=175e-9,h_gap_m=1e-6,
                         d_c2c_m=offset,state=state,liquid_index=liquid)


def fixture_modes():
    ex=np.ones((16,16))/16
    a=ModeSnapshot(1.9+0j,ex,np.zeros_like(ex),1.,.8,'fixture1')
    b=ModeSnapshot(1.8+0j,ex,np.zeros_like(ex),1.,.7,'fixture2')
    return [a,b]


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config/python_joint.json')
        self.bank=build_material_bank(self.config)

    def test_shared_table_and_fresh_ten_modes(self):
        b=SyntheticBackend(self.config)
        with tempfile.TemporaryDirectory() as folder:
            result=search_endpoints_adaptive(b,self.config,self.bank,output_dir=folder)
            self.assertEqual(result.search_method,'adaptive_1d')
            self.assertTrue(require_cross_state_signs(result.endpoint_records)['passed'])
            self.assertEqual(len([c for c in b.calls if not c[3]]),10)
            self.assertLessEqual(result.search_summary['widths_evaluated'],33)
            upper=[(c[1].w1_m,c[1].state,c[2]) for c in b.calls if c[0]=='upper' and c[3]]
            self.assertEqual(len(upper),len(set(upper)))
            self.assertEqual(len([c for c in b.calls if c[0]=='lower' and c[3]]),2)
            rows=list(csv.DictReader(io.StringIO((Path(folder)/'endpoint_width_samples.csv').read_text())))
            best_in=max(float(r['bottleneck_rad_per_m']) for r in rows if r['eligible_end']=='input' and r['valid']=='True')
            best_out=max(float(r['bottleneck_rad_per_m']) for r in rows if r['eligible_end']=='output' and r['valid']=='True')
            self.assertLessEqual(result.delta_beta_max_rad_per_m,min(best_in,best_out)+1e-7)
            self.assertTrue(require_endpoint_constraints(result.endpoint_records,self.config.endpoint_search)["passed"])
            self.assertEqual(json.loads((Path(folder)/'endpoint_adaptive_verification.json').read_text())['status'],'passed')

    def test_known_separated_interior_maxima(self):
        class Sinusoidal(SyntheticBackend):
            def solve_isolated_mode(self,g,wavelength,*,guide,reference=None,compact_x=None,use_cache=True,**kwargs):
                self.calls.append((guide,g,wavelength,use_cache,reference))
                u=(g.w1_m-self.config.geometry.w1_min_m)/(self.config.geometry.w1_max_m-self.config.geometry.w1_min_m)
                shift=.02*np.sin(2*np.pi*u-np.pi)
                if g.state=='bar':shift*=.8
                mode=ModeSnapshot(complex(1.8 if guide=='lower' else 1.8+shift),
                                  np.ones((4,4))/4,np.zeros((4,4)),1.,.8,'synthetic sinusoid')
                return mode,1.
        with tempfile.TemporaryDirectory() as folder:
            result=search_endpoints_adaptive(Sinusoidal(self.config),self.config,self.bank,output_dir=folder)
        self.assertAlmostEqual(result.w1_input_m/1e-6,.725,delta=.005)
        self.assertAlmostEqual(result.w1_output_m/1e-6,1.575,delta=.005)
        self.assertAlmostEqual(result.delta_beta_max_rad_per_m,2*np.pi*.016/1.34e-6,delta=5.)

    def test_full_workflow_uses_adaptive_result(self):
        class Full(SyntheticBackend):
            def evaluate_coefficients(self,g,wavelength,**kwargs):
                q=np.sqrt(500.**2+100.**2)
                return extract_coupling(1e7+500.,1e7-500.,1e7+q,1e7-q),None
        with tempfile.TemporaryDirectory() as folder:
            config=replace(self.config,
                           materials=replace(self.config.materials,silicon_data_csv=str(ROOT/'data/green_si_2008_1260_1340.csv')),
                           optimizer=replace(self.config.optimizer,max_function_evaluations_per_grid=1,max_grid_refinements=0),
                           propagation=replace(self.config.propagation,rk4_steps_initial=16,rk4_steps_max=64))
            p=Path(folder)/'config.json';p.write_text(json.dumps(config.as_dict()))
            with patch.object(cli,'LumapiBackend',Full),contextlib.redirect_stderr(io.StringIO()):
                out=cli.run_workflow(p,Path(folder)/'run')
            metadata=json.loads((out/'metadata.json').read_text())
            self.assertEqual(metadata['endpoint_search_assumption']['search_summary']['backend'],'adaptive_1d')
            self.assertTrue((out/'performance_summary.csv').is_file())

    def test_cross_first_skips_bar_for_mixed_edge_signs(self):
        class Mixed(SyntheticBackend):
            def solve_isolated_mode(self,g,wavelength,**kwargs):
                mode,overlap=super().solve_isolated_mode(g,wavelength,**kwargs)
                if kwargs['guide']=='upper' and g.state=='cross':
                    mode.neff=1.81 if wavelength==self.config.device.wavelength_min_m else 1.79
                return mode,overlap
        b=Mixed(self.config)
        evaluator=WidthEvaluator(b,self.config,self.bank)
        evaluator.precompute_lower()
        sample=evaluator.evaluate_width(1e-6)
        self.assertFalse(sample['valid'])
        self.assertEqual(len([c for c in b.calls if c[0]=='upper']),2)
        self.assertFalse(any(c[1].state=='bar' for c in b.calls))

    def test_budget_failure_preserves_diagnostics_and_bounds(self):
        class Positive(SyntheticBackend):
            def solve_isolated_mode(self,*args,**kwargs):
                mode,overlap=super().solve_isolated_mode(*args,**kwargs)
                if kwargs['guide']=='upper':mode.neff+=.3
                return mode,overlap
        config=replace(self.config,endpoint_search=replace(self.config.endpoint_search,adaptive_initial_widths=3,adaptive_max_widths=7))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'no sampled sign-feasible input'):
                search_endpoints_adaptive(Positive(config),config,self.bank,output_dir=folder)
            report=json.loads((Path(folder)/'endpoint_adaptive_summary.json').read_text())
            self.assertLessEqual(report['widths_evaluated'],7)
            self.assertFalse(report['width_bounds_adjusted'])
            self.assertFalse((Path(folder)/'endpoint_adaptive_best.json').exists())
            self.assertTrue((Path(folder)/'endpoint_width_samples.csv').is_file())

    def test_fresh_sign_failure_is_not_a_success(self):
        class Changed(SyntheticBackend):
            def solve_isolated_mode(self,*args,**kwargs):
                mode,overlap=super().solve_isolated_mode(*args,**kwargs)
                if kwargs.get('use_cache') is False and kwargs['guide']=='upper':mode.neff+=.1
                return mode,overlap
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(EndpointSignError):
                search_endpoints_adaptive(Changed(self.config),self.config,self.bank,validation_only=True,output_dir=folder)
            report=json.loads((Path(folder)/'endpoint_adaptive_verification.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertFalse((Path(folder)/'endpoint_adaptive_best.json').exists())

    def test_dispatch_and_validation_cli_default_to_adaptive(self):
        self.assertEqual(self.config.endpoint_search.method,'adaptive_1d')
        with patch('vlac.adaptive_endpoint.search_endpoints_adaptive',return_value='adaptive') as f:
            self.assertEqual(search_endpoints(None,self.config,self.bank),'adaptive')
            f.assert_called_once()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json';p.write_text(json.dumps(self.config.as_dict()))
            with patch.object(cli,'LumapiBackend',SyntheticBackend),patch.object(cli,'build_material_bank',return_value=self.bank),contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
                out=cli.run_endpoint_only(p,Path(folder)/'endpoint')
                rc=cli.main(['validate-endpoint','--config',str(p),'--out',str(Path(folder)/'validation/report.json')])
            self.assertEqual(rc,0)
            result=json.loads((out/'endpoint_summary.json').read_text())
            self.assertEqual(result['search_method'],'adaptive_1d')
            for path in result['files'].values():self.assertTrue(Path(path).is_file())
            report=json.loads((Path(folder)/'validation/report.json').read_text())
            self.assertEqual(report['search_summary']['backend'],'adaptive_1d')
            self.assertTrue(report['cross_state_sign_verification']['passed'])


class CacheAndTemplateTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config/python_joint.json');self.bank=build_material_bank(self.config)

    def test_lower_raw_cache_ignores_absent_upper_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            b=backend_without_mode(self.config,self.bank,folder)
            with patch.object(b,'_mode_candidates',return_value=fixture_modes()) as solve:
                b._raw_mode_candidates(geometry(self.config,.6e-6,-3e-6),1.3e-6,include_upper=False,include_lower=True)
                b._raw_mode_candidates(geometry(self.config,1.1e-6,3e-6),1.3e-6,include_upper=False,include_lower=True)
                self.assertEqual(solve.call_count,1)
                b._raw_memory.clear()
                b._raw_mode_candidates(geometry(self.config,.6e-6,-3e-6),1.3e-6,include_upper=False,include_lower=True)
                self.assertEqual(solve.call_count,1)  # durable raw cache, not only the LRU
                b._raw_mode_candidates(geometry(self.config,state='bar'),1.3e-6,include_upper=False,include_lower=True)
                self.assertEqual(solve.call_count,2)
                b._raw_mode_candidates(geometry(self.config),1.3e-6,include_upper=False,include_lower=True,mesh_accuracy=4)
                self.assertEqual(solve.call_count,3)
                b.x_span_m*=1.1
                b._raw_mode_candidates(geometry(self.config),1.3e-6,include_upper=False,include_lower=True)
                self.assertEqual(solve.call_count,4)
                b._raw_mode_candidates(geometry(self.config),1.3e-6,include_upper=False,include_lower=True,use_cache=False)
                self.assertEqual(solve.call_count,5)

    def test_reference_changes_reselect_without_extra_eigensolves(self):
        b=backend_without_mode(self.config,self.bank)
        modes=fixture_modes();g=geometry(self.config)
        with patch.object(b,'_mode_candidates',return_value=modes) as solve:
            a=b.solve_bare_modes(g,1.3e-6)
            ref=BareModeReference(replace(modes[0],neff=1.91),replace(modes[1],neff=1.79))
            b.solve_bare_modes(g,1.3e-6,references=(ref,))
            self.assertEqual(solve.call_count,2)
            coupled_ref=ModeReference(ref.upper,ref.lower,(modes[0],modes[1]))
            c,_=b.evaluate_coefficients(g,1.3e-6,references=(coupled_ref,),previous_kappa=1j)
            count=solve.call_count
            b.evaluate_coefficients(g,1.3e-6,previous_kappa=-1j)
            self.assertEqual(solve.call_count,count)
            self.assertEqual(count,3)

    def test_templates_and_materials_reused_and_batch_matches_scalar(self):
        class Session:
            def __init__(self): self.scripts=[];self.transfers=0
            def eval(self,script):self.scripts.append(script)
            def putv(self,name,value):pass
            def getv(self,name):self.transfers+=1;return {'count':1,'cards':[self.card()]}
            def nummodes(self):return 1
            def getdata(self,card,key):self.transfers+=1;return self.card()[key]
            @staticmethod
            def card():
                return dict(neff=np.array([[1.9+.00001j]]),x=np.linspace(-13e-6,13e-6,4),
                            y=np.linspace(-3.5e-6,4.7e-6,4),Ex=np.ones((4,4)),Ey=np.zeros((4,4)))
        b=backend_without_mode(self.config,self.bank);b.session=Session()
        first=b._mode_candidates(geometry(self.config),1.26e-6,include_upper=True,include_lower=False)
        b._mode_candidates(geometry(self.config,state='bar'),1.34e-6,include_upper=False,include_lower=True)
        b._mode_candidates(geometry(self.config,1.1e-6,2e-6),1.26e-6,include_upper=True,include_lower=True)
        scripts=b.session.scripts
        self.assertEqual(scripts.count('newproject;'),1)
        self.assertEqual(sum(script.count('addfde;') for script in scripts),1)
        self.assertEqual(sum(script.count('addmaterial(') for script in scripts),4)
        self.assertEqual(scripts.count('switchtolayout;'),2)
        self.assertEqual(b.session.transfers,3)
        self.assertNotIn('set("name","FDE")','\n'.join(scripts))
        self.assertIn('setnamed("upper_ridge","enabled",0)','\n'.join(scripts))
        for category in ('material_setup','geometry_setup','eigensolve','field_transfer','field_processing'):
            self.assertGreaterEqual(b.performance_stats[category]['seconds'],0)
        config=replace(self.config,solver=replace(self.config.solver,batch_mode_data=False))
        scalar=backend_without_mode(config,self.bank);scalar.session=Session()
        second=scalar._mode_candidates(geometry(config),1.26e-6,include_upper=True,include_lower=False)
        np.testing.assert_allclose(first[0].ex,second[0].ex)
        self.assertEqual(first[0].neff,second[0].neff)
        self.assertEqual(scalar.session.transfers,5)
        scalar.config=replace(scalar.config,solver=replace(scalar.config.solver,reuse_mode_project=False))
        scalar._mode_candidates(geometry(config),1.34e-6,include_upper=True,include_lower=False)
        self.assertEqual(scalar.session.scripts.count('newproject;'),2)

    def test_progress_reports_setup_and_cache_timings(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stderr(io.StringIO()):
            reporter=ProgressReporter('fixture',folder)
            cache=SQLiteCache(Path(folder)/'cache.sqlite',progress=reporter)
            cache.put('a',[1]);cache.get('a')
            reporter.record_timing('geometry_setup',.01)
            self.assertIn('cache_lookup',reporter.runtime_stats['timings'])
            self.assertEqual(reporter.runtime_stats['timings']['geometry_setup']['calls'],1)
            reporter.finish()
            self.assertIn('geometry_setup',json.loads((Path(folder)/'run_status.json').read_text())['timings'])


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config/python_joint.json');self.bank=build_material_bank(self.config)
        self.endpoint=EndpointSearchResult(1e-6,175e-9,.6e-6,1.2e-6,100.,np.array([1.26e-6,1.34e-6]),[],[],False,[],{})

    @staticmethod
    def minimize_once(fun,x0,**kwargs):
        return SimpleNamespace(x=x0,success=True,message='synthetic optimizer',fun=fun(x0))

    def coefficient_function(self,axis='none',logs=None):
        def evaluate(backend,config,bank,endpoint,w,dc,z,wavelengths,**kwargs):
            if logs is not None:logs.append((len(z),len(wavelengths)))
            result=[]
            for wavelength in wavelengths:
                row=[]
                for position in z:
                    value=(np.cos(3*np.pi*position/config.device.length_m) if axis=='z' else
                           np.cos(3*np.pi*(wavelength-config.device.wavelength_min_m)/(config.device.wavelength_max_m-config.device.wavelength_min_m)) if axis=='wavelength' else 0.)
                    row.append(SimpleNamespace(delta_beta=100*complex(value),kappa=1.+0j,extraction_flag='ok'))
                result.append(row)
            return result
        return evaluate

    def test_normalized_powell_recovers_physical_controls_and_stops(self):
        target_w=.72e-6;target_d=1.0e-6
        def evaluate(backend,config,bank,endpoint,w,dc,z,wavelengths,**kwargs):
            value=np.sum(((w[1:-1]-target_w)/.9e-6)**2)+np.sum(((dc[1:-1]-target_d)/16e-6)**2)
            return [[SimpleNamespace(delta_beta=complex(value),kappa=0j,extraction_flag='ok') for _ in z] for _ in wavelengths]
        def metric(coefficients,*args):
            nz=len(coefficients[0]);return coefficients[0][0].delta_beta.real,np.zeros(nz),np.zeros(nz)
        config=replace(self.config,optimizer=replace(self.config.optimizer,max_function_evaluations_per_grid=300,max_grid_refinements=3))
        with patch('vlac.optimize.evaluate_design_profiles',side_effect=evaluate),patch('vlac.optimize.minimax_objective',side_effect=metric):
            result=optimize_profiles(None,config,self.bank,self.endpoint)
        np.testing.assert_allclose(result.w1_control_m[1:-1],target_w,atol=1e-9)
        np.testing.assert_allclose(result.d_c2c_control_m[1:-1],target_d,atol=1e-9)
        self.assertEqual(result.w1_control_m[0],self.endpoint.w1_input_m)
        self.assertEqual(result.w1_control_m[-1],self.endpoint.w1_output_m)
        self.assertEqual(len(result.optimizer_history),1)
        self.assertTrue(result.grid_converged)
        self.assertEqual(result.optimizer_history[0]['coordinate_system'],'unit_interval')

    def test_only_failing_z_axis_refined_at_band_edges(self):
        config=replace(self.config,optimizer=replace(self.config.optimizer,max_grid_refinements=1))
        logs=[]
        with patch('vlac.optimize.minimize',side_effect=self.minimize_once),patch('vlac.optimize.evaluate_design_profiles',side_effect=self.coefficient_function('z',logs)):
            result=optimize_profiles(None,config,self.bank,self.endpoint)
        self.assertEqual(result.grid_convergence_history[0]['requested_refinement_axes'],['z'])
        self.assertEqual(len(result.z_m),9)
        np.testing.assert_allclose(result.wavelengths_m,self.endpoint.wavelengths_m,rtol=0,atol=0)
        self.assertTrue(all(count==2 for _,count in logs))
        self.assertIsNone(result.grid_convergence_history[0]['wavelength_coefficient_interpolation_error'])
        self.assertEqual(result.grid_convergence_history[0]['wavelength_check'],'band_edges_only_no_interior_check')

    def test_interior_wavelengths_are_not_simulated_even_when_they_would_fail(self):
        seen=[]
        def evaluate(backend,config,bank,endpoint,w,dc,z,wavelengths,**kwargs):
            seen.extend(wavelengths)
            return self.coefficient_function('wavelength')(backend,config,bank,endpoint,w,dc,z,wavelengths,**kwargs)
        with patch('vlac.optimize.minimize',side_effect=self.minimize_once),patch('vlac.optimize.evaluate_design_profiles',side_effect=evaluate):
            result=optimize_profiles(None,self.config,self.bank,self.endpoint)
        self.assertEqual(set(seen),set(self.endpoint.wavelengths_m))
        self.assertTrue(result.grid_converged)

    def test_axis_limit_does_not_claim_unchecked_convergence(self):
        config=replace(self.config,device=replace(self.config.device,z_samples_max=5,wavelength_samples_max=5))
        with patch('vlac.optimize.minimize',side_effect=self.minimize_once),patch('vlac.optimize.evaluate_design_profiles',side_effect=self.coefficient_function()):
            result=optimize_profiles(None,config,self.bank,self.endpoint)
        self.assertFalse(result.grid_converged)
        self.assertEqual(result.grid_convergence_history[-1]['stop_reason'],'refinement_budget_or_axis_limit')


if __name__=='__main__':unittest.main()
