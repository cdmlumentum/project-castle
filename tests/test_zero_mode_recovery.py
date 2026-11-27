"""Zero-mode recovery and transfer diagnosis using recording MODE API stubs."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import contextlib
from contextlib import closing
import io
import json
import sqlite3
import tempfile
import unittest

import numpy as np

from test_efficiency import ROOT, backend_without_mode, geometry, fixture_modes
from vlac.config import load_config
from vlac.materials import build_material_bank
from vlac.modes import InvalidModeError
from vlac.progress import ProgressReporter


class Session:
    def __init__(self,count=1,zero_fields=False,batch_zero=False):
        self.scripts=[];self.count=count;self.zero_fields=zero_fields;self.batch_zero=batch_zero
        self.direct_fields=0
    def eval(self,script):self.scripts.append(script)
    def putv(self,*args):pass
    def nummodes(self):return self.count
    def card(self):
        return {'neff':np.array([[1.75+0j]]),'x':np.linspace(-13e-6,13e-6,4),
                'y':np.linspace(-3.5e-6,4.7e-6,4),'Ex':np.zeros((4,4)) if self.zero_fields else np.ones((4,4)),
                'Ey':np.zeros((4,4))}
    def getv(self,name):
        return {'count':0,'cards':[]} if self.batch_zero else {'count':self.count,'cards':[self.card() for _ in range(self.count)]}
    def getdata(self,card,key):self.direct_fields+=1;return self.card()[key]


class ZeroModeTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config/python_joint.json')
        self.bank=build_material_bank(self.config)

    def raw(self,backend,**kwargs):
        return backend._raw_mode_candidates(geometry(self.config),1.26e-6,include_upper=False,include_lower=True,**kwargs)

    def test_empty_then_success_rebuilds_once_and_only_success_is_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            b=backend_without_mode(self.config,self.bank,folder)
            b._project_ready=True
            def solve(*args,**kwargs):
                if not hasattr(b,'fixture_attempt'):b.fixture_attempt=1;return []
                self.assertFalse(b._project_ready)
                return fixture_modes()
            with patch.object(b,'_mode_candidates',side_effect=solve) as call:
                self.assertTrue(self.raw(b))
                self.assertEqual(call.call_count,2)
                self.assertEqual(b.cache.count(),1)
                self.assertTrue(self.raw(b))
                self.assertEqual(call.call_count,2)
            self.assertEqual(b.last_mode_issue['status'],'recovered_after_zero_modes')

    def test_persistent_empty_is_never_cached_and_next_call_retries_again(self):
        with tempfile.TemporaryDirectory() as folder:
            b=backend_without_mode(self.config,self.bank,folder)
            with patch.object(b,'_mode_candidates',return_value=[]) as call:
                for _ in range(2):
                    with self.assertRaisesRegex(InvalidModeError,'No empty result was cached'):
                        self.raw(b)
                self.assertEqual(call.call_count,4)
                self.assertEqual(b.cache.count(),0)
                self.assertFalse(b._raw_memory)

    def test_existing_empty_cache_entry_is_ignored_and_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            b=backend_without_mode(self.config,self.bank,folder)
            with patch.object(b,'_mode_candidates',return_value=fixture_modes()) as call:
                self.raw(b)
                with closing(sqlite3.connect(b.cache.path)) as db:
                    key=db.execute('SELECT cache_key FROM cache_v2').fetchone()[0]
                b.cache.put(key,[]);b._raw_memory.clear()
                self.assertTrue(self.raw(b))
                self.assertEqual(call.call_count,2)
                self.assertTrue(b.cache.get(key))

    def test_fresh_verification_still_bypasses_both_cache_tiers_during_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            b=backend_without_mode(self.config,self.bank,folder)
            with patch.object(b,'_mode_candidates',side_effect=[[],fixture_modes()]) as call:
                self.assertTrue(self.raw(b,use_cache=False))
                self.assertEqual(call.call_count,2)
                self.assertEqual(b.cache.count(),0)
                self.assertFalse(b._raw_memory)

    def test_orientation_and_search_are_reset_on_preflight_and_every_solve(self):
        b=backend_without_mode(self.config,self.bank);b.session=Session()
        b.preflight(1.3e-6)
        b._mode_candidates(geometry(self.config),1.26e-6,include_upper=False,include_lower=True)
        b._mode_candidates(geometry(self.config,state='bar'),1.34e-6,include_upper=True,include_lower=False)
        combined='\n'.join(b.session.scripts)
        self.assertEqual(combined.count('set("solver type",3)'),3)
        self.assertEqual(combined.count('setanalysis("search","near n")'),2)
        self.assertEqual(combined.count('setanalysis("use max index",1)'),2)
        self.assertNotIn('set("name","FDE")',combined)

    def test_batch_zero_is_confirmed_and_direct_transfer_recovers_without_another_solve(self):
        b=backend_without_mode(self.config,self.bank);b.session=Session(batch_zero=True)
        result=self.raw(b)
        self.assertTrue(result)
        self.assertEqual(b.session.scripts.count('findmodes;'),1)
        self.assertEqual(b.session.direct_fields,5)
        self.assertEqual(b._last_mode_diagnostics['mode_list_count'],1)

    def test_mode_list_with_unusable_fields_is_a_processing_failure_not_zero_modes(self):
        b=backend_without_mode(self.config,self.bank);b.session=Session(zero_fields=True)
        with self.assertRaisesRegex(RuntimeError,'MODE found eigenmodes'):
            self.raw(b)
        self.assertEqual(b.session.scripts.count('findmodes;'),1)
        self.assertEqual(b.last_mode_issue['status'],'all_transferred_fields_unusable')
        self.assertEqual(b.last_mode_issue['mode_data']['mode_list_count'],1)
        self.assertEqual(b.last_mode_issue['mode_data']['discarded_field_details'][0]['native_ex_peak'],0.)

    def test_api_failures_are_not_retried_or_hidden_as_physical_rejections(self):
        b=backend_without_mode(self.config,self.bank)
        with patch.object(b,'_mode_candidates',side_effect=RuntimeError('API license failure')) as call:
            with self.assertRaisesRegex(RuntimeError,'license'):
                self.raw(b)
            self.assertEqual(call.call_count,1)

    def test_failure_diagnostics_include_actual_compact_lower_geometry_and_can_disable_retry(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stderr(io.StringIO()):
            b=backend_without_mode(self.config,self.bank)
            b.progress=ProgressReporter('fixture',folder)
            with patch.object(b,'_mode_candidates',return_value=[]):
                with self.assertRaisesRegex(InvalidModeError,'1260 nm'):
                    b.solve_isolated_mode(geometry(self.config),1.26e-6,guide='lower')
            report=json.loads((Path(folder)/'fde_mode_diagnostics.json').read_text())
            self.assertEqual(len(report['attempts']),2)
            entry=report['attempts'][-1]
            self.assertAlmostEqual(entry['domain_m'][0],10e-6)
            self.assertEqual(entry['mesh_cells'],[203,208])
            self.assertEqual(entry['geometry']['w2_m'],2e-6)
            names={r['name'] for r in entry['active_rectangles']}
            self.assertIn('lower_ridge',names);self.assertIn('lower_slab',names)
            self.assertNotIn('upper_ridge',names);self.assertNotIn('upper_slab',names)
            b.progress.finish()
        cfg=replace(self.config,solver=replace(self.config.solver,zero_mode_retries=0))
        b=backend_without_mode(cfg,self.bank)
        with patch.object(b,'_mode_candidates',return_value=[]) as call:
            with self.assertRaises(InvalidModeError):self.raw(b)
            self.assertEqual(call.call_count,1)
        for bad in (-1,3,True):
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'zero_mode_retries'):
                replace(self.config,solver=replace(self.config.solver,zero_mode_retries=bad)).validate()


if __name__=='__main__':unittest.main()
