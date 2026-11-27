"""Offline supervisor tests; fixture configuration only, no optical solves."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]


class BudgetedBlock2Tests(unittest.TestCase):
    def test_trial_budget_does_not_inherit_block1_cap(self):
        for extra,expected in (([],12000),(['--max-solves','1500'],1500)):
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as directory:
                base=Path(directory);endpoint=base/'endpoint';endpoint.mkdir()
                raw=json.loads((ROOT/'config/native.json').read_text(encoding='utf-8'))
                raw['execution']['max_solves']=200
                (endpoint/'configuration.json').write_text(json.dumps(raw),encoding='utf-8')
                out=base/'trial'
                module=runpy.run_path(str(ROOT/'tools/run_block2_budgeted.py'))
                main=module['main'];namespace=main.__globals__
                namespace['load_block']=Mock()
                child=SimpleNamespace(pid=123,poll=lambda:3,returncode=3)
                argv=['supervise','--endpoint-dir',str(endpoint),'--out',str(out),'--hours','8',*extra]
                fake_ctypes=SimpleNamespace(windll=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=Mock(return_value=1))))
                namespace['ctypes']=fake_ctypes
                with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                    with patch.object(namespace['subprocess'],'check_output',return_value='SYNTHETIC-source-id'),patch.object(namespace['subprocess'],'Popen',return_value=child):
                        self.assertEqual(main(),3)
                config=json.loads((out/'run_configuration.json').read_text())
                record=json.loads((out/'execution.json').read_text())
                self.assertEqual(config['execution']['max_solves'],expected)
                self.assertEqual(config['execution']['max_seconds'],28800)
                self.assertEqual(config['optimizer']['complete_within_seconds'],27000)
                self.assertEqual(record['maximum_fde_solves'],expected)
                self.assertEqual((datetime.fromisoformat(record['deadline_utc'])-datetime.fromisoformat(record['started_utc'])).total_seconds(),28800)
                self.assertEqual(config['solver']['isolated_reference'],'retained_slabs_v1')
                self.assertEqual(config['optimizer']['method'],'sweep_anchored_v1')
                for key in ('geometry','device'):
                    self.assertEqual(config[key],raw[key])
                for key in ('mesh_cells_x','mesh_cells_y','tracking_method','endpoint_domain_policy'):
                    self.assertEqual(config['solver'][key],raw['solver'][key])

    def test_wall_deadline_terminates_only_the_owned_child_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);endpoint=base/'endpoint';endpoint.mkdir()
            (endpoint/'configuration.json').write_bytes((ROOT/'config/native.json').read_bytes())
            out=base/'trial'
            namespace=runpy.run_path(str(ROOT/'tools/run_block2_budgeted.py'))['main'].__globals__
            namespace['load_block']=Mock()
            namespace['os']=SimpleNamespace(name='nt',environ={},getpid=lambda:123)
            namespace['ctypes']=SimpleNamespace(windll=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=Mock(return_value=1))))
            namespace['time']=SimpleNamespace(monotonic=Mock(side_effect=[0,28800,28801]))
            child=SimpleNamespace(pid=456,poll=lambda:None,returncode=1,wait=Mock())
            with patch.object(sys,'argv',['supervise','--endpoint-dir',str(endpoint),'--out',str(out)]),contextlib.redirect_stdout(io.StringIO()):
                with patch.object(namespace['subprocess'],'check_output',return_value='SYNTHETIC-source-id'),patch.object(namespace['subprocess'],'Popen',return_value=child),patch.object(namespace['subprocess'],'run') as terminate:
                    self.assertEqual(namespace['main'](),1)
            terminate.assert_called_once_with(['taskkill','/PID','456','/T','/F'],capture_output=True)
            child.wait.assert_called_once()
            self.assertEqual(json.loads((out/'execution.json').read_text())['status'],'emergency_timeout')

    def test_reject_invalid_budgets_before_creating_output(self):
        module=runpy.run_path(str(ROOT/'tools/run_block2_budgeted.py'))
        for budget in (['--hours','9'],['--hours','0'],['--max-solves','0']):
            with self.subTest(budget=budget),tempfile.TemporaryDirectory() as directory:
                out=Path(directory)/'trial'
                with patch.object(sys,'argv',['supervise','--endpoint-dir','unused','--out',str(out),*budget]),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):module['main']()
                self.assertFalse(out.exists())
