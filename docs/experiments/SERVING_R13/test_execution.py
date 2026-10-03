#!/usr/bin/env python3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest,warnings
from tilemega.serving.execution import *
class ExecutionTests(unittest.TestCase):
    def test_sidecar_does_not_mutate_manifest(self):
        with TemporaryDirectory() as tmp:
            so=Path(tmp)/'plan.so';manifest=Path(str(so)+'.plan.json');manifest.write_text('original')
            write_execution(so,'L1',1,'L2',samples=[1,2,3])
            self.assertEqual(resolve_execution(so,'auto','auto','auto',False),('L1',True,'L2'))
            self.assertEqual(manifest.read_text(),'original')
    def test_explicit_overrides(self):
        with TemporaryDirectory() as tmp:
            so=Path(tmp)/'plan.so';write_execution(so,'L1',1,'L1')
            self.assertEqual(resolve_execution(so,'L2',0,'L2',True),('L2',False,'L2'))
    def test_missing_sidecar_warns(self):
        with TemporaryDirectory() as tmp,warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always')
            self.assertEqual(resolve_execution(Path(tmp)/'plan.so','auto','auto','auto',True),('L2',True,'L2'))
            self.assertEqual(len(log),1)
    def test_cli_only_keys_filtered(self):
        f=dict(pg='pages',decode_executor='measure',decode_loop=1,prefill_executor='L1',watchdog=0)
        self.assertEqual(compiler_features(f),dict(pg='pages',watchdog=0))
if __name__=='__main__':unittest.main()
