"""Completed recovery artifacts can be reused without accepting dirty timings."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('reuse',ROOT/'scripts/dnn1_tvm_reuse_driver.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Reuse(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.work=self.root/'runs/dnn1/recovery'
        self.entry=self.work/'bf16/mbv1_b4_bf16/key';self.entry.mkdir(parents=True)
        self.vm=self.entry/'model.so';self.vm.write_bytes(b'compiled model')
        self.receipt=self.entry/'compiled.json'
        self.receipt.write_text(json.dumps(dict(identity=dict(recipe=dict(schema=6),batch=4,precision='bf16'),model_sha256=module.sha(self.vm))))
        self.session=self.root/'session.json';self.session.write_text(json.dumps(dict(model='mbv1',batch=4,status='complete',ended=10)))
        self.value=dict(work_dir=str(self.work),source_session=str(self.session),source_session_sha256=module.sha(self.session),compiled_receipts={str(self.receipt):module.sha(self.receipt)})
        self.context=patch.object(module._base,'ROOT',self.root);self.context.start();self.addCleanup(self.context.stop)

    def test_completed_artifacts_reuse_does_not_modify_source(self):
        before=self.receipt.read_bytes();self.assertEqual(module.validate_reuse('mbv1_b4',self.value),str(self.work))
        self.assertEqual(self.receipt.read_bytes(),before)

    def test_missing_or_mutated_vm_and_session_rejected(self):
        self.vm.write_bytes(b'changed')
        with self.assertRaises(ValueError):module.validate_reuse('mbv1_b4',self.value)
        self.vm.write_bytes(b'compiled model');self.session.write_text('{}')
        with self.assertRaises(ValueError):module.validate_reuse('mbv1_b4',self.value)

    def test_unfinished_or_different_cell_cannot_reuse(self):
        with self.assertRaises(ValueError):module.validate_reuse('mbv1_b8',self.value)
        self.session.write_text(json.dumps(dict(model='mbv1',batch=4,status='running')))
        self.value['source_session_sha256']=module.sha(self.session)
        with self.assertRaises(ValueError):module.validate_reuse('mbv1_b4',self.value)

if __name__=='__main__':unittest.main()
