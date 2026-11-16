"""Retained full P5 fixtures require byte proof before original publication."""
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_tvm_resume_finalize as finalizer
import dnn1_receipt_integrity as integrity
import dnn1_tvm_resume_test as fixtures


class FinalizeTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CollectorTest(methodName='test_complete_clean_cell_aggregates_every_process')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        identity_path = f.job / 'physical-device-identity.json'
        identity = dict(gpu_index=5, cuda_uuid='fixture-uuid', nvml_uuid='GPU-fixture-uuid',
            cuda_pci_bus_id='0000:87:00.0', numa_node=1,
            outer_guard_holds_percard_lock=True, probe_exit_code=0,
            wrapper_sha256=integrity.sha(ROOT/'scripts/run_r16_with_device_identity.py'),
            guard_policy_sha256=integrity.sha(integrity.base.POLICY))
        identity_path.write_text(json.dumps(identity, indent=2)+'\n')
        for job in f.session['logical_jobs']:
            process = job['final']; path = Path(process['measurement'])
            value = json.loads(path.read_text())
            value['configuration']['tvm_diag']['cache_identity']['gpu_uuid'] = 'fixture-uuid'
            path.write_text(json.dumps(value, indent=2)+'\n')
            process['receipt_sha256'] = integrity.sha(path)
            value.update(physical_device_identity=identity, physical_device_identity_file=str(identity_path),
                         physical_device_identity_sha256=integrity.sha(identity_path))
            path.write_text(json.dumps(value, indent=2)+'\n')
        finalizer.base.write(f.job/'cell-result.json', f.session)
        finalizer.base.write(f.attempt/'guard_result.json', {'code':0})
        (f.attempt/'guard.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in f.outer))

    def collect(self):
        f = self.fixture
        with patch.object(finalizer.original.publisher, 'publish', side_effect=lambda a,b: shutil.copyfile(a,b)), \
             patch.object(finalizer.original.auditor, 'audit', return_value={'cells':[{'sampled_windows_clean':True}]}):
            return finalizer.collect(f.cell, f.attempt, f.root/'finalized')

    def test_exact_annotations_pass_original_full_p5_and_keep_raw_files(self):
        session = self.fixture.job/'cell-result.json'; before = integrity.sha(session)
        row = self.collect()
        self.assertEqual(row['status'],'pass')
        self.assertEqual(float(row['median_ms']),2.)
        self.assertEqual(integrity.sha(session),before)
        windows = finalizer.base.read(self.fixture.root/'finalized/accepted.json')['windows']
        self.assertEqual(len(windows),5)
        self.assertTrue(all(w['receipt_integrity']['mode']=='exact official annotation only' for w in windows))

    def test_changed_measurement_cannot_publish_any_row(self):
        process = self.fixture.session['logical_jobs'][0]['final']; path = Path(process['measurement'])
        value = json.loads(path.read_text()); value['median_ms']=.1
        path.write_text(json.dumps(value,indent=2)+'\n')
        with self.assertRaisesRegex(ValueError,'original byte SHA'):self.collect()
        self.assertFalse((self.fixture.root/'finalized/raw.csv').exists())


if __name__ == '__main__':
    unittest.main()
