import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tilemega.serving.validation_guard import validation_guard

class ValidationGuard(unittest.TestCase):
    @patch('tilemega.serving.validation_guard._exclusive',return_value=False)
    def test_default_still_requires_exclusive_gpu(self,guard):
        self.assertFalse(validation_guard(Path('/unused'),'before',True))
        guard.assert_called_once_with(Path('/unused'),'before',True)

    @patch('tilemega.serving.validation_guard._exclusive')
    @patch('tilemega.serving.validation_guard._gpu_owners',return_value=({1,2},2048,4096))
    def test_explicit_correctness_permission_never_polls(self,owners,guard):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'guard.jsonl'
            self.assertTrue(validation_guard(out,'before',True,allow_shared=True))
            record=json.loads(out.read_text())
            self.assertFalse(record['timing_eligible'])
            self.assertEqual(record['pids'],[1,2])
            self.assertEqual(record['guard_mode'],'shared_correctness')
        guard.assert_not_called();owners.assert_called_once()

if __name__=='__main__':unittest.main()
