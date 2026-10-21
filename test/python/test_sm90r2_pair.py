"""Do not accept stale checkpoints, shared GPU processes, or partial pairs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run_sm90r2_pair import validate_pair,measurement
from collect_sm90r2_guarded_pairs import main as collect


class PairContractTests(unittest.TestCase):
    def setUp(self):
        self.before=dict(pid=100,checkpoint=dict(seed=0,checkpoint_sha256='a'*64),batch=1,
            prompt_ids_sha256='b'*64,prompt_length=64,max_tokens=1024,round_index=0,
            physical_device_identity=dict(cuda_uuid='card0'))
        self.after=copy.deepcopy(self.before);self.after['pid']=101

    def test_distinct_gpu_processes_share_exact_inputs_and_card(self):
        validate_pair(self.before,self.after)
        for key,bad in [('pid',100),('checkpoint',dict(seed=1)),('batch',16),
                        ('round_index',1),('physical_device_identity',dict(cuda_uuid='card7'))]:
            value=copy.deepcopy(self.after);value[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_pair(self.before,value)

    def test_incomplete_measurement_cannot_enter_a_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);m=p/'job/measurements/B1';m.mkdir(parents=True)
            (m/'measurements.json').write_text(json.dumps(dict(max_tokens=1024,generation_runs=[{}]*7)))
            with self.assertRaises(ValueError):measurement(p)

    def test_outer_guard_rejection_invalidates_both_finished_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);attempt=p/'attempt';(attempt/'pair').mkdir(parents=True)
            (attempt/'guard_result.json').write_text(json.dumps(dict(code=75)))
            (attempt/'pair/paired.json').write_text(json.dumps(dict(status='completed')))
            state=p/'state.json';state.write_text(json.dumps(dict(job=dict(status='done',out=str(attempt)))))
            with patch.object(sys,'argv',['collect','--states',str(state),'--out',str(p/'result.csv')]):
                with self.assertRaisesRegex(ValueError,'accepted full-pair'):collect()
            self.assertFalse((p/'result.csv').exists())


if __name__=='__main__':unittest.main()
