# SPDX-License-Identifier: BSD-3-Clause
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from validate_r16_numeric_gate import validate_payload
from r16_measurement_validation import file_sha

class NumericGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)/'measurements';self.path=self.directory/'B1/measurements.json'
        self.path.parent.mkdir(parents=True)
        prompt=Path(self.tmp.name)/'prompt.json';prompt.write_text(json.dumps([[1]*64]))
        self.c=dict(batch=1,gpu=7,config_sha256='config',prompt_ids=str(prompt),
                    script_sha256='driver',system='TileMega',mode='L2',loop=0,moe=False)
        self.pair=dict(prefill=dict(artifact_id='prefill'),decode=dict(artifact_id='decode',plan=dict(pg='off')))
        self.populate(16)
    def write(self,path,value):Path(path).write_text(json.dumps(value))
    def populate(self,count):
        self.value=dict(system='TileMega',evidence='verified',valid=True,batch=1,max_tokens=count,prompt_length=64,
                        gpu_index=7,model_config_sha256='config',prompt_ids_sha256=file_sha(self.c['prompt_ids']),pid=100,
                        configuration=dict(identities=copy.deepcopy(self.pair)),generation_runs=[])
        for n in (1,count):
            for rnd in range(4):
                name=f'tokens_N{n}_run{rnd}.json';self.write(self.path.parent/name,[[2]*n])
                self.value['generation_runs'].append(dict(N=n,run=rnd,warmup=rnd==0,tokens_file=name))
        self.write(self.path,self.value)
        self.write(self.directory/'candidate-evidence.json',dict(artifact_ids={p:i['artifact_id'] for p,i in self.pair.items()}))
        self.write(self.directory/'c1.json',dict(pass_=True,positions=count))
        c1=json.loads((self.directory/'c1.json').read_text());c1['pass']=c1.pop('pass_');self.write(self.directory/'c1.json',c1)
        checks=[dict(mode=m,decode_loop=l,**{'pass':True}) for m,l in [('L1',0),('L2',0),('L1',1)]]
        self.write(self.directory/'c2.json',dict(same_binary=True,checks=checks,**{'pass':True}))
        calls=[dict(tokens=n,executor='L2',requested_device_loop=False,device_loop_used=False) for n in (1,count) for _ in range(4)]
        calls += [dict(tokens=count,executor=r['mode'],requested_device_loop=bool(r['decode_loop']),device_loop_used=bool(r['decode_loop'])) for r in checks]
        self.write(self.directory/'actual-execution.json',dict(driver_sha256='driver',calls=calls))
    def validate(self,stage='smoke',uuid=None):return validate_payload(self.path,self.c,self.pair,stage,uuid)
    def test_short_gate_accepts_complete_consistent_pair(self):self.assertTrue(self.validate()['passed'])
    def test_earlier_bad_repeat_rejects_even_when_last_repeat_and_c2_pass(self):
        self.write(self.path.parent/'tokens_N16_run1.json',[[3]*16])
        with self.assertRaisesRegex(ValueError,'repetitions'):self.validate()
    def test_numeric_failures_and_wrong_artifact_are_rejected(self):
        for filename,key,value in [('c1.json','pass',False),('c1.json','positions',15),('c2.json','same_binary',False),('candidate-evidence.json','artifact_ids',{'decode':'other'})]:
            path=self.directory/filename;original=path.read_text();row=json.loads(original);row[key]=value;self.write(path,row)
            with self.subTest(filename=filename,key=key),self.assertRaises(ValueError):self.validate()
            path.write_text(original)
    def test_wrong_actual_executor_and_native_label_are_rejected(self):
        actual=self.directory/'actual-execution.json';original=actual.read_text();row=json.loads(original);row['calls'][0]['executor']='L1';self.write(actual,row)
        with self.assertRaises(ValueError):self.validate()
        actual.write_text(original);self.value['system']='vllm';self.write(self.path,self.value)
        with self.assertRaises(ValueError):self.validate()
    def test_long_moe_requires_256_step_teacher_forced_prefix(self):
        self.populate(1024);self.c['moe']=True
        with self.assertRaises(ValueError):self.validate('longcheck')
        self.write(self.directory/'c1.json',{'pass':True,'positions':256});self.assertEqual(self.validate('longcheck')['c1_positions'],256)
    def test_physical_proof_cannot_be_missing_or_relabelled(self):
        with self.assertRaises(KeyError):self.validate(uuid='GPU-expected')
        identity=dict(cuda_uuid='expected',nvml_uuid='GPU-expected',gpu_index=7);receipt=self.directory/'identity.json';self.write(receipt,identity)
        self.value.update(physical_device_identity=identity,physical_device_identity_file=str(receipt),physical_device_identity_sha256=file_sha(receipt));self.write(self.path,self.value)
        self.assertTrue(self.validate(uuid='GPU-expected')['passed'])
        with self.assertRaises(ValueError):self.validate(uuid='GPU-another')
if __name__=='__main__':unittest.main()
