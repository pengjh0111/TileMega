import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90r6_ablate import ARMS,sha,validate
from sm90r6_ablation_gates import validate_subset


class AblationGates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.arms={}
        for label in ARMS:
            binary=self.root/(label+'.so');binary.write_bytes(label.encode())
            plan=dict(native_dense={'theta':1},export_sha256='export',grid=132,runtime_batch=1)
            if label=='AB4c':plan['execution']='kernel_sequence'
            self.write(str(binary)+'.identity.json',dict(binary_sha256=sha(binary),plan=plan,shared_memory_bytes=123456))
            c2=self.root/(label+'-c2.json')
            self.write(c2,dict(passed=True,processes=[{}]*5,steps=1024,model='model',batch=1,
                artifacts=dict(candidate={'sha256':sha(binary)},reference={'sha256':sha(binary)})))
            self.arms[label]=dict(binary=str(binary),c2=str(c2))
            if label in ('AB4b','AB4c'):
                c1=self.root/(label+'-c1.json')
                self.write(c1,dict(pass_=True,**{'pass':True},c1_pass=True,spec=dict(model='model',batch=1,
                    correctness={'c1':256},arms={'winner':{'decode_sha256':sha(binary)}})))
                self.arms[label]['c1']=str(c1)
        self.change(self.arms['AB4c']['c2'],lambda d:d['artifacts']['reference'].update(
            sha256=sha(self.arms['AB4b']['binary'])))

    def write(self,path,value):Path(path).write_text(json.dumps(value))

    def change(self,path,edit):
        value=json.loads(Path(path).read_text());edit(value);self.write(path,value)

    def test_complete_artifact_bound_gates(self):
        self.assertEqual(tuple(validate(self.arms,'model',1)),ARMS)

    def test_partial_domain_keeps_all_remaining_gates(self):
        arms=copy.deepcopy(self.arms);del arms['AB1a']
        self.assertEqual(len(validate_subset(arms,'model',1,('AB1a',))),7)
        with self.assertRaises(ValueError):validate_subset(arms,'model',1)
        self.change(arms['AB4c']['c1'],lambda d:d['spec']['correctness'].update(c1=255))
        with self.assertRaises(ValueError):validate_subset(arms,'model',1,('AB1a',))

    def test_partial_domain_cannot_omit_f_or_two_arms(self):
        for excluded in [('F',),('AB1a','AB1b')]:
            arms={k:v for k,v in self.arms.items() if k not in excluded}
            with self.assertRaises(ValueError):validate_subset(arms,'model',1,excluded)

    def test_missing_f_is_rejected(self):
        arms=copy.deepcopy(self.arms);del arms['F']
        with self.assertRaises(ValueError):validate(arms,'model',1)

    def test_incomplete_c2_is_rejected(self):
        self.change(self.arms['AB2']['c2'],lambda d:d.update(processes=[{}]*4))
        with self.assertRaises(ValueError):validate(self.arms,'model',1)

    def test_c1_must_cover_256_positions(self):
        self.change(self.arms['AB4c']['c1'],lambda d:d['spec']['correctness'].update(c1=255))
        with self.assertRaises(ValueError):validate(self.arms,'model',1)

    def test_sequence_cannot_retile(self):
        self.change(self.arms['AB4c']['binary']+'.identity.json',lambda d:d['plan']['native_dense'].update(theta=2))
        with self.assertRaises(ValueError):validate(self.arms,'model',1)

    def test_c2_from_another_binary_is_rejected(self):
        self.change(self.arms['AB1a']['c2'],lambda d:d['artifacts']['candidate'].update(sha256='another'))
        with self.assertRaises(ValueError):validate(self.arms,'model',1)


if __name__=='__main__':unittest.main()
