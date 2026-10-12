"""CPU contracts for request-shaped exports, caches and prompt preparation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tilemega.cache import export_key
from tilemega.cli import DEFAULTS, expand_workloads, read_config
from tilemega.serving.prepare_prompts import tokenize_passage
from tilemega.serving.workload import validate_prompt_ids


class WorkloadCLI(unittest.TestCase):
    def test_sweep_outputs_are_isolated_and_input_is_not_mutated(self):
        config=copy.deepcopy(DEFAULTS)
        config['workload'].update(points=[[64,1024],[128,2048]],capacity=4096)
        config['output']['dir']='runs/sweep'
        original=copy.deepcopy(config)
        first,second=expand_workloads(config)
        self.assertEqual(config,original)
        self.assertEqual(first['output']['dir'],'runs/sweep/ISL64_OSL1024')
        self.assertEqual(second['output']['dir'],'runs/sweep/ISL128_OSL2048')
        self.assertEqual(second['workload']['capacity'],4096)
        self.assertEqual(second['workload']['prompt_len'],128)
        self.assertNotIn('points',second['workload'])
        self.assertEqual(expand_workloads(DEFAULTS)[0]['output'],DEFAULTS['output'])

    def test_read_config_accepts_long_request_and_rejects_short_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'config.json'
            config=dict(model={'path':temporary},workload=dict(prompt_len=128,max_new_tokens=2048))
            path.write_text(json.dumps(config))
            self.assertEqual(read_config(path)['workload']['prompt_len'],128)
            config['workload']['capacity']=1088
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'capacity'):read_config(path)

    def test_decode_cache_cannot_alias_different_isl_at_same_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'input';path.write_text('stable')
            keys={export_key(path,path,'decode',1,capacity,prompt_len=isl)
                  for isl,capacity in ((64,4096),(128,4096),(128,2176))}
            self.assertEqual(len(keys),3)

    def test_rectangular_prompts_match_workload(self):
        self.assertEqual(validate_prompt_ids([[1]*128]*16,16,128),128)
        for ids,batch,isl in (([],1,64),([[1]*64],2,64),([[1]*64],1,128),
                              ([[True]],1,1),([[1],[1,2]],1,None)):
            with self.assertRaises(ValueError):validate_prompt_ids(ids,batch,isl)

    def test_long_prompts_reuse_the_same_passage(self):
        calls=[]
        def tokenizer(text,**kwargs):
            calls.append(text)
            return SimpleNamespace(input_ids=[99]+list(map(int,text.split())))
        passage=' '.join(map(str,range(100)))
        self.assertEqual(tokenize_passage(tokenizer,passage,64),[99]+list(range(63)))
        self.assertEqual(calls,[passage])
        result=tokenize_passage(tokenizer,passage,128)
        self.assertEqual(len(result),128)
        self.assertEqual(result[101:],[*range(27)])


if __name__=='__main__':unittest.main()
