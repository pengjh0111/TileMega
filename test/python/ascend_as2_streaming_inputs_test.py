"""Poisoned KV tails and independent batch/head/query domains; no device work."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import torch
from tilemega.bench.as2_body import RUNS,prepare_attention,read


class StreamingInputs(unittest.TestCase):
    def test_gqa3_prefill_tail_ignores_poison_and_keeps_independent_batches(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            out=Path(raw)/'input'
            prepare_attention(argparse.Namespace(qheads=6,kvheads=2,keys=17,head_dim=64,
                batch=16,query_rows=16,capacity=19,out=out))
            m=json.loads((out/'config.json').read_text())
            self.assertEqual(m['shape'],[16,6,16,64])
            k=read(out/'k.bin',torch.bfloat16,(16,2,19,64))
            v=read(out/'v.bin',torch.bfloat16,(16,2,19,64))
            y=read(out/'ref_fp32.bin',torch.float32,m['shape'])
            self.assertTrue(torch.isnan(k[:,:,17:]).all() and torch.isnan(v[:,:,17:]).all())
            self.assertTrue(torch.isfinite(y).all());self.assertFalse(torch.equal(y[0],y[1]))
            self.assertFalse(torch.equal(y[:,0],y[:,3]))
            for name,sha in m['input_sha256'].items():
                self.assertEqual(__import__('hashlib').sha256((out/name).read_bytes()).hexdigest(),sha)


if __name__=='__main__':unittest.main()
