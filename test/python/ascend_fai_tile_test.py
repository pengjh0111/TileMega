import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from tilemega.bench import as2_body as body


class FaiTileReferenceContracts(unittest.TestCase):
    def setUp(self):
        (body.RUNS/'tmp').mkdir(exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(prefix='fai_cpu_',dir=body.RUNS/'tmp')
        self.root=Path(self.directory.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.directory.cleanup()

    def prepare(self,rows,width,keys,name):
        out=self.root/name
        with contextlib.redirect_stdout(io.StringIO()):
            body.prepare_fai_tile(SimpleNamespace(rows=rows,head_dim=width,keys=keys,out=out))
        return out,json.loads((out/'config.json').read_text())

    def test_distinct_domains_padding_tail_and_reference(self):
        for rows,d,keys in ((3,128,129),(32,64,257)):
            out,c=self.prepare(rows,d,keys,f'd{d}')
            q=body.read(out/'q.bin',torch.bfloat16,(2,32,d))
            k=body.read(out/'k.bin',torch.bfloat16,(2,1088,d))
            v=body.read(out/'v.bin',torch.bfloat16,(2,1088,d))
            ref=body.read(out/'ref_fp32.bin',torch.float32,(2,32,d))
            self.assertFalse(torch.equal(k[0,:keys],k[1,:keys]))
            self.assertTrue(torch.isnan(k[:,keys:]).all())
            self.assertTrue(torch.isnan(v[:,keys:]).all())
            self.assertTrue(torch.count_nonzero(q[:,rows:])==0)
            for domain in range(2):
                scores=(q[domain].float()@k[domain,:keys].float().T)*d**-.5
                expected=scores.softmax(-1)@v[domain,:keys].float()
                torch.testing.assert_close(ref[domain],expected,atol=1e-6,rtol=1e-6)
            self.assertEqual(c['tasks'],2)
            self.assertEqual(c['epochs'],4)
            self.assertIn('no causal/prefill claim',c['reference'])
            self.assertEqual(c['precision_contract_id'],'catlass49_bf16_qkv_p_fp32_online_statistics_v1')

    def test_reproducible_identity_and_immutable_directory(self):
        a,ca=self.prepare(3,128,129,'first')
        _,cb=self.prepare(3,128,129,'second')
        self.assertEqual(ca['input_sha256'],cb['input_sha256'])
        self.assertEqual(ca['correctness_policy_sha256'],cb['correctness_policy_sha256'])
        with self.assertRaises(FileExistsError):
            body.prepare_fai_tile(SimpleNamespace(rows=3,head_dim=128,keys=129,out=a))
        with self.assertRaises(ValueError):
            body.prepare_fai_tile(SimpleNamespace(rows=33,head_dim=128,keys=129,out=self.root/'invalid'))

    def test_compare_rejects_cpu_fixture_as_device_evidence(self):
        source,c=self.prepare(3,128,129,'inputs')
        job=self.root/'job';job.mkdir();out=self.root/'outputs';out.mkdir()
        receipt=dict(project='as2',status='passed',exit_code=0,pid=123,cpu_test_fixture=True,
            command=['/private/as2_fai_tile','3','128','129',str(source),str(out)])
        (job/'job.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'completed matching'):
            body.compare_fai_tile(SimpleNamespace(input=source,output=out,job=job))
        self.assertFalse((out/'numeric.json').exists())


if __name__=='__main__':
    unittest.main()
