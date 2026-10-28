"""CPU guards for a paired study that changes only derived KV storage."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tilemega.bench.as2_ablation import validate_cached_kv_bindings
from tilemega.bench.as2_models import RUNS


class CachedKvStudy(unittest.TestCase):
    def fixture(self,root):
        extent=16*8*1088*128
        def buffer(i,name,dtype,offset,size):
            return dict(id=i,name=name,dtype=dtype,offset_bytes=offset,
                        bytes=size,allocated_bytes=size,required_input=False)
        old=[buffer(0,'query','bf16',0,512),buffer(1,'kv_cache.k.0','bf16',512,extent*2),
             buffer(2,'kv_cache.v.0','bf16',512+extent*2,extent*2)]
        arena=512+extent*4
        extra=[buffer(3,'kv_cache.fp32.kv_cache.k.0','f32',arena,extent*4),
               buffer(4,'kv_cache.fp32.kv_cache.v.0','f32',arena+extent*4,extent*4)]
        paths=[]
        for cached in (False,True):
            p=root/str(cached);p.mkdir();paths.append(p)
            buffers=old+extra if cached else old;total=arena+extent*8 if cached else arena
            identity=dict(buffers=buffers,arena_bytes=total,input_buffers=[],
                kv_buffers=[1,2,3,4] if cached else [1,2],phases={})
            if cached:identity['derived_fp32_kv_state']=[['kv_cache.k.0','kv_cache.v.0',extra[0]['name'],extra[1]['name']]]
            for phase,variant in (('prefill',6),('decode',8)):
                d=p/phase;d.mkdir();m=[4,8,24,1,1,0,1,2,16,64 if phase=='prefill' else 1,1088,0,total,131072,2000000000]
                (d/'metadata.bin').write_bytes(struct.pack('<12I3Q',*m))
                s=[(1<<64)-1]*17+[0]*16
                s[0:3]=[0,512,512+extent*2];s[17]=13;s[23:27]=[8,128,2,1088];s[31]=9 if cached else variant
                if cached:s[10:12]=[arena,arena+extent*4]
                (d/'stages.bin').write_bytes(struct.pack('<17Q16I',*s))
                for name in ('tasks.bin','waits.bin','queue.bin','offsets.bin'):(d/name).write_bytes(b'unchanged-fifo')
                identity['phases'][phase]=dict(bound_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir()})
            (p/'request.json').write_text(json.dumps(identity))
        return paths

    def test_derived_state_is_permitted_but_math_and_ownership_are_not(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            a,b=self.fixture(Path(raw));validate_cached_kv_bindings(a,b)
            q=b/'decode/queue.bin';q.write_bytes(b'other-order')
            x=json.loads((b/'request.json').read_text());x['phases']['decode']['bound_hashes']['queue.bin']=hashlib.sha256(q.read_bytes()).hexdigest()
            (b/'request.json').write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'ownership, waits or order'):validate_cached_kv_bindings(a,b)
            q.write_bytes((a/'decode/queue.bin').read_bytes())
            x['phases']['decode']['bound_hashes']['queue.bin']=hashlib.sha256(q.read_bytes()).hexdigest()
            stage=b/'decode/stages.bin';s=list(struct.unpack('<17Q16I',stage.read_bytes()))
            s[32]=0x358637bd  # A changed epsilon cannot be hidden by new table hashes.
            stage.write_bytes(struct.pack('<17Q16I',*s))
            x['phases']['decode']['bound_hashes']['stages.bin']=hashlib.sha256(stage.read_bytes()).hexdigest()
            (b/'request.json').write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'logical operands or mathematics'):validate_cached_kv_bindings(a,b)

    def test_state_must_be_reset_and_inputs_cannot_change(self):
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            a,b=self.fixture(Path(raw));path=b/'request.json';original=json.loads(path.read_text())
            x=copy.deepcopy(original);x['kv_buffers'].remove(3);path.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'request-reset'):validate_cached_kv_bindings(a,b)
            x=copy.deepcopy(original);x['input_buffers']=[dict(sha256='other-weight')];path.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'canonical packed inputs'):validate_cached_kv_bindings(a,b)


if __name__=='__main__':unittest.main()
