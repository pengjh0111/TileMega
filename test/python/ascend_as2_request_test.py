"""Address relocation preserves solved queues and rejects shared-state mismatches."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from tilemega.bench.as2_models import RUNS
from tilemega.bench.as2_request import bind, digest


class RequestBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='request_contract_',dir=RUNS/'search')
        self.root=Path(self.temp.name);self.inputs=self.root/'inputs';self.inputs.mkdir()
        self.pre=self.phase('prefill',64);self.dec=self.phase('decode',1)
        files=[]
        for i,name,data in [(0,'serving.tokens',bytes(4352)),(1,'w',bytes(512))]:
            path=self.inputs/f'{i}.bin';path.write_bytes(data)
            files.append(dict(name=name,file=path.name,bytes=len(data),sha256=digest(path)))
        identity=dict(model_id='test',batch=1,seq=64,past=0,config_sha256='config',
            weight_manifest_sha256='weights',input_sha256='tokens',metadata_sha256=digest(self.pre/'metadata.bin'),buffers=files)
        (self.inputs/'inputs.json').write_text(json.dumps(identity))

    def tearDown(self):self.temp.cleanup()

    def phase(self,phase,seq):
        parent=self.root/phase;parent.mkdir();path=parent/'plan';path.mkdir()
        (parent/'import.json').write_text(json.dumps(dict(phase=phase,model_id='test',config_sha256='config')))
        buffers=[];offset=0
        for name,size,required,dtype,recipe in [('serving.tokens',4352,True,'i32',''),
                ('w',512,True,'bf16','{"kind":"alias","source":"w"}'),
                ('kv_cache.k.0',4352,False,'bf16',''),('hidden',seq*32,False,'bf16','')]:
            allocated=((size+1023)//512)*512
            buffers.append(dict(id=len(buffers),name=name,dtype=dtype,offset_bytes=offset,
                bytes=size,allocated_bytes=allocated,external_name=name if required else '',pack_json=recipe,required_input=required))
            offset+=allocated
        (path/'buffers.json').write_text(json.dumps(dict(buffers=buffers)))
        (path/'metadata.bin').write_bytes(struct.pack('<12I3Q',2,2,6,1,1,0,1,16,1,seq,1088,1,offset,131072,2000000000))
        (path/'allocations.bin').write_bytes(b''.join(struct.pack('<3Q2I',b['offset_bytes'],b['bytes'],b['allocated_bytes'],b['required_input'],0) for b in buffers))
        operands=[b['offset_bytes'] for b in buffers]+[(1<<64)-1]*13
        (path/'stages.bin').write_bytes(struct.pack('<17Q16I',*operands,*([0]*16)))
        for name in ('tasks.bin','waits.bin','offsets.bin','queue.bin'):(path/name).write_bytes(name.encode())
        (path/'implementations.json').write_text('[]')
        return path

    def test_shared_buffers_and_unchanged_schedules(self):
        out=self.root/'bound';bind(self.pre,self.dec,self.inputs,out)
        receipt=json.loads((out/'request.json').read_text())
        self.assertEqual(receipt['kv_buffers'],[2])
        for phase,source in [('prefill',self.pre),('decode',self.dec)]:
            for name in ('tasks.bin','waits.bin','offsets.bin','queue.bin'):
                self.assertEqual(digest(source/name),digest(out/phase/name))
            stage=struct.unpack('<17Q16I',(out/phase/'stages.bin').read_bytes())
            self.assertEqual(stage[:4],tuple(b['offset_bytes'] for b in receipt['buffers']))
        self.assertEqual((out/'prefill/allocations.bin').read_bytes(),(out/'decode/allocations.bin').read_bytes())
        self.assertEqual((out/'inputs/1.bin').stat().st_ino,(self.inputs/'1.bin').stat().st_ino)
        self.assertEqual(receipt['buffers'][-1]['bytes'],64*32)

    def test_reject_config_mismatch(self):
        p=self.dec.parent/'import.json';d=json.loads(p.read_text());d['config_sha256']='wrong';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'config mismatch'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_reject_changed_input(self):
        (self.inputs/'1.bin').write_bytes(b'x'*512)
        with self.assertRaisesRegex(ValueError,'input identity changed'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_reject_incompatible_kv(self):
        p=self.dec/'buffers.json';d=json.loads(p.read_text());d['buffers'][2]['dtype']='f32';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'Incompatible shared buffer'):bind(self.pre,self.dec,self.inputs,self.root/'bad')


if __name__=='__main__':unittest.main()
