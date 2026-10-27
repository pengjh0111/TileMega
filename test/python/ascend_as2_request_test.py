"""Address relocation preserves solved queues and rejects shared-state mismatches."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import torch
from tilemega.bench.as2_models import RUNS
from tilemega.bench.as2_request import bind, digest
from tilemega.bench.as2_native_audit import compare_capture_payloads
from tilemega.bench.as2_native import incremental_packing_bytes
from tilemega.bench.as2_last_forward import matching_prefix_rows


class LastForwardRowsTest(unittest.TestCase):
    def test_divergent_feedback_is_excluded_only_in_explicit_diagnostics(self):
        reference=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
        actual=reference.clone();actual[0,-1]=10;actual[1,0]=11
        with self.assertRaisesRegex(ValueError,'diverged'):matching_prefix_rows(actual,reference)
        self.assertEqual(matching_prefix_rows(actual,reference,True),([0,2],[1]))
        with self.assertRaisesRegex(ValueError,'No same-token-prefix'):matching_prefix_rows(reference+1,reference,True)


class PackingBudgetTest(unittest.TestCase):
    def test_hardlinks_preserve_budget_and_changed_layout_requires_space(self):
        def buffer(name,size,recipe):return dict(name=name,bytes=size,dtype='bf16',pack_json=recipe,required_input=True)
        weight=buffer('weight',1<<30,'{"kind":"qkv_group_interleave"}')
        buffers=[weight,buffer('alias',1024,'{"kind":"alias"}'),buffer('rope',8192,''),buffer('tokens',4096,'')]
        key=(weight['name'],weight['bytes'],weight['dtype'],weight['pack_json'])
        reused=incremental_packing_bytes(buffers,{key:('record','file')})
        self.assertEqual(reused,(1<<20)+4*4096+8192+4096)
        self.assertEqual(incremental_packing_bytes(buffers,{})-reused,1<<30)
        changed=[dict(weight,pack_json='{"kind":"gate_up_interleave"}'),*buffers[1:]]
        self.assertEqual(incremental_packing_bytes(changed,{key:('record','file')})-reused,1<<30)


class RequestCapturePayloadTest(unittest.TestCase):
    def setUp(self):
        self.required={'tokens.bin','prefill/control.bin','decode/control.bin','decode/1.bin'}
        self.fast={name:('same',16) for name in self.required}
        self.fast['decode/1.bin']=('logits',4<<20)
        self.diagnostic=dict(self.fast,**{'decode/2.bin':('large_extra',2<<20)})

    def test_all_fast_outputs_and_only_large_numeric_extras(self):
        self.assertEqual(compare_capture_payloads(self.diagnostic,self.fast,self.required),['decode/2.bin'])

    def test_active_capture_requires_a_bound_phase_extent(self):
        data=dict(self.fast);data['decode/active/3.bin']=('active',65536)
        with self.assertRaises(ValueError):compare_capture_payloads(data,self.fast,self.required)
        self.assertEqual(compare_capture_payloads(data,self.fast,self.required,{'decode/active/3.bin':65536}),['decode/active/3.bin'])
        for size in (32768,2<<20):
            with self.assertRaises(ValueError):compare_capture_payloads(data,self.fast,self.required,{'decode/active/3.bin':size})

    def test_reject_missing_logits_changes_and_small_or_semantic_extras(self):
        for name,value in [('decode/2.bin',('small',16)),('decode/kv_k_s0.bin',('kv',2<<20))]:
            data=dict(self.fast);data[name]=value
            with self.assertRaises(ValueError):compare_capture_payloads(data,self.fast,self.required)
        for name in self.required:
            data=dict(self.fast);del data[name]
            with self.assertRaises(ValueError):compare_capture_payloads(self.diagnostic,data,self.required)
            data=dict(self.fast);data[name]=('changed',data[name][1])
            with self.assertRaises(ValueError):compare_capture_payloads(self.diagnostic,data,self.required)


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
        for phase,size in [('prefill',64*32),('decode',32)]:
            extents=struct.unpack('<4Q',(out/phase/'active_bytes.bin').read_bytes())
            self.assertEqual(extents,(4352,512,4352,size))
            self.assertEqual(receipt['phases'][phase]['bound_hashes']['active_bytes.bin'],digest(out/phase/'active_bytes.bin'))

    def test_reject_config_mismatch(self):
        p=self.dec.parent/'import.json';d=json.loads(p.read_text());d['config_sha256']='wrong';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'config mismatch'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_reject_changed_input(self):
        (self.inputs/'1.bin').write_bytes(b'x'*512)
        with self.assertRaisesRegex(ValueError,'input identity changed'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_reject_incompatible_kv(self):
        p=self.dec/'buffers.json';d=json.loads(p.read_text());d['buffers'][2]['dtype']='f32';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'Incompatible shared buffer'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_reject_phase_geometry_mismatch(self):
        p=self.dec/'stages.bin';stage=list(struct.unpack('<17Q16I',p.read_bytes()));stage[31]=1
        p.write_bytes(struct.pack('<17Q16I',*stage))
        with self.assertRaisesRegex(ValueError,'different compiled GEMM geometries'):
            bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def abi4(self):
        for path in (self.pre,self.dec):
            metadata=list(struct.unpack('<12I3Q',(path/'metadata.bin').read_bytes()));metadata[0]=4
            (path/'metadata.bin').write_bytes(struct.pack('<12I3Q',*metadata))
        p=self.inputs/'inputs.json';x=json.loads(p.read_text());x['metadata_sha256']=digest(self.pre/'metadata.bin');p.write_text(json.dumps(x))

    def test_abi4_relocation_preserves_phase_identity(self):
        self.abi4();out=self.root/'abi4';bind(self.pre,self.dec,self.inputs,out)
        for phase,path in [('prefill',self.pre),('decode',self.dec)]:
            self.assertEqual(struct.unpack('<I',(out/phase/'metadata.bin').read_bytes()[:4])[0],4)
            self.assertEqual(digest(path/'tasks.bin'),digest(out/phase/'tasks.bin'))

    def test_reject_mixed_phase_abi(self):
        self.abi4();p=self.dec/'metadata.bin';m=list(struct.unpack('<12I3Q',p.read_bytes()));m[0]=3;p.write_bytes(struct.pack('<12I3Q',*m))
        with self.assertRaisesRegex(ValueError,'same validated worker topology'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_split_partial_and_combine_addresses_relocate_without_rescheduling(self):
        self.abi4()
        p=self.pre/'buffers.json';description=json.loads(p.read_text());buffers=description['buffers']
        for name,size,dtype in [('native.gemm.partial.0',96,'f32'),('native.gemm.raw.0',16,'bf16')]:
            offset=(1<<32)+4096 if len(buffers)==4 else buffers[-1]['offset_bytes']+buffers[-1]['allocated_bytes']
            buffers.append(dict(id=len(buffers),name=name,dtype=dtype,offset_bytes=offset,bytes=size,
                allocated_bytes=1024,external_name='',pack_json='',required_input=False))
        p.write_text(json.dumps(description))
        (self.pre/'allocations.bin').write_bytes(b''.join(struct.pack('<3Q2I',b['offset_bytes'],b['bytes'],b['allocated_bytes'],b['required_input'],0) for b in buffers))
        partial=list(struct.unpack('<17Q16I',(self.pre/'stages.bin').read_bytes()))
        partial[16]=buffers[-2]['offset_bytes'];partial[23]=3;partial[24]=2048;partial[25]=1;partial[31]=1
        combine=partial.copy();combine[17]=6;combine[18]=1;combine[5]=buffers[-1]['offset_bytes']
        (self.pre/'stages.bin').write_bytes(struct.pack('<17Q16I',*partial)+struct.pack('<17Q16I',*combine))
        m=list(struct.unpack('<12I3Q',(self.pre/'metadata.bin').read_bytes()));m[4]=2;m[12]=buffers[-1]['offset_bytes']+1024
        (self.pre/'metadata.bin').write_bytes(struct.pack('<12I3Q',*m))
        stage=list(struct.unpack('<17Q16I',(self.dec/'stages.bin').read_bytes()));stage[31]=1
        (self.dec/'stages.bin').write_bytes(struct.pack('<17Q16I',*stage))
        p=self.inputs/'inputs.json';x=json.loads(p.read_text());x['metadata_sha256']=digest(self.pre/'metadata.bin');p.write_text(json.dumps(x))
        out=self.root/'decomposed';bind(self.pre,self.dec,self.inputs,out)
        bound=list(struct.iter_unpack('<17Q16I',(out/'prefill/stages.bin').read_bytes()))
        named={b['name']:b for b in json.loads((out/'request.json').read_text())['buffers']}
        self.assertEqual(bound[0][16],named['native.gemm.partial.0']['offset_bytes'])
        self.assertEqual(bound[1][16],bound[0][16])
        self.assertEqual(bound[1][5],named['native.gemm.raw.0']['offset_bytes'])
        self.assertNotEqual(bound[0][16],partial[16])
        self.assertEqual(bound[0][17:],tuple(partial[17:]))
        self.assertEqual(bound[1][17:],tuple(combine[17:]))
        for name in ('tasks.bin','waits.bin','offsets.bin','queue.bin'):
            self.assertEqual(digest(self.pre/name),digest(out/'prefill'/name))

    def test_reject_old_prices_after_abi_change(self):
        self.abi4();self.priced()
        with self.assertRaisesRegex(ValueError,'different runtime ABI'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def priced(self):
        for phase,path in [('prefill',self.pre),('decode',self.dec)]:
            metadata=list(struct.unpack('<12I3Q',(path/'metadata.bin').read_bytes()));metadata[11]=0
            (path/'metadata.bin').write_bytes(struct.pack('<12I3Q',*metadata))
            (path/'pricing_bounds.bin').write_bytes(struct.pack('<4I',1,0,0 if phase=='prefill' else 66,2))
            (path/'prices.json').write_text(json.dumps(dict(model_id='test',config_sha256='config',batch=1,seq=metadata[9],groups=2,phase=phase,past_min=0,past_max=0 if phase=='prefill' else 66)))
            p=path.parent/'import.json';x=json.loads(p.read_text());x['priced_solver_executed']=True;p.write_text(json.dumps(x))
        p=self.inputs/'inputs.json';x=json.loads(p.read_text());x['metadata_sha256']=digest(self.pre/'metadata.bin');p.write_text(json.dumps(x))

    def test_priced_provenance_and_bounds_survive_relocation(self):
        self.priced();out=self.root/'priced';bind(self.pre,self.dec,self.inputs,out)
        receipt=json.loads((out/'request.json').read_text())
        self.assertEqual(receipt['optimization_status'],'measured_native_rank_eft_shared_gm')
        for phase,path in [('prefill',self.pre),('decode',self.dec)]:
            for name in ('pricing_bounds.bin','prices.json'):
                self.assertEqual(receipt['phases'][phase]['bound_hashes'][name],digest(path/name))

    def test_reject_mixed_pricing_modes(self):
        self.priced();m=list(struct.unpack('<12I3Q',(self.dec/'metadata.bin').read_bytes()));m[11]=1
        (self.dec/'metadata.bin').write_bytes(struct.pack('<12I3Q',*m))
        with self.assertRaisesRegex(ValueError,'same pricing mode'):bind(self.pre,self.dec,self.inputs,self.root/'bad')

    def test_reject_out_of_range_empirical_context(self):
        self.priced();(self.dec/'pricing_bounds.bin').write_bytes(struct.pack('<4I',1,0,1088,2))
        with self.assertRaisesRegex(ValueError,'pricing bounds'):bind(self.pre,self.dec,self.inputs,self.root/'bad')


if __name__=='__main__':unittest.main()
