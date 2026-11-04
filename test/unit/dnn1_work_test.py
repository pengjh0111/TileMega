"""CPU-only checks for typed batch geometry and semantic full-batch work."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import dnn1_work
from dnn1_logical_work import logical_work

class WorkGeometry(unittest.TestCase):
    def test_explicit_batch_templates_and_original_arity(self):
        bodies=[('layout','Sm90ForwardLayout<64,128,16>'),
                ('pool','Sm90ForwardMaxPool<3,2,16>'),
                ('global_pool','Sm90ForwardGlobalPool<16>'),
                ('depthwise','Sm90ForwardDepthwise<8,8,64,1,2,Chain,8,16>'),
                ('dwpw','Sm90ForwardDwPw<8,8,64,64,1,2,Chain,8,Dw,16>')]
        shape=dict(kWidth=32,kHeight=32,kChannels=64,kOutChannels=64,
                   kOutputHeight=16,kOutputWidth=16,kPaddedChannels=64,kDepthwisePack=8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source.cu';manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(runtime_batch=16,grid=132,stages=[dict(body=k) for k,_ in bodies])))
            src.write_text('\n'.join(f'using Body_{i}={t};' for i,(_,t) in enumerate(bodies)))
            with patch('tilemega.build.forward_contract.dispatch'),patch.object(dnn1_work,'generated_tables',return_value=([],[])),patch.object(dnn1_work,'constants',side_effect=lambda s,k,i:shape if k=='Shape' else {}):
                rows=dnn1_work.describe(src,manifest)['stages']
                self.assertEqual([r['covered_batches'] for r in rows],[16]*5)
                self.assertEqual([r['tasks'] for r in rows],[128,4096,16,64,64])
                self.assertEqual(rows[3]['activation_bytes'],16*32*32*64*2)
                self.assertEqual(rows[3]['weight_bytes'],64*9*8*2)
                previous_pool = rows[1]
                flat = bodies.copy()
                flat[1] = ('pool', 'Sm90ForwardMaxPool<3,2,16,128>')
                src.write_text('\n'.join(f'using Body_{i}={t};' for i,(_,t) in enumerate(flat)))
                flattened = dnn1_work.describe(src,manifest)['stages'][1]
                self.assertEqual(flattened['tasks'], 2048)
                for key in ('covered_batches', 'activation_bytes', 'output_bytes', 'flops'):
                    self.assertEqual(flattened[key], previous_pool[key])
                old=['Sm90ForwardLayout<64>','Sm90ForwardMaxPool<3,2>','Sm90ForwardGlobalPool','Sm90ForwardDepthwise<8,8,64,1,2,Chain,8>','Sm90ForwardDwPw<8,8,64,64,1,2,Chain,8,Dw>']
                src.write_text('\n'.join(f'using Body_{i}={t};' for i,t in enumerate(old)))
                rows=dnn1_work.describe(src,manifest)['stages']
                self.assertEqual([r['covered_batches'] for r in rows],[1]*5)
                self.assertEqual([r['tasks'] for r in rows],[1024,256,1,4,4])

    def test_semantic_conv_batch_uses_runtime_not_static_export_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bridge=root/'bridge.json';bridge.write_text('{}');source=root/'plan.json'
            c=dict(batch=1,output_height=7,output_width=7,channels=32,kernel_r=3,kernel_s=3)
            g=dict(rows_per_batch=49,n=64,k=288)
            stages=[dict(index=i,body=body,representative=body,conv_descriptor=c,gemm_descriptor=g if body!='depthwise' else None) for i,body in enumerate(('conv','depthwise','dwpw'))]
            results=[]
            for batch in (1,16):
                source.write_text(json.dumps(dict(batch=batch,bridge=str(bridge),buffers=[],stages=stages)))
                manifest=dict(runtime_batch=batch,export_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),stages=stages)
                results.append(logical_work(source,manifest))
            for one,many in zip(*results):
                self.assertEqual(many['logical_flops'],16*one['logical_flops'])
                self.assertEqual(many['logical_weight_bytes'],one['logical_weight_bytes'])
                self.assertEqual(many['logical_descriptor_batch'],1)
                self.assertEqual(many['logical_runtime_batch'],16)

if __name__=='__main__':unittest.main()
