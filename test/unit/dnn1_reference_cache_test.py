"""Immutable oracle identity must fail closed without touching CUDA."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from tilemega.dnn.reference_cache import RECIPE,load_reference_cache,sha


class ReferenceCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.export=root/'export';self.export.mkdir()
        (self.export/'checkpoint').mkdir()
        self.cache=root/'cache';self.cache.mkdir()
        self.bridge=root/'bridge.json';self.inputs=root/'inputs.safetensors'
        self.paths=dict(exported_program=self.export/'exported_program.pt2',
            checkpoint=self.export/'checkpoint/model.safetensors',
            manifest=self.export/'manifest.json',bridge=self.bridge,inputs=self.inputs)
        for name,path in self.paths.items():path.write_bytes(name.encode())
        self.values={'bf16/0':torch.ones((1,3,2,2),dtype=torch.bfloat16),
                     'fp32/0':torch.ones((1,3,2,2),dtype=torch.float32)}
        save_file(self.values,str(self.cache/'reference.safetensors'))
        self.manifest=dict(schema=1,batch=1,model='nafnet',cpu_only=True,cuda_initialized=False,
            recipe=json.loads(json.dumps(RECIPE)),
            recipe_sha256=hashlib.sha256(json.dumps(RECIPE,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            sources={name:{'sha256':sha(path)} for name,path in self.paths.items()},
            torch={'version':torch.__version__,'init_sha256':sha(torch.__file__),
                   'libtorch_cpu_sha256':sha(Path(torch.__file__).resolve().parent/'lib/libtorch_cpu.so')},
            reference_file_sha256=sha(self.cache/'reference.safetensors'),
            references={name:{'shape':list(v.shape),'dtype':str(v.dtype),
                'sha256':hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()}
                for name,v in self.values.items()})
        self.write()

    def write(self):
        (self.cache/'manifest.json').write_text(json.dumps(self.manifest))

    def load(self):
        return load_reference_cache(self.cache,self.export,self.bridge,1,self.inputs)

    def test_identified_outputs(self):
        bf16,fp32=self.load()
        self.assertTrue(torch.equal(bf16[0],self.values['bf16/0']))
        self.assertTrue(torch.equal(fp32[0],self.values['fp32/0']))
        self.assertFalse(torch.cuda.is_initialized())

    def test_each_source_is_bound(self):
        for name,path in self.paths.items():
            original=path.read_bytes();path.write_bytes(original+b' changed')
            with self.subTest(source=name),self.assertRaisesRegex(ValueError,'source mismatch'):
                self.load()
            path.write_bytes(original)

    def test_recipe_and_build_are_bound(self):
        self.manifest['recipe']['threads']=4;self.write()
        with self.assertRaisesRegex(ValueError,'recipe/domain'):self.load()
        self.manifest['recipe']=json.loads(json.dumps(RECIPE))
        self.manifest['torch']['version']='another build';self.write()
        with self.assertRaisesRegex(ValueError,'torch build'):self.load()

    def test_tensor_and_metadata_are_bound(self):
        self.manifest['reference_file_sha256']='0'*64;self.write()
        with self.assertRaisesRegex(ValueError,'tensor file'):self.load()
        self.manifest['reference_file_sha256']=sha(self.cache/'reference.safetensors')
        self.manifest['references']['fp32/0']['shape'][1]=4;self.write()
        with self.assertRaisesRegex(ValueError,'output identity'):self.load()

    def test_cache_cannot_change_reference_precision(self):
        self.values['fp32/0']=self.values['fp32/0'].to(torch.bfloat16)
        save_file(self.values,str(self.cache/'reference.safetensors'))
        self.manifest['reference_file_sha256']=sha(self.cache/'reference.safetensors')
        value=self.values['fp32/0'];record=self.manifest['references']['fp32/0']
        record['dtype']=str(value.dtype)
        record['sha256']=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        self.write()
        with self.assertRaisesRegex(ValueError,'precision or batch'):self.load()


if __name__=='__main__':unittest.main()
