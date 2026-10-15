"""CPU contracts prevent a cached TensorRT binary from crossing build identity."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_dnn_baselines', ROOT / 'python/tilemega/dnn/baselines.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def identity():
    return dict(schema_version=1, onnx_sha256='a'*64, exported_program_sha256='b'*64,
                source_checkpoint_sha256='c'*64, precision='bf16', tensorrt_version='10.11.0.33',
                gpu_uuid='GPU-fixture', compute_capability=[9, 0], static_batch=1,
                builder_flags={'builder_optimization_level':5, 'precision_flag':'BF16'})


class EngineCacheTests(unittest.TestCase):
    def test_repeated_fresh_load_builds_once(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            a,path,meta=baseline.serialized_engine_cache(folder,identity(),lambda:calls.append(1) or b'official serialized fixture')
            b,path2,meta2=baseline.serialized_engine_cache(folder,identity(),lambda: self.fail('cache hit rebuilt engine'))
            self.assertEqual(a,b);self.assertEqual(path,path2);self.assertEqual(calls,[1])
            self.assertFalse(meta['hit']);self.assertTrue(meta2['hit'])

    def test_two_fresh_processes_share_one_build_under_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            code = """
import importlib.util,json,pathlib,sys,time
spec=importlib.util.spec_from_file_location('baseline',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def build():
    with (pathlib.Path(sys.argv[2])/'builds.log').open('a') as f:f.write('one build\\n')
    time.sleep(0.05)
    return b'serialized fixture'
_,_,meta=m.serialized_engine_cache(sys.argv[2],json.loads(sys.argv[3]),build)
print(json.dumps(meta))
"""
            cmd=[sys.executable,'-c',code,str(ROOT/'python/tilemega/dnn/baselines.py'),folder,json.dumps(identity())]
            children=[subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
            results=[]
            for child in children:
                stdout,stderr=child.communicate(timeout=20)
                self.assertEqual(child.returncode,0,stderr);results.append(json.loads(stdout))
            self.assertEqual((Path(folder)/'builds.log').read_text().splitlines(),['one build'])
            self.assertEqual(len({x['builder_pid'] for x in results}),1)
            self.assertEqual(len({x['deserialize_pid'] for x in results}),2)
            self.assertEqual(sorted(x['hit'] for x in results),[False,True])

    def test_each_build_coordinate_has_a_distinct_key(self):
        original=identity();key=baseline.engine_cache_key(original)
        for field,value in [('onnx_sha256','d'*64),('exported_program_sha256','d'*64),
                            ('source_checkpoint_sha256','d'*64),('precision','fp16'),
                            ('tensorrt_version','10.12'),('gpu_uuid','GPU-other'),
                            ('compute_capability',[8,9]),('static_batch',2),
                            ('builder_flags',{'builder_optimization_level':4})]:
            with self.subTest(field=field):
                changed=copy.deepcopy(original);changed[field]=value
                self.assertNotEqual(key,baseline.engine_cache_key(changed))

    def test_serialized_corruption_is_rejected_before_builder_or_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            _,path,_=baseline.serialized_engine_cache(folder,identity(),lambda:b'engine')
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                baseline.serialized_engine_cache(folder,identity(),lambda:self.fail('corruption must not rebuild'))

    def test_wrong_identity_metadata_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            _,path,_=baseline.serialized_engine_cache(folder,identity(),lambda:b'engine')
            meta_path=path.parent/'identity.json';meta=json.loads(meta_path.read_text())
            meta['identity']['precision']='fp16';meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'identity conflict'):
                baseline.serialized_engine_cache(folder,identity(),lambda:self.fail('conflict must not rebuild'))

    def test_partial_publication_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            entry=Path(folder)/baseline.engine_cache_key(identity());entry.mkdir()
            (entry/'engine.plan').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'partial'):
                baseline.serialized_engine_cache(folder,identity(),lambda:self.fail('partial must not rebuild'))

    def test_unidentified_engine_cannot_enter_cache(self):
        for change in ({'onnx_sha256':'not-a-hash'},{'gpu_uuid':''},{'precision':'fp8'}):
            with self.assertRaises(ValueError):baseline.engine_cache_key(dict(identity(),**change))
        incomplete=identity();del incomplete['builder_flags']
        with self.assertRaises(ValueError):baseline.engine_cache_key(incomplete)

    def test_engine_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _,path,_=baseline.serialized_engine_cache(folder,identity(),lambda:b'engine')
            stored=path.with_suffix('.original');path.rename(stored);path.symlink_to(stored)
            with self.assertRaisesRegex(ValueError,'partial'):
                baseline.serialized_engine_cache(folder,identity(),lambda:self.fail('symlink must not rebuild'))


if __name__ == '__main__':unittest.main()
