"""Native artifact allocations must describe the linked RDC entry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tilemega.build.identity import bind_execution, digest, generate, sha, verify
from tilemega.build.linked_resources import kernel_resources


class NativeLinkedIdentityTest(unittest.TestCase):
    def fixture(self, directory):
        root=Path(directory);so=root/'plan.so';so.write_bytes(b'linked binary')
        (root/'runtime.h').write_text('source')
        snapshot=root/'source.json'
        snapshot.write_text(json.dumps(dict(source_files={'runtime.h':sha(root/'runtime.h')})))
        Path(str(so)+'.cu').write_text('#define TILEMEGA_TRACE_STAGE 0\n')
        Path(str(so)+'.plan.json').write_text(json.dumps(dict(native_dense={},gemms=[])))
        Path(str(so)+'.build_command.txt').write_text('nvcc -rdc=true -arch=sm_90a plan.so.cu')
        Path(str(so)+'.ptxas.log').write_text("""Compiling entry function 'sm90_dense_native_kernel'
Function properties for sm90_dense_native_kernel
Used 74 registers
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Function properties for consumer
8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
""")
        raw='''Fatbin elf code:
arch = sm_90a
Function sm90_dense_native_kernel:
 REG:254 STACK:8 SHARED:1024 LOCAL:0
'''
        linked=dict(binary_sha256=sha(so),arch='sm_90a',stdout=raw,stderr='',
            resources=kernel_resources(raw,'sm_90a',['sm90_dense_native_kernel']),
            command=['cuobjdump','--dump-resource-usage',str(so)],tool_sha256='a'*64)
        return so,snapshot,linked

    def generate(self,so,snapshot,linked):
        library=Mock();library.tm_plan_shared_bytes.return_value=199808
        with patch('tilemega.build.identity.ctypes.CDLL',return_value=library), \
                patch('tilemega.build.identity.subprocess.check_output',return_value='nvcc 12.8'), \
                patch('tilemega.build.feature_contract.validate'), \
                patch('tilemega.build.feature_contract.validate_gemm_dispatch',return_value=[]), \
                patch('tilemega.build.feature_contract.implementations',return_value={}), \
                patch('tilemega.build.linked_resources.inspect',return_value=linked):
            return generate(so,so.parent,snapshot)

    def verify(self,so):
        with patch('tilemega.build.feature_contract.validate_gemm_dispatch',return_value=[]):
            return verify(so)

    def test_linked_entry_replaces_prelink_allocation_and_keeps_callee_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            so,snapshot,linked=self.fixture(directory)
            identity=self.generate(so,snapshot,linked)
            self.assertEqual(identity['ptxas_resources']['sm90_dense_native_kernel']['registers'],74)
            row=identity['resources']['sm90_dense_native_kernel']
            self.assertEqual((row['registers'],row['stack_bytes']),(254,8))
            self.assertEqual(identity['shared_memory_bytes'],199808)
            self.assertEqual(row['static_smem_bytes'],1024)
            self.assertEqual(row['callees']['consumer']['spill_store_bytes'],8)
            self.assertTrue(bind_execution(identity,'L1',True,static_dense=True)['spill'])
            self.assertEqual(self.verify(so),identity)

    def test_verify_checks_internal_linked_evidence_even_after_outer_rehash(self):
        for part in ['allocation','dump','arch','binary']:
            with self.subTest(part=part),tempfile.TemporaryDirectory() as directory:
                so,snapshot,linked=self.fixture(directory)
                identity=self.generate(so,snapshot,linked)
                if part=='allocation':identity['resources']['sm90_dense_native_kernel']['registers']=74
                elif part=='dump':identity['linked_resource_evidence']['stdout']=linked['stdout'].replace('REG:254','REG:74')
                elif part=='arch':identity['linked_resource_evidence']['arch']='sm_90'
                else:identity['linked_resource_evidence']['binary_sha256']='changed'
                identity['artifact_id']=digest({k:v for k,v in identity.items() if k!='artifact_id'})
                Path(str(so)+'.identity.json').write_text(json.dumps(identity))
                with self.assertRaises(ValueError):self.verify(so)

    def test_generation_rejects_inspection_of_different_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            so,snapshot,linked=self.fixture(directory);linked['binary_sha256']='changed'
            with self.assertRaises(ValueError):self.generate(so,snapshot,linked)
            self.assertFalse(Path(str(so)+'.identity.json').exists())

    def test_legacy_identity_without_linked_evidence_remains_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            so,snapshot,linked=self.fixture(directory)
            identity=self.generate(so,snapshot,linked)
            identity['resources']=identity.pop('ptxas_resources')
            identity.pop('linked_resource_evidence')
            identity['artifact_id']=digest({k:v for k,v in identity.items() if k!='artifact_id'})
            Path(str(so)+'.identity.json').write_text(json.dumps(identity))
            self.assertEqual(self.verify(so),identity)


if __name__=='__main__':unittest.main()
