"""A transported VM retains distinct tuning/execution device provenance."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('publisher',ROOT/'scripts/dnn1_tvm_publish.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Provenance(unittest.TestCase):
    def fixture(self):
        identity=dict(tvm_version='0.27.0',tvm_commit='commit',gpu_uuid='new',
            recipe=dict(schema=6),dependency_versions={'apache-tvm-ffi':'ffi'},
            tvm_source_identity=dict(ffi_commit='ffi_commit',source_diff_sha256='diff'))
        diag=dict(tvm_version='0.27.0',tvm_commit='commit',cache_identity=identity,model_sha256='vm')
        transport=dict(source_identity=dict(identity,gpu_uuid='old'),destination_identity=identity,
            source_gpu_uuid='old',execution_gpu_uuid='new',tuning_gpu_uuid='origin',
            source_vm_sha256='vm',retuned=False,compiler_or_model_changed=False,
            source_compiled_sha256='source',version='tvm-cache-transport-v1')
        return dict(configuration=dict(tvm_diag=diag)),transport

    def test_untransported_versions_remain_unchanged(self):
        receipt,_=self.fixture()
        self.assertEqual(set(module.source_versions(receipt)),{'tvm','tvm_source_commit','apache-tvm-ffi',
            'tvm_ffi_source_commit','tvm_source_diff_sha256','tvm_recipe_schema'})

    def test_valid_transport_keeps_origin_and_execution_uuids(self):
        receipt,transport=self.fixture();receipt['configuration']['tvm_diag']['cache_transport']=transport
        versions=module.source_versions(receipt)
        self.assertEqual(versions['tvm_tuning_gpu_uuid'],'origin')
        self.assertEqual(versions['tvm_execution_gpu_uuid'],'new')
        self.assertEqual(versions['tvm_transport_source_compiled_sha256'],'source')

    def test_mismatched_artifact_or_identity_rejected(self):
        for field,value in [('source_vm_sha256','changed'),('execution_gpu_uuid','wrong'),
                            ('compiler_or_model_changed',True),('retuned',True)]:
            receipt,transport=self.fixture();transport[field]=value
            receipt['configuration']['tvm_diag']['cache_transport']=transport
            with self.subTest(field=field),self.assertRaises(ValueError):module.source_versions(receipt)
        receipt,transport=self.fixture();transport['source_identity']['recipe']=dict(schema=7)
        receipt['configuration']['tvm_diag']['cache_transport']=transport
        with self.assertRaises(ValueError):module.source_versions(receipt)


if __name__=='__main__':unittest.main()
