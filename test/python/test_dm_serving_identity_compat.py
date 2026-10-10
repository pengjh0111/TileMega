"""Host-only ABI compatibility for coexisting DM and R14 artifact schemas."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tilemega.build.identity import optional_identity,bind_execution,digest,sha

class IdentityCompatibility(unittest.TestCase):
    def fixture(self,root):
        so=root/'plan.so';so.write_bytes(b'host fixture')
        for suffix,value in (('.cu','source'),('.plan.json','{}')):
            Path(str(so)+suffix).write_text(value)
        result=dict(schema='tilemega.dm1.identity.v1',so_sha256=sha(so),binary_sha256=sha(so),
                    cu_sha256=sha(str(so)+'.cu'),manifest_sha256=sha(str(so)+'.plan.json'),
                    execution=dict(pg='l2'),trace=False,
                    kernels={'tilemega_l1_kernel':dict(spill=False)})
        result['artifact_id']=digest(result)
        Path(str(so)+'.identity.json').write_text(json.dumps(result))
        return so,result

    def test_dm_identity_is_verified_and_bound_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            so,expected=self.fixture(Path(directory))
            identity=optional_identity(so)
            self.assertEqual(identity,expected)
            bound=bind_execution(identity,'L1')
            self.assertEqual(bound['artifact_id'],expected['artifact_id'])
            self.assertEqual(bound['kernel'],'tilemega_l1_kernel')
            self.assertFalse(bound['spill'])
            self.assertNotIn('plan',identity)  # Never rewrite the hashed identity.

    def test_dm_artifact_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            so,_=self.fixture(Path(directory));so.write_bytes(b'changed')
            with self.assertRaises(ValueError):optional_identity(so)

    def test_dm_snapshot_cli_and_serving_api_are_independent(self):
        from tilemega.build import dm_identity
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'snapshot.json'
            argv=['dm_identity','--root',directory,'--snapshot',str(destination),
                  '--capture','--compiler','/compiler']
            with patch('sys.argv',argv),patch.object(dm_identity,'source_snapshot',return_value={'head':'fixture'}) as capture:
                dm_identity.main()
            self.assertEqual(json.loads(destination.read_text()),{'head':'fixture'})
            capture.assert_called_once_with(Path(directory),Path('/compiler'))

if __name__=='__main__':unittest.main()
