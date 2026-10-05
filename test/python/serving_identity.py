import copy,sys,unittest,json,tempfile
from unittest.mock import patch,Mock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/experiments/SERVING_R14'))
from tilemega.build.identity import parse_resources,bind_execution,digest,generate,verify,sha
from identity_join import checked_join
class IdentityTests(unittest.TestCase):
    def identity(self):
        text="""ptxas info : Compiling entry function 'tilemega_l1_kernel' for 'sm_89'
ptxas info : Function properties for tilemega_l1_kernel
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 228 registers, 0 bytes smem
ptxas info : Function properties for helper
16 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads
ptxas info : Compiling entry function 'tilemega_l2_kernel' for 'sm_89'
ptxas info : Function properties for tilemega_l2_kernel
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 176 registers, 0 bytes smem
ptxas info : Function properties for helper
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""
        return {'artifact_id':'binary-a','plan':{'pg':'pages'},'resources':parse_resources(text),'trace':False}
    def test_generation_binds_artifacts_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);so=root/'plan.so';source=root/'runtime.h'
            source.write_text('input');so.write_bytes(b'compiled artifact')
            Path(str(so)+'.cu').write_text('#define TILEMEGA_WATCHDOG 0\n')
            Path(str(so)+'.ptxas.log').write_text("ptxas info : Compiling entry function 'tilemega_l1_kernel' for 'sm_89'\nptxas info : Used 228 registers\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n")
            Path(str(so)+'.build_command.txt').write_text('nvcc -arch=sm_89 -DTILEMEGA_TRACE_STAGE=0 plan.so.cu -o plan.so')
            Path(str(so)+'.plan.json').write_text(json.dumps({'pg':'pages','phase':'decode','gemms':[{'index':0}]}))
            snap=root/'source.json';snap.write_text(json.dumps({'head':'abc','source_files':{'runtime.h':sha(source)}}))
            library=Mock();library.tm_plan_shared_bytes.return_value=99328
            with patch('tilemega.build.identity.ctypes.CDLL',return_value=library), patch('tilemega.build.identity.subprocess.check_output',return_value='nvcc 12.8'):
                identity=generate(so,root,snap)
                self.assertEqual(verify(so)['artifact_id'],identity['artifact_id'])
                self.assertFalse(identity['trace']);self.assertEqual(identity['shared_memory_bytes'],99328)
                self.assertEqual(identity['implementations']['attention']['page_policy'],'Packed')
                source.write_text('changed during build')
                with self.assertRaisesRegex(ValueError,'source changed'):generate(so,root,snap)
            so.write_bytes(b'different binary')
            with self.assertRaisesRegex(ValueError,'artifact identity mismatch'):verify(so)

    def test_spilled_callee_belongs_to_its_entry(self):
        i=self.identity();self.assertTrue(bind_execution(i,'L1')['spill']);self.assertFalse(bind_execution(i,'L2')['spill'])
    def test_execution_is_part_of_join_key(self):
        i=self.identity();a={'execution_identity':bind_execution(i,'L1')};b={'execution_identity':bind_execution(i,'L2')}
        self.assertEqual(checked_join(a,copy.deepcopy(a))['left'],a)
        with self.assertRaises(ValueError):checked_join(a,b)
        i['artifact_id']='binary-b';b={'execution_identity':bind_execution(i,'L1')}
        with self.assertRaises(ValueError):checked_join(a,b)
    def test_trace_is_not_e2e(self):
        i=self.identity();i['trace']=True;a={'execution_identity':bind_execution(i,'L1')}
        with self.assertRaises(ValueError):checked_join(a,a)
        checked_join(a,a,trace_table=True)
    def test_mutated_metadata_and_unavailable_kernel_rejected(self):
        i=self.identity();a={'execution_identity':bind_execution(i,'L1')};a['execution_identity']['spill']=False
        with self.assertRaises(ValueError):checked_join(a,a)
        with self.assertRaises(ValueError):bind_execution(i,'L1',True)
        with self.assertRaises(ValueError):bind_execution(i,'L2',True)
if __name__=='__main__':unittest.main()
