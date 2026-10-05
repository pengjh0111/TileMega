import json,tempfile,unittest
from pathlib import Path
from anchor import loop_required,replay_vllm_record
from ptx_pdl_check import late_trigger_has_publication
from compare_kernels import resources
from analyze import trace_cell

class ReviewTests(unittest.TestCase):
    def test_batch_boundary_in_trace_paths(self):
        self.assertEqual(trace_cell('/raw/B1_qwen3_B16/P/trace'), 'qwen3_B16')
        self.assertEqual(trace_cell('/raw/B1_llama_B1/P/trace'), 'llama_B1')
    def test_loop_only_for_explicit_tilemega(self):
        self.assertFalse(loop_required(dict(kind="vllm")))
        self.assertFalse(loop_required(dict(kind="tm_old",decode_loop=True)))
        self.assertFalse(loop_required(dict(kind="tm",decode_loop="auto")))
        self.assertTrue(loop_required(dict(kind="tm",decode_loop=1)))
    def test_ret_excludes_cold_blocks(self):
        body="griddepcontrol.launch_dependents;\nret;\n$COLD:\nst.global.u64 [%rd1],%rd2;\n"
        self.assertFalse(late_trigger_has_publication(body,0))
    def test_reachable_publication_rejected(self):
        for branch in ["bra $COLD;", "@%p bra $COLD;"]:
            body="griddepcontrol.launch_dependents;\n"+branch+"\nret;\n$COLD:\nst.global.u64 [%rd1],%rd2;\nret;\n"
            self.assertTrue(late_trigger_has_publication(body,0))
    def test_resource_call_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"r.log"
            f.write_text("Compiling entry function 'old'\nFunction properties for helper\n0 bytes stack frame, 0 bytes spill stores\nCompiling entry function 'new'\nFunction properties for helper\n0 bytes stack frame, 32 bytes spill stores\n")
            r=resources(f)
            self.assertNotEqual(r["old::helper"],r["new::helper"])
    def test_vllm_replay_is_evidence_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            step=Path(tmp);out=step/"cell"/"vllm"/"round0";(out/"B1").mkdir(parents=True)
            metrics=dict(batch=1,e2e_seconds=3.0,ttft_seconds=.01,vllm_version="0.30.0")
            (out/"command.json").write_text(json.dumps(dict(command=["python","/repo/vllm_baseline.py"])))
            (out/"B1/measurements.json").write_text(json.dumps(metrics))
            (step/"guard_result.json").write_text(json.dumps(dict(code=0)))
            record=dict(metrics,out=str(out),exit_code=3,error="requested loop was not used")
            self.assertEqual(replay_vllm_record(record)["exit_code"],0)
            (step/"guard_result.json").write_text(json.dumps(dict(code=75)))
            with self.assertRaises(ValueError):replay_vllm_record(record)
            self.assertEqual(replay_vllm_record(dict(record,exit_code=1))["exit_code"],1)
if __name__=="__main__":unittest.main()
