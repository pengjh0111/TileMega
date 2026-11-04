"""I-1 range boundaries and kernel-derived windows, independent of GPU timing."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import sm90r6_engine_trace as trace
spec = importlib.util.spec_from_file_location('i1_hook', ROOT/'scripts/sm90r6_trace_hooks/sitecustomize.py')
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class TraceProtocol(unittest.TestCase):
    def test_exact_profiler_range(self):
        calls = []
        api = SimpleNamespace(synchronize=lambda: calls.append('sync'), cudart=lambda:
            SimpleNamespace(cudaProfilerStart=lambda: calls.append('start') or 0,
                            cudaProfilerStop=lambda: calls.append('stop') or 0))
        with tempfile.TemporaryDirectory() as directory:
            capture = hook.Capture('vllm', Path(directory)/'capture', api)
            for _ in range(30):capture.before(True)
            capture.before(False)
            for _ in range(256):capture.before(True)
            self.assertEqual(calls, [])
            capture.before(True)
            self.assertEqual(calls, ['sync', 'start'])
            for _ in range(63):capture.before(True)
            self.assertTrue(capture.active)
            capture.before(True)
            self.assertEqual(calls, ['sync', 'start', 'sync', 'stop'])
            trace.capture_complete(Path(directory))

    def test_variable_duration_steps(self):
        rows = []; start = 0
        for i in range(64):
            rows.extend([('embedding', start, start+2), ('gemm', start+1, start+5),
                         ('argmax', start+6, start+8)])
            start += 10+i
        windows = trace.steps(rows)
        self.assertEqual(len(windows), 32)
        self.assertEqual(windows[0][1]-windows[0][0], 3)
        self.assertEqual(windows[0][3]-windows[0][2], 26)
        self.assertEqual(windows[-1][3]-windows[-1][2], 57)
        result = trace.metrics(rows, windows, 4, 2)
        self.assertEqual(result['busy_us'], .007)  # overlap [0,2] with [1,5]
        self.assertEqual(result['step_us'], .0415)
        rows[20] = ('unexpected', *rows[20][1:])
        with self.assertRaises(ValueError):trace.steps(rows)

    def test_decode_filters(self):
        self.assertTrue(hook.is_decode('vllm', None,
            [SimpleNamespace(num_scheduled_tokens={1: 1, 2: 1}, total_num_scheduled_tokens=2)], {}))
        self.assertFalse(hook.is_decode('vllm', None,
            [SimpleNamespace(num_scheduled_tokens={1: 64}, total_num_scheduled_tokens=64)], {}))
        self.assertFalse(hook.is_decode('trtllm', SimpleNamespace(in_warmup=True),
            [SimpleNamespace(context_requests=[], generation_requests=[1])], {}))

    def test_split_k_operator_mapping(self):
        rows = [('nvjet_regular', 0, 2), ('norm_kernel', 2, 3),
                ('nvjet_regular', 3, 5), ('nvjet_regular', 5, 7),
                ('nvjet_splitK_TNT', 7, 9), ('cublasLt::splitKreduce_kernel', 10, 11),
                ('nvjet_head', 11, 13)]
        config = dict(num_hidden_layers=1, hidden_size=2, num_attention_heads=1,
                      intermediate_size=4, vocab_size=8)
        metrics = trace.gemm_metrics(rows, [(0, len(rows), 0, 14)], config, 'test', 'vllm', 1)
        down = next(row for row in metrics if row['family'] == 'down')
        self.assertAlmostEqual(down['median_gbps'], 16/3)
        self.assertEqual(len(trace.gemm_groups(rows)), 5)
        with self.assertRaises(ValueError):trace.gemm_groups(rows[5:])
        self.assertEqual(trace.classify('reshape_and_cache_flash_kernel'), 'rope_kv')
        self.assertEqual(trace.classify('unrolled_elementwise_kernel<FillFunctor>'), 'other')


if __name__ == '__main__':
    unittest.main()
