"""Synthetic CPU traces verify ledger integrity and DT-A reporting only."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dnn1_marks', ROOT / 'scripts/dnn1_marks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
NAMES = ['slot_wait', 'main_loop', 'partial_write', 'splitk_reduce', 'epilogue',
         'halo_wait', 'depthwise_compute', 'fused_depthwise', 'fused_pointwise',
         'kv_wait', 'attention_qk', 'attention_softmax', 'attention_pv',
         'small_load', 'small_compute', 'small_store']


def fixture(path, scale=1, overlap=False):
    path.mkdir()
    record = struct.Struct('<' + 'Q' * (4 + 4 * len(NAMES)))
    raw = bytearray()
    times = [[(100, 200, 240), (105, 220, 250)], [(245, 300, 340), (260, 320, 350)]]
    if overlap:
        times[1][0] = (219, 300, 340)
    for stage, group in enumerate(times):
        for cta, (enter, arrive, leave) in enumerate(group):
            fields = [enter * scale, arrive * scale, leave * scale, 0]
            for name in NAMES:
                duration = (70 + cta * 10) if stage == 0 else 20
                fields += ([duration * scale, 1, (enter + 1) * scale,
                            (enter + 1 + duration) * scale] if name == 'epilogue' else [0] * 4)
            raw += record.pack(*fields)
    info = dict(schema=1, launches=1, timing_eligible=False, stages=2, grid=2,
                segments=len(NAMES), record_bytes=record.size, bytes=len(raw),
                segment_names=NAMES, raw_sha256=hashlib.sha256(raw).hexdigest())
    (path / 'trace.bin').write_bytes(raw)
    (path / 'trace.json').write_text(json.dumps(info))


class Ledger(unittest.TestCase):
    def test_five_process_medians_preserve_fields_and_stop_on_rc7(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = [root / f'p{i}' for i in range(5)]
            for path, scale in zip(traces, [1, 2, 3, 4, 100]):
                fixture(path, scale)
            work = [dict(index=i, body='gemm', representative=f'cpu_fixture_{i}', dram_bytes=300,
                flops=100, tasks=2, slots=2, geometry_source='SYNTHETIC_CPU_FIXTURE', k_iter=4,
                weight_bytes_per_task=100, m=49, tm=64, n=32, k=256, pixels=0, channels=0,
                covered_batches=1, runtime_batch=1) for i in range(2)]
            (root / 'work.json').write_text(json.dumps(dict(runtime_batch=1, stages=work)))
            (root / 'target.json').write_text(json.dumps(dict(sm90_native_constants=dict(
                dram_gbps=100, cta_dram_gbps=100))))
            argv = ['dnn1_marks.py', '--work', str(root / 'work.json'), '--target', str(root / 'target.json'),
                    '--model', 'CPU_FIXTURE', '--batch', '1', '--peak-tflops', '1',
                    '--peak-source', 'SYNTHETIC_CPU_FIXTURE', '--out', str(root / 'out')]
            for path in traces:
                argv += ['--trace', str(path)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 2)
            summary = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(summary['dt_a'], 'STOP_RC7')
            self.assertEqual(summary['stage_time_ns'], 750)
            self.assertEqual(summary['category_shares']['RC-7'], 0.4)
            self.assertEqual(summary['low_m_row_utilization_time_share'], 1)
            with (root / 'out/stage_ledger.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['category'] for r in rows], ['RC-3', 'RC-7'])
            self.assertEqual(rows[0]['stage_ns'], '450')
            self.assertEqual(rows[0]['epilogue_ns'], '240')
            self.assertEqual(rows[0]['critical_ctas'], '1;1;1;1;1')
            self.assertEqual(rows[0]['n'], '32')
            self.assertEqual(rows[0]['k'], '256')
            self.assertEqual(rows[0]['processes'], '5')

    def test_raw_hash_and_actual_barrier_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / 'changed')
            p = root / 'changed/trace.bin'
            p.write_bytes(p.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'trace bytes differ'):
                module.read_trace(root / 'changed')
            fixture(root / 'overlap', overlap=True)
            work = [dict(index=i, body='gemm', dram_bytes=300, flops=100, tasks=2, slots=2,
                geometry_source='SYNTHETIC_CPU_FIXTURE', k_iter=4, weight_bytes_per_task=100,
                m=49, tm=64, covered_batches=1, runtime_batch=1) for i in range(2)]
            with self.assertRaisesRegex(ValueError, 'overlap across the grid barrier'):
                module.analyze(root / 'overlap', work,
                    dict(sm90_native_constants=dict(dram_gbps=100, cta_dram_gbps=100)), 1, 0)


if __name__ == '__main__':
    unittest.main()
