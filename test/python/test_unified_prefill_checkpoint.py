import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('unified_checkpoint', ROOT / 'scripts/check_unified_serving.py')
unified = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified)


class PrefillCheckpointTest(unittest.TestCase):
    def test_preparation_uses_each_r14_prefill_not_old_decode_fixture_prefill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = dict(root=str(root), numeric_cases={}, cases=[], trees={})
            registry = {}
            for cell in ('llama_B1', 'llama_B16', 'qwen3_B1', 'qwen3_B16'):
                batch = int(cell.split('_B')[1])
                source = root / cell
                source.mkdir()
                so = source / 'plan.so'
                so.write_bytes(cell.encode())
                Path(str(so)+'.plan.json').write_text(json.dumps(dict(phase='prefill', seq=64, pg='l2', watchdog=0)))
                Path(str(so)+'.build_command.txt').write_text('nvcc -Iinclude source.cu -o plan.so')
                (source / 'selected.mlir').write_text('module {}')
                (source / 'target.json').write_text('{}')
                registry[cell] = [dict(label='R13D_prime', prefill=str(so))]
                record['numeric_cases'][cell] = dict(batch=batch, prefill='/old/prefill.so', files={})
                record['cases'].append(dict(cell=cell, label='R13D_prime', input=str(source)))
            path = root / 'docs' / 'experiments' / 'SERVING_R14/phase_a_arms.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(registry))
            original = path.read_bytes()
            out = root / 'runs/r15/prefill'
            with patch.object(unified, 'validate', return_value=record), patch.object(
                    unified, 'git', return_value='frozen-head'), contextlib.redirect_stdout(io.StringIO()):
                unified.prepare_prefill(root / 'decode', out, 10800)
            result = json.loads((out / 'inputs.json').read_text())
            self.assertEqual(len(result['cases']), 4)
            self.assertEqual(result['nvcc_timeout_s'], 10800)
            for case in result['cases']:
                self.assertEqual(case['phase'], 'prefill')
                self.assertEqual(result['numeric_cases'][case['cell']]['prefill'],
                                 registry[case['cell']][0]['prefill'])
                for filename, expected in case['files'].items():
                    self.assertEqual(unified.sha(filename), expected)
            self.assertEqual(path.read_bytes(), original)

    def test_numerical_pair_uses_same_variant_and_rejects_changed_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            decode = root / 'decode.so'
            decode.write_bytes(b'reference')
            plans = [dict(cell='llama_B1', label='R13D_prime', variant='reference',
                          so=str(decode), so_sha256=unified.sha(decode))]
            (root / 'build.json').write_text(json.dumps(dict(plans=plans)))
            record = dict(decode_checkpoint=str(root))
            row = dict(cell='llama_B1', variant='reference', so='/new/reference_prefill.so')
            case = dict(phase='prefill')
            self.assertEqual(unified.execution_pair(record, case, row), (str(decode), row['so']))
            with self.assertRaisesRegex(ValueError, 'matching decode artifact'):
                unified.execution_pair(record, case, dict(row, variant='candidate'))
            decode.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'matching decode artifact'):
                unified.execution_pair(record, case, row)
            self.assertEqual(unified.execution_pair({}, {}, row), (row['so'], None))

    def test_incomplete_cell_set_is_rejected(self):
        with patch.object(unified, 'validate', return_value=dict(numeric_cases={})), self.assertRaisesRegex(
                ValueError, 'all four default cells'):
            unified.prepare_prefill(Path('/previous'), Path('/unused'), 10800)

    def test_reuse_preserves_inputs_and_freezes_explicit_compiler_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'previous'
            original.mkdir()
            marker = original / 'inputs.json'
            marker.write_text('frozen original')
            record = dict(root=str(root), cases=[dict(label='unchanged')])
            out = root / 'retry'
            with patch.object(unified, 'validate', return_value=record), patch.object(
                    unified, 'git', return_value='runner-head'), contextlib.redirect_stdout(io.StringIO()):
                unified.reuse_inputs(original, out, 10800)
            frozen = json.loads((out / 'inputs.json').read_text())
            self.assertEqual(frozen['cases'], [dict(label='unchanged')])
            self.assertEqual(frozen['nvcc_timeout_s'], 10800)
            self.assertEqual(frozen['reuses_inputs'], str(original))
            self.assertEqual(marker.read_text(), 'frozen original')


if __name__ == '__main__':
    unittest.main()
