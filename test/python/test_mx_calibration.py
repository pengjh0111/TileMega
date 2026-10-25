"""Resource references must select the declared unprefetched implementation."""
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(__file__).resolve().parents[2] / 'experiments/mx5/calibration.py'
spec = importlib.util.spec_from_file_location('mx5_calibration', source)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class Calibration(unittest.TestCase):
    def test_catalogue_uses_recorded_coordinates_and_zero_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {
                'different_name': dict(body='gemm', bench=True, threads=512, tn=64, depth=3, pw=0),
                'attention64': dict(body='attention', bench=True, threads=576, dim=64, query=4, prefetch_kind=0),
                'prefetched': dict(body='head', bench=True, threads=512, tn=16, depth=1, pw=1),
                'unit': dict(body='gemm', bench=False, threads=512, tn=16, depth=1, pw=0),
            }
            for name, metadata in records.items():
                (root / (name + '.json')).write_text(json.dumps(metadata))
            result = calibration.baseline_libraries(root)
            self.assertEqual(result, {
                ('gemm', 512, 64, 3): root / 'different_name.so',
                ('attention', 576, 64, 4): root / 'attention64.so',
            })

    def test_duplicate_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(body='head', bench=True, threads=512, tn=128, depth=3, pw=0)
            for name in ('one', 'two'):
                (root / (name + '.json')).write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                calibration.baseline_libraries(root)

    def test_resource_inspection_requires_guard_before_cuda_import(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'mx_gpu_job.sh'):
                next(calibration.inspect([], Path('/unused')))

    def test_library_identity_rejects_wrong_body_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.so'
            path.write_bytes(b'not loaded')
            path.with_suffix('.json').write_text(json.dumps(dict(body='head', bench=True, returncode=0)))
            with patch.object(calibration.C, 'CDLL') as loader:
                with self.assertRaisesRegex(RuntimeError, 'baseline identity mismatch'):
                    calibration.load_library(path, 'gemm', dict(threads=512), object())
                loader.assert_not_called()


class NativeFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class DerivedCalibration(unittest.TestCase):
    def artifact(self, root, kind='gemm', status=0, resources=(180, 0, 0, 1)):
        key = dict(threads=576, b=2)
        if kind == 'attention':
            key.update(dim=128, q=4, hkv=2, capacity=1088, length=256,
                       b_min=2, qk_norm=1, prefetch_kind=2)
            fields = dict(dim=128, query=4, prefetch_kind=2)
            macros = dict(MX5_TEST_DIM=128, MX5_TEST_Q=4, MX5_BENCH_PREFETCH_KIND=2)
            coordinates = [128, 4, 2, 576]
        else:
            key.update(n=256, k=512, tn=32, wn=4, depth=2, prefetch=4)
            if kind == 'gemm':
                key.update(op=0, norm=1, part=1)
            fields = dict(tn=32, depth=2, pw=4)
            macros = dict(MX5_TEST_TN=32, MX5_TEST_DEPTH=2, MX5_BENCH_PW=4)
            coordinates = [32, 2, 4, 576]
        bench = calibration.module(kind)
        path = root / (kind + '.so')
        path.write_bytes(b'fake library')
        filename = 'stage_bench.cu' if kind == 'gemm' else kind + '_stage_bench.cu'
        source_name = 'experiments/mx5/' + filename
        source_data = b'// immutable fixture\n'
        with tarfile.open(root / 'sources.tar.gz', 'w:gz') as archive:
            entry = tarfile.TarInfo(source_name)
            entry.size = len(source_data)
            archive.addfile(entry, io.BytesIO(source_data))
        metadata = dict(body=kind, bench=True, returncode=0, threads=576, **fields,
                        binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(source_name.encode() + b'\0' + source_data + b'\0').hexdigest(),
                        base_head='fixture', command=['cucc', '-DMX5_TEST_THREADS=576',
                        *[f'-D{name}={value}' for name, value in macros.items()], source_name])
        path.with_suffix('.json').write_text(json.dumps(metadata))
        prefix = 'mx5_bench' if kind == 'gemm' else 'mx5_' + kind + '_bench'
        calls = []

        def query(argument, *runtime):
            args = argument._obj
            self.assertEqual(args.iterations, 0)
            self.assertEqual(args.copies, 1)
            self.assertEqual(args.b, key['b'])
            self.assertIsNone(args.counter)
            self.assertIsNone(args.stamps)
            if kind == 'gemm':
                self.assertEqual((args.n, args.k), (key['n'], key['k']))
                self.assertEqual(runtime, (key['wn'], key['op'], key['norm'], key['part'], None))
            elif kind == 'head':
                self.assertEqual((args.n, args.k, args.head.capacity, args.head.past),
                                 (key['n'], key['k'], 1088, 64))
                self.assertEqual(runtime, (key['wn'], None))
            else:
                self.assertEqual((args.hkv, args.attention.capacity, args.attention.past),
                                 (key['hkv'], 1088, 255))
                self.assertEqual(runtime, (key['b_min'], key['qk_norm'], None))
            for index, value in enumerate(resources):
                args.resources[index] = value
            calls.append(runtime)
            return status

        def export(values):
            for index, value in enumerate(coordinates):
                values[index] = value

        library = SimpleNamespace(**{prefix: NativeFunction(query),
            prefix + '_args_size': NativeFunction(lambda: calibration.C.sizeof(bench.Args)),
            prefix + '_coordinates': NativeFunction(export)})
        row = dict(kind=kind, key=key, uses=[dict(model='fixture', stage='qkv')],
                   disposition='derived', library='/baseline.so', fits=True,
                   r_next_reference=dict(resources=dict(mtreg=150)),
                   artifacts=[dict(resource_step=0, prefetch=calibration.prefetch(kind, key),
                                   library=str(path)),
                              dict(resource_step=1, prefetch=2, library='/never/fallback.so')])
        return row, path, bench, library, calls

    def test_nonzero_prefetch_all_bodies_check_exact_runtime_without_launch(self):
        for kind in ('gemm', 'head', 'attention'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                row, path, bench, library, calls = self.artifact(Path(directory), kind)
                skipped = dict(disposition='unprefetched_resource_failure', fits=False, untouched=True)
                with patch.dict(os.environ, TILEMEGA_GPU_LOCK_HELD='1'), \
                     patch.object(calibration, 'initialize_cuda'), \
                     patch.object(calibration.C, 'CDLL', return_value=library) as loader:
                    checked = list(calibration.check([skipped, row, row]))
                self.assertIs(checked[0], skipped)
                self.assertEqual(len(calls), 2)
                loader.assert_called_once_with(str(path))
                result = checked[1]
                self.assertEqual(result['selected_resource_step'], 0)
                self.assertEqual(result['library'], str(path))
                self.assertEqual(result['query_status'], 0)
                self.assertEqual(result['resources'], dict(mtreg=180, local_bytes=0,
                                                          static_smem_bytes=0, resident_ctas=1))
                self.assertTrue(result['fits'])
                self.assertEqual(result['r_next_reference'], row['r_next_reference'])
                self.assertEqual(result['uses'], row['uses'])
                self.assertEqual(row['library'], '/baseline.so')

    def test_wrong_prefetch_rejected_before_loading(self):
        for kind in ('gemm', 'head', 'attention'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                row, path, bench, _, _ = self.artifact(Path(directory), kind)
                key = dict(row['key'])
                key['prefetch_kind' if kind == 'attention' else 'prefetch'] = 1
                with patch.object(calibration.C, 'CDLL') as loader:
                    with self.assertRaisesRegex(RuntimeError, 'coordinates mismatch'):
                        calibration.load_library(path, kind, key, bench)
                    loader.assert_not_called()

    def test_cached_library_still_checks_each_keys_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            row, path, _, library, calls = self.artifact(Path(directory), 'attention')
            wrong = dict(row, key=dict(row['key'], q=8))
            with patch.dict(os.environ, TILEMEGA_GPU_LOCK_HELD='1'), \
                 patch.object(calibration, 'initialize_cuda'), \
                 patch.object(calibration.C, 'CDLL', return_value=library):
                checked = calibration.check([row, wrong])
                self.assertTrue(next(checked)['fits'])
                with self.assertRaisesRegex(RuntimeError, 'coordinates mismatch'):
                    next(checked)
            self.assertEqual(len(calls), 1)

    def test_failure_resource_values_are_null_and_stack_overflow_is_visible(self):
        for status, resources in [(7, (0, 0, 0, 0)), (0, (180, 64, 0, 1)),
                                  (0, (228, 0, 0, 1))]:
            with self.subTest(status=status, resources=resources), tempfile.TemporaryDirectory() as directory:
                row, _, _, library, _ = self.artifact(Path(directory), status=status, resources=resources)
                with patch.dict(os.environ, TILEMEGA_GPU_LOCK_HELD='1'), \
                     patch.object(calibration, 'initialize_cuda'), \
                     patch.object(calibration.C, 'CDLL', return_value=library):
                    result = next(calibration.check([row]))
                self.assertFalse(result['fits'])
                self.assertEqual(result['query_status'], status)
                self.assertEqual(result['resources']['local_bytes'], None if status else resources[1])
                self.assertEqual(result['resources']['mtreg'], None if status else resources[0])

    def test_check_and_query_require_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'mx_gpu_job.sh'):
                next(calibration.check([]))
            with self.assertRaisesRegex(RuntimeError, 'mx_gpu_job.sh'):
                calibration.query_one('gemm', {}, '/unused', {}, {})

    def test_missing_primary_library_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            row, path, _, _, _ = self.artifact(Path(directory))
            path.unlink()
            with patch.dict(os.environ, TILEMEGA_GPU_LOCK_HELD='1'), \
                 patch.object(calibration, 'initialize_cuda'), \
                 patch.object(calibration.C, 'CDLL') as loader:
                with self.assertRaisesRegex(RuntimeError, 'missing bench library'):
                    next(calibration.check([row]))
                loader.assert_not_called()

    def test_source_and_command_mismatches_rejected_before_loading(self):
        for mismatch in ('source', 'command'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                row, path, bench, _, _ = self.artifact(Path(directory), 'head')
                metadata = json.loads(path.with_suffix('.json').read_text())
                if mismatch == 'source':
                    metadata['source_sha256'] = 'bad'
                else:
                    metadata['command'] = [value.replace('MX5_BENCH_PW=4', 'MX5_BENCH_PW=2')
                                           for value in metadata['command']]
                path.with_suffix('.json').write_text(json.dumps(metadata))
                with patch.object(calibration.C, 'CDLL') as loader:
                    with self.assertRaisesRegex(RuntimeError, 'source identity mismatch|command mismatch'):
                        calibration.load_library(path, 'head', row['key'], bench)
                    loader.assert_not_called()

    def test_exported_prefetch_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row, path, bench, library, _ = self.artifact(Path(directory), 'head')
            def wrong_coordinates(values):
                for index, value in enumerate((32, 2, 0, 576)):
                    values[index] = value
            library.mx5_head_bench_coordinates = NativeFunction(wrong_coordinates)
            with patch.object(calibration.C, 'CDLL', return_value=library):
                with self.assertRaisesRegex(RuntimeError, 'exported coordinates mismatch'):
                    calibration.load_library(path, 'head', row['key'], bench)

    def test_main_snapshots_source_and_records_failed_runs(self):
        for guarded in (False, True):
            with self.subTest(guarded=guarded), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                keys = root / 'keys.jsonl'
                keys.write_text('')
                out = root / 'nested' / 'checked.jsonl'
                argv = ['calibration', 'check', '--keys', str(keys), '--out', str(out)]
                environment = {'TILEMEGA_GPU_LOCK_HELD': '1'} if guarded else {}
                with patch('sys.argv', argv), patch.dict(os.environ, environment, clear=True):
                    if guarded:
                        with patch.object(calibration, 'initialize_cuda'):
                            calibration.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'mx_gpu_job.sh'):
                            calibration.main()
                snapshot = Path(str(out) + '.tool.py')
                self.assertEqual(snapshot.read_bytes(), source.read_bytes())
                identity = json.loads(Path(str(out) + '.identity.json').read_text())
                self.assertEqual(identity['tool_source'], str(snapshot))
                self.assertEqual(identity['tool_sha256'], hashlib.sha256(snapshot.read_bytes()).hexdigest())
                self.assertEqual(identity['run_status'], 'complete' if guarded else 'failed')

    def test_main_preflights_all_sidecars_before_work(self):
        for suffix in ('', '.tool.py', '.identity.json', '.builds.json', '.references.json'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'plan.jsonl'
                existing = Path(str(out) + suffix)
                existing.write_text('historical')
                argv = ['calibration', 'plan', '--keys', '/unused', '--domain', '/unused',
                        '--library-root', '/unused', '--out', str(out)]
                with patch('sys.argv', argv), patch.object(calibration, 'plan') as derive, \
                     patch('sys.stderr', new=io.StringIO()):
                    with self.assertRaises(SystemExit):
                        calibration.main()
                    derive.assert_not_called()
                self.assertEqual(existing.read_text(), 'historical')
                self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == '__main__': unittest.main()
