"""Stage keys must describe the hashed binary while N/K/B stay runtime inputs."""
import copy
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[2] / 'experiments/mx5/stage_bench.py'
spec = importlib.util.spec_from_file_location('mx5_stage_bench', source)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class StageBenchMetadata(unittest.TestCase):
    def setUp(self):
        self.key = dict(n=1024, k=1024, b=1, tn=16, wn=1, op=0, norm=1, part=0)
        self.meta = dict(body='gemm', bench=True, returncode=0, tn=16, depth=1,
                         threads=576, pw=None, command=['cucc', '-DMX5_TEST_THREADS=576',
                                                     '-DMX5_TEST_TN=16', '-DMX5_TEST_DEPTH=1'])

    def test_legacy_keys_and_runtime_shapes(self):
        for n, k, b in [(1024, 1024, 1), (2048, 8192, 16)]:
            with self.subTest(n=n, k=k, b=b):
                actual = bench.validate_metadata(dict(self.key, n=n, k=k, b=b), self.meta)
                self.assertEqual(actual, dict(tn=16, depth=1, threads=576, pw=1))

    def test_explicit_coordinates(self):
        coordinates = dict(tn=16, depth=1, threads=576, pw=1)
        self.assertEqual(bench.validate_metadata(dict(self.key, **coordinates), self.meta), coordinates)
        for name, wrong in dict(tn=32, depth=2, threads=512, pw=2).items():
            with self.subTest(name=name):
                key = dict(self.key, **coordinates)
                key[name] = wrong
                with self.assertRaisesRegex(RuntimeError, f'key/build mismatch for {name}'):
                    bench.validate_metadata(key, self.meta)

    def test_explicit_prefetch_and_default_depth(self):
        self.meta['pw'] = 8
        self.meta['command'].append('-DMX5_BENCH_PW=8')
        self.assertEqual(bench.validate_metadata(dict(self.key, pw=8), self.meta)['pw'], 8)
        with self.assertRaisesRegex(RuntimeError, 'key/build mismatch for pw'):
            bench.validate_metadata(dict(self.key, pw=1), self.meta)

    def test_metadata_must_match_build_command(self):
        for name, value in dict(tn=32, depth=2, threads=512, pw=8).items():
            with self.subTest(name=name):
                meta = dict(self.meta, **{name: value})
                with self.assertRaisesRegex(RuntimeError, f'metadata/command mismatch for {name}'):
                    bench.validate_metadata(self.key, meta)
        self.meta['command'].append('-DMX5_BENCH_PW=8')
        with self.assertRaisesRegex(RuntimeError, 'metadata/command mismatch for pw'):
            bench.validate_metadata(self.key, self.meta)

    def test_reject_duplicate_and_missing_definitions(self):
        for suffix in [['-DMX5_TEST_TN=16'], ['-D', 'MX5_TEST_TN=32']]:
            with self.subTest(suffix=suffix):
                meta = copy.deepcopy(self.meta)
                meta['command'].extend(suffix)
                with self.assertRaisesRegex(RuntimeError, 'duplicate build definition'):
                    bench.validate_metadata(self.key, meta)
        self.meta['command'].pop()
        with self.assertRaisesRegex(RuntimeError, 'metadata/command mismatch for depth'):
            bench.validate_metadata(self.key, self.meta)

    def test_split_definition_arguments(self):
        self.meta['command'] = ['cucc', '-D', 'MX5_TEST_THREADS=576', '-D', 'MX5_TEST_TN=16',
                                '-D', 'MX5_TEST_DEPTH=1']
        self.assertEqual(bench.validate_metadata(self.key, self.meta)['tn'], 16)

    def test_reject_noninteger_coordinates_and_wrong_build_kind(self):
        for name, value in [('tn', 16.5), ('depth', True), ('threads', '576'), ('pw', None)]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, f'key {name} must be an integer'):
                    bench.validate_metadata(dict(self.key, **{name: value}), self.meta)
        for changes in [dict(bench=False), dict(body='head'), dict(returncode=1)]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(RuntimeError, 'successful GEMM stage-bench build'):
                    bench.validate_metadata(self.key, dict(self.meta, **changes))

    def load_fixture(self, binary_hash=None, abi_size=None):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / 'bench.so'
            library.write_bytes(b'CPU-only metadata fixture')
            meta = dict(self.meta, binary_sha256=binary_hash or hashlib.sha256(library.read_bytes()).hexdigest())
            library.with_suffix('.json').write_text(json.dumps(meta))
            fake = Mock()
            fake.mx5_bench_args_size.return_value = C.sizeof(bench.Args) if abi_size is None else abi_size
            with patch.object(bench.C, 'CDLL', return_value=fake) as loader:
                result = bench.load_bench(dict(self.key, library=str(library)))
                loader.assert_called_once_with(str(library))
            return result

    def test_binary_identity_and_abi_rejection(self):
        with self.assertRaisesRegex(RuntimeError, 'binary identity mismatch'):
            self.load_fixture(binary_hash='wrong')
        with self.assertRaisesRegex(RuntimeError, 'argument ABI mismatch'):
            self.load_fixture(abi_size=C.sizeof(bench.Args) + 4)

    def test_valid_binary_uses_explicit_ctypes_abi(self):
        lib, meta, binary, compiled = self.load_fixture()
        self.assertEqual(meta['binary_sha256'], binary)
        self.assertEqual(compiled['pw'], 1)
        self.assertEqual(lib.mx5_bench_args_size.argtypes, [])
        self.assertEqual(lib.mx5_bench_args_size.restype, C.c_int)
        self.assertEqual(lib.mx5_bench.argtypes,
                         [C.POINTER(bench.Args), C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p])
        self.assertEqual(lib.mx5_bench.restype, C.c_int)


if __name__ == '__main__': unittest.main()
