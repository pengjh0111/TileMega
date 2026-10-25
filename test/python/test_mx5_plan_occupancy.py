"""Portable CPU-only mocks; these tests establish no real GPU occupancy."""
import ctypes as C
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tilemega import mx5_plan_occupancy as occupancy


ASSEMBLY = '''0000000000004370 <tm_plan_shared_bytes>:
    4370: b8 30 80 00 00        mov    $0x8030,%eax
    4375: c3                    ret

'''
NM = '0000000000004b80 W ' + occupancy.STUB + '\n0000000000004370 T tm_plan_shared_bytes\n'


class FakeRuntime:
    def __init__(self, attr_rc=0, set_rc=0, occupancy_rc=0, resident=1, threads=576):
        self.codes = attr_rc, set_rc, occupancy_rc
        self.resident, self.threads, self.calls = resident, threads, []

    def attributes(self, destination, address):
        self.calls.append('attributes')
        value = C.cast(destination, C.POINTER(occupancy.Attributes)).contents
        value.numRegs, value.localSizeBytes, value.sharedSizeBytes = 168, 0, 0
        value.maxThreadsPerBlock = self.threads
        return self.codes[0]

    def set_shared(self, address, attribute, shared):
        self.calls.append(('set_shared', attribute, shared))
        return self.codes[1]

    def occupancy(self, destination, address, threads, shared):
        self.calls.append(('occupancy', threads, shared))
        if not self.codes[2]:
            C.cast(destination, C.POINTER(C.c_int)).contents.value = self.resident
        return self.codes[2]


class FakeLibrary:
    def __init__(self, shared=32816):
        self.shared, self.calls = shared, []

    def shared_bytes(self):
        self.calls.append('shared_bytes')
        return self.shared

    def stub_address(self, name):
        self.calls.append(('stub_address_only', name))
        return C.c_void_p(0x1234)

    def __getattr__(self, name):
        raise AssertionError('unexpected library operation (creation/launch/stub calls forbidden): ' + name)


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = dict(kernel=dict(name=occupancy.KERNEL, registers=168, stack_bytes=0, static_smem_bytes=0),
                           kernel_instructions={occupancy.KERNEL: 1}, device_images=[{}])
        self.spec = dict(kind='mx5_plan_occupancy_inputs', schema_version=1, threads=576, context_kind='native_identity')
        values = dict(binary='CPU mock ELF bytes', generated_source='#include <tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh>\nnamespace tm_mx_plan { struct Plan { static constexpr int kGrid=104, kThreads=576; }; }',
                      resources=json.dumps(self.report), log='CPU mock MIR', source_snapshot='{}',
                      manifest=json.dumps(dict(executor='mx_static', threads=576, kThreads=576)))
        for name, text in values.items():
            self.write(name, text)
        context = dict(source={}, binary_sha256=self.spec['binary']['sha256'],
                       generated_source_sha256=self.spec['generated_source']['sha256'],
                       resources_sha256=self.spec['log']['sha256'], manifest_sha256=self.spec['manifest']['sha256'],
                       resources={occupancy.KERNEL: self.report['kernel']})
        self.write('context', json.dumps(context))

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        self.spec[name] = occupancy.record(path)

    def validate(self, nm=NM, assembly=ASSEMBLY):
        return occupancy.validate_inputs(self.spec, inspect_resource=lambda *_: copy.deepcopy(self.report),
                                         command=lambda argv: nm if argv[0] == 'nm' else assembly)

    def test_hash_and_actual_resource_binding(self):
        self.assertEqual(self.validate()['stub'], occupancy.STUB)
        (self.root / 'binary').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'bound input changed'):
            self.validate()

    def test_report_must_be_recomputed_and_unique_dense(self):
        with self.assertRaisesRegex(ValueError, 'resource report'):
            occupancy.validate_inputs(self.spec, inspect_resource=lambda *_: {}, command=lambda _: '')
        self.report['kernel_instructions']['another_kernel'] = 1
        self.write('resources', json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'single dense kernel'):
            self.validate()

    def test_unique_exported_code_stub_exact_pair(self):
        for bad in (NM + NM.splitlines()[0] + '\n', NM.replace(occupancy.STUB, occupancy.STUB + '_other'),
                    NM.replace(' W ', ' D ')):
            with self.subTest(nm=bad), self.assertRaises(ValueError):
                self.validate(nm=bad)

    def test_only_pure_host_constant_query_supported(self):
        self.assertEqual(occupancy.shared_constant(ASSEMBLY), 32816)
        with self.assertRaises(ValueError):
            occupancy.shared_constant(ASSEMBLY.replace('ret', 'call 1234'))
        with self.assertRaises(ValueError):
            occupancy.shared_constant(ASSEMBLY.replace('0x8030', '0x10001'))
        with self.assertRaises(ValueError):
            occupancy.shared_constant(ASSEMBLY.replace('    4375:', 'unparsed suspicious instruction\n    4375:'))

    def test_generated_constexpr_and_native_context(self):
        self.write('generated_source', (self.root / 'generated_source').read_text().replace('constexpr', ''))
        with self.assertRaisesRegex(ValueError, 'constexpr kThreads'):
            self.validate()

    def test_compile_receipt_failure_snapshot_context(self):
        self.spec['context_kind'] = 'compile_receipt'
        context = {name + '_sha256': self.spec[name]['sha256'] for name in ('binary', 'generated_source', 'log', 'source_snapshot')}
        context.update(returncode=0, same_compile_mir=self.spec['log']['sha256'])
        self.write('context', json.dumps(context))
        self.assertEqual(self.validate()['threads'], 576)
        context['binary_sha256'] = '0' * 64
        self.write('context', json.dumps(context))
        with self.assertRaisesRegex(ValueError, 'receipt mismatch'):
            self.validate()

    def test_guard_requires_consistent_env_actual_ancestor_flock(self):
        env = dict(TILEMEGA_DEVICE_INDEX='6', CUDA_VISIBLE_DEVICES='6', MACA_VISIBLE_DEVICES='6',
                   TILEMEGA_GPU_LOCK_HELD='1', TILEMEGA_GPU_LOCK='/tmp/tilemega_mx_gpu6.lock')
        stat = SimpleNamespace(st_dev=42, st_ino=88)
        line = '1: FLOCK ADVISORY WRITE 123 00:2a:88 0 EOF'
        ancestors = {123: 'python -m tilemega.bench.gpu_guard --policy x'}
        self.assertEqual(occupancy.validate_guard(env, stat, line, ancestors)['owner']['pid'], 123)
        for changed, locks, owners in ((dict(env, MACA_VISIBLE_DEVICES='7'), line, ancestors),
                                      (env, '', ancestors), (env, line, {})):
            with self.assertRaises(ValueError):
                occupancy.validate_guard(changed, stat, locks, owners)

    def test_mock_query_only_success_and_real_zero_preserved(self):
        validated = self.validate()
        for resident in (0, 1):
            runtime, library = FakeRuntime(resident=resident), FakeLibrary()
            row = occupancy.query(validated, runtime, library, {'CPU_MOCK_ONLY': True})
            self.assertEqual(row['kind'], 'mx5_plan_occupancy')
            self.assertEqual(row['query_status'], 0)
            self.assertEqual(row['resident_ctas'], resident)
            self.assertEqual(row['threads'], 576)
            self.assertEqual(row['binary_sha256'], self.spec['binary']['sha256'])
            self.assertEqual(row['iterations'], 0)
            self.assertTrue(row['query_only'] and row['no_launch'])
            self.assertFalse(row['execution_authorized'] or row['grants_512_exception'])
            self.assertEqual(library.calls, ['shared_bytes', ('stub_address_only', occupancy.STUB)])
            self.assertEqual(runtime.calls, ['attributes', ('set_shared', 8, 32816), ('occupancy', 576, 32816)])
            self.assertEqual(len(bytes.fromhex(row['attributes_raw_hex'])), 56)

    def test_mock_api_errors_and_unknown_residency_preserved(self):
        validated = self.validate()
        for keyword, source in (('attr_rc', 'attributes_returncode'), ('set_rc', 'set_dynamic_shared_returncode'),
                                ('occupancy_rc', 'occupancy_returncode')):
            with self.subTest(api=keyword):
                row = occupancy.query(validated, FakeRuntime(**{keyword: 17}), FakeLibrary(), {'CPU_MOCK_ONLY': True})
                self.assertEqual(row['query_status'], 17)
                self.assertEqual(row[source], 17)
                self.assertIsNone(row['resident_ctas'])
        row = occupancy.query(validated, FakeRuntime(threads=512), FakeLibrary(), {})
        self.assertNotEqual(row['query_status'], 0)
        self.assertIn('maxThreads', row['exception'])

    def test_shared_host_query_uses_uint64_restype(self):
        function = mock.Mock(return_value=32816)
        handle = SimpleNamespace(tm_plan_shared_bytes=function)
        with mock.patch.object(occupancy.C, 'CDLL', return_value=handle):
            library = occupancy.Library('CPU mock library')
            self.assertEqual(library.shared_bytes(), 32816)
        self.assertIs(function.restype, C.c_uint64)
        self.assertEqual(function.argtypes, [])
        function.assert_called_once_with()

    def test_runtime_wrapper_abi_and_stub_never_invoked(self):
        handle = SimpleNamespace(wcudaFuncGetAttributes=mock.Mock(), wcudaFuncSetAttribute=mock.Mock(),
                                 wcudaOccupancyMaxActiveBlocksPerMultiprocessor=mock.Mock())
        with mock.patch.object(occupancy.C, 'CDLL', return_value=handle):
            runtime = occupancy.Runtime()
        self.assertEqual(C.sizeof(occupancy.Attributes), 56)
        self.assertEqual(runtime.occupancy.argtypes, [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t])
        self.assertEqual(runtime.set_shared.argtypes, [C.c_void_p, C.c_int, C.c_int])
        self.assertEqual(runtime.attributes.argtypes, [C.POINTER(occupancy.Attributes), C.c_void_p])
        called = []
        stub = C.CFUNCTYPE(None)(lambda: called.append('forbidden'))
        with mock.patch.object(occupancy.C, 'CDLL', return_value=SimpleNamespace(**{occupancy.STUB: stub})):
            library = occupancy.Library('CPU mock library')
            address = library.stub_address(occupancy.STUB)
        self.assertIsNotNone(address.value)
        self.assertEqual(called, [])

    def test_post_query_input_change_retains_raw_api_evidence(self):
        inputs = self.root / 'inputs.json'
        inputs.write_text(json.dumps(self.spec))
        output = self.root / 'changed_output.json'
        runtime_file = self.root / 'mock_runtime'
        runtime_file.write_text('CPU mock runtime identity')
        runtime = FakeRuntime()
        real_call = runtime.occupancy
        def changed(*args):
            status = real_call(*args)
            (self.root / 'binary').write_text('changed after mock query')
            return status
        runtime.occupancy = changed
        validated = self.validate()
        with mock.patch.object(occupancy, 'guard_evidence', return_value={'CPU_MOCK_ONLY': True}), \
             mock.patch.object(occupancy, 'validate_inputs', return_value=validated), \
             mock.patch.object(occupancy, 'Runtime', return_value=runtime), \
             mock.patch.object(occupancy, 'Library', return_value=FakeLibrary()), \
             mock.patch.object(occupancy, 'RUNTIME', runtime_file):
            status = occupancy.main(['--inputs', str(inputs), '--out', str(output), '--query-only', '--iterations', '0'])
        row = json.loads(output.read_text())
        self.assertEqual(status, 1)
        self.assertNotEqual(row['query_status'], 0)
        self.assertIn('attributes_raw_hex', row)
        self.assertEqual(row['occupancy_returncode'], 0)
        self.assertIn('bound input changed', row['exception'])

    def test_main_exclusive_output_cpu_mock_only(self):
        inputs = self.root / 'inputs.json'
        inputs.write_text(json.dumps(self.spec))
        output = self.root / 'output.json'
        runtime_file = self.root / 'mock_runtime'
        runtime_file.write_text('CPU mock runtime identity')
        with mock.patch.object(occupancy, 'guard_evidence', return_value={'CPU_MOCK_ONLY': True}), \
             mock.patch.object(occupancy, 'validate_inputs', return_value=self.validate()), \
             mock.patch.object(occupancy, 'Runtime', return_value=FakeRuntime()), \
             mock.patch.object(occupancy, 'Library', return_value=FakeLibrary()), \
             mock.patch.object(occupancy, 'RUNTIME', runtime_file):
            argv = ['--inputs', str(inputs), '--out', str(output), '--query-only', '--iterations', '0']
            self.assertEqual(occupancy.main(argv), 0)
            self.assertEqual(json.loads(output.read_text())['resident_ctas'], 1)
            original = output.read_bytes()
            with self.assertRaises(SystemExit):
                occupancy.main(argv)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
