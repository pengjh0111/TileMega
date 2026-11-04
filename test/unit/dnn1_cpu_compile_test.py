"""The bounded CPU lane never certifies failed or mutable build inputs."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_cpu_compile as c


class CpuLane(unittest.TestCase):
    def manifest(self, limit=168):
        return dict(phase='forward', native_forward={},
                    register_contract=dict(threads=160, callee_limit=limit))

    def test_exact_native_flags_and_register_constraint(self):
        command = c.nvcc_command(Path('/cuda/nvcc'), Path('/source'), Path('/out/plan.so'), self.manifest())
        self.assertEqual(command, ['/cuda/nvcc', '-rdc=true', '-std=c++17', '-O3', '-gencode',
            'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr', '-shared', '-Xcompiler',
            '-fPIC', '-cudart', 'shared', '-Xptxas', '-v', '--split-compile', '4',
            '-I/source/include', '-I/source/third_party/cutlass/include', '/out/plan.so.cu',
            '-o', '/out/plan.so', '-Xptxas', '--device-function-maxrregcount=168'])
        self.assertNotIn('--device-function-maxrregcount=255', c.nvcc_command(
            Path('/cuda/nvcc'), Path('/source'), Path('/out/plan.so'), self.manifest(255)))
        with self.assertRaises(ValueError): c.nvcc_command(Path('/n'), Path('/s'), Path('/o'), self.manifest(256))

    def test_source_mutation_and_output_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input'; p.write_text('before'); bound = {str(p): c.sha(p)}
            self.assertTrue(c.unchanged(bound)); p.write_text('after'); self.assertFalse(c.unchanged(bound))
            output = Path(d) / 'plan.so'; c.reserve(output)
            with self.assertRaises(ValueError): c.reserve(output)
            self.assertFalse(output.exists())

    def test_failure_and_postcompile_mutation_never_get_identity(self):
        for mutation in (False, True):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d); names = ('compiler', 'nvcc', 'reject', 'input', 'coordinates', 'target')
                paths = {}
                for name in names:
                    paths[name] = root / name; paths[name].write_text(name)
                a = SimpleNamespace(source_root=root, compiler=paths['compiler'], nvcc=paths['nvcc'],
                    reject_cuda=paths['reject'], input=paths['input'], coordinates=paths['coordinates'],
                    target=paths['target'], output=root / 'plan.so', batch=1)
                calls = []
                def identity(root, args, env):
                    calls.append(args); self.assertIn('--capture', args)
                    snapshot = Path(args[args.index('--snapshot') + 1])
                    c.write(snapshot, dict(root=str(root), source_files={'input': c.sha(paths['input'])}, source_digest='fixture'))
                    Path(str(snapshot) + '.patch').write_text('fixture patch')
                def bounded(command, receipt, log, env, root):
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                    self.assertEqual(env['LD_PRELOAD'], str(paths['reject']))
                    Path(log).write_text('fixture compiler log')
                    emit = command[0] == str(paths['compiler'])
                    if emit:
                        cu = Path(command[3]); cu.write_text('// fixture CU')
                        cg = Path(command[command.index('--dump-cg') + 1]); cg.write_text('fixture CG')
                        manifest = self.manifest(); manifest.update(export_sha256=c.sha(paths['input']),
                            coordinates_sha256=c.sha(paths['coordinates']), plan_sha256=c.sha(cg),
                            plan_module=str(cg), runtime_batch=1, runtime_target=str(paths['target']))
                        c.write(str(cu) + '.plan.json', manifest)
                    else:
                        a.output.write_text('uncertified fixture binary')
                        if mutation: paths['input'].write_text('changed while compiler ran')
                    ok = emit or mutation
                    result = dict(command=command, status='completed' if ok else 'build_failed',
                        exit_code=0 if ok else 1, lock=c.LOCK, max_address_space_gib=48, timeout_s=5400,
                        classification='compilation; not a numerical test')
                    c.write(receipt, result); return (0 if ok else 1), result
                with patch.object(c, 'identity_call', side_effect=identity), patch.object(c, 'bounded', side_effect=bounded), \
                     patch.object(c.os, 'sched_setaffinity'), patch.object(c.os, 'sched_getaffinity', return_value=c.AFFINITY):
                    with self.assertRaises(RuntimeError): c.compile_artifact(a)
                self.assertEqual(len(calls), 1)
                self.assertFalse(Path(str(a.output) + '.identity.json').exists())
                result = json.loads(Path(str(a.output) + '.compile-result.json').read_text())
                self.assertEqual(result['status'], 'compiler_error')
                self.assertEqual(result['classifier'], 'dnn1_bounded_cpu_lane_v1')
                self.assertEqual(result['inputs_unchanged'], not mutation)

    def test_wrong_lane_or_unbounded_receipt_rejected(self):
        r = dict(status='completed', exit_code=0, command=['nvcc'], lock=c.LOCK,
                 max_address_space_gib=48, timeout_s=5400, classification='compilation; not a numerical test')
        self.assertTrue(c.bounded_success(0, r, ['nvcc']))
        for key, value in [('lock', '/tmp/old_shared.lock'), ('timeout_s', 99999), ('command', ['other'])]:
            bad = dict(r, **{key: value}); self.assertFalse(c.bounded_success(0, bad, ['nvcc']))


if __name__ == '__main__': unittest.main()
