import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'generated_checkpoint', ROOT / 'scripts/check_generated_models.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


class GeneratedCheckpointProcessTest(unittest.TestCase):
    def test_compiler_diagnostics_are_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(checkpoint.compiler_options({}, folder), [])
            options = checkpoint.compiler_options(
                dict(nvcc_split_compile=3, nvcc_diagnostics=True), folder)
            self.assertIn('-Xptxas=--split-compile=3', options)
            self.assertIn('--keep', options)
            self.assertTrue((folder / 'nvcc_intermediates').is_dir())
            self.assertFalse(any(option.startswith('-O') for option in options))
            with self.assertRaises(ValueError):
                checkpoint.compiler_options(dict(nvcc_split_compile=0), folder)

    def test_retry_copies_inputs_without_relabeling_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, out = Path(directory) / 'old', Path(directory) / 'retry'
            previous.mkdir()
            for name in ('source.json', 'source.json.patch', 'target.json'):
                (previous / name).write_text(name)
            bridge, config = previous / 'bridge.json', previous / 'config.json'
            bridge.write_text('bridge')
            config.write_text('config')
            record = dict(root='/frozen/compiler/tree', compiler='/frozen/compiler',
                compiler_sha256='compiler-sha', source_sha256=checkpoint.sha(previous / 'source.json'),
                target_sha256=checkpoint.sha(previous / 'target.json'), timing_eligible=False,
                cases=[dict(label='decoder', bridge=str(bridge), config=str(config),
                    files={str(bridge):checkpoint.sha(bridge), str(config):checkpoint.sha(config)})])
            (previous / 'inputs.json').write_text(json.dumps(record))
            args = SimpleNamespace(reuse_inputs=previous, out=out, labels='decoder',
                nvcc_timeout_s=28800, nvcc_split_compile=3, nvcc_diagnostics=True)
            with patch.object(checkpoint, 'validate') as validate:
                checkpoint.reuse_inputs(args)
            validate.assert_called_once_with(previous, 'decoder')
            retry = json.loads((out / 'inputs.json').read_text())
            for key in ('root', 'compiler', 'compiler_sha256', 'source_sha256', 'target_sha256'):
                self.assertEqual(retry[key], record[key])
            self.assertEqual(retry['cases'][0]['files'][str(out / 'decoder/bridge.json')],
                             checkpoint.sha(bridge))
            self.assertEqual(retry['reused_inputs_sha256'], checkpoint.sha(previous / 'inputs.json'))
            self.assertFalse(retry['timing_eligible'])
            self.assertFalse((out / 'decoder/plan.so').exists())

    def test_invalid_frozen_input_prevents_retry_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, out = Path(directory) / 'old', Path(directory) / 'retry'
            previous.mkdir()
            (previous / 'inputs.json').write_text(json.dumps(dict(cases=[dict(label='bert')])))
            args = SimpleNamespace(reuse_inputs=previous, out=out, labels='bert')
            with patch.object(checkpoint, 'validate', side_effect=ValueError('compiler changed')):
                with self.assertRaisesRegex(ValueError, 'compiler changed'):
                    checkpoint.reuse_inputs(args)
            self.assertFalse(out.exists())

    def test_timeout_reaps_compiler_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            child_pid = folder / 'child.pid'
            program = (
                'import signal,subprocess,sys,time\n'
                'p=subprocess.Popen([sys.executable,"-c",'
                '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])\n'
                'open(sys.argv[1],"w").write(str(p.pid))\n'
                'time.sleep(60)\n')
            pid = None
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    checkpoint.execute([sys.executable, '-c', program, str(child_pid)],
                                       ROOT, folder, 'compile', timeout=1)
                pid = int(child_pid.read_text())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    path = Path('/proc', str(pid), 'stat')
                    if not path.exists() or path.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                        break
                    time.sleep(.02)
                else:
                    self.fail('compiler descendant survived its timeout')
                self.assertTrue((folder / 'compile.command.json').is_file())
            finally:
                if pid is not None:
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
