import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('replay', Path(__file__).resolve().parents[2] / 'scripts/replay_serving_plan.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class Replay(unittest.TestCase):
    def test_rebinds_all_project_sources_and_preserves_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old, new = base / 'old', base / 'new'
            (old / 'include/tilemega').mkdir(parents=True)
            (new / 'lib/Target').mkdir(parents=True)
            (new / 'lib/Target/TargetSpec.cpp').touch()
            flags = ['-O3', '-Xptxas=-v', '-DTEST=7']
            for inc in [['-I' + str(old / 'include')], ['-I', str(old / 'include')]]:
                command = ['nvcc', *flags, *inc, '-I/cutlass/include',
                           str(old / 'plan.cu'), str(old / 'lib/Target/TargetSpec.cpp'), '-o', 'old.so']
                actual = replay.nvcc_command(command, new, new / 'plan.cu', new / 'plan.so')
                self.assertEqual(actual[1:4], flags)
                self.assertIn(str(new / 'lib/Target/TargetSpec.cpp'), actual)
                self.assertIn('-I/cutlass/include', actual)
                self.assertEqual(actual[-1], str(new / 'plan.so'))
                self.assertNotIn(str(old / 'plan.cu'), actual)
                self.assertIn(str(new / 'include'), ' '.join(actual))

    def test_rejects_missing_output_or_source(self):
        for cmd in [['nvcc', 'plan.cu'], ['nvcc', '-o', 'p.so'], ['nvcc', 'p.cu', '-o']]:
            with self.assertRaises(ValueError):
                replay.nvcc_command(cmd, Path('/new'), Path('/new/p.cu'), Path('/new/p.so'))

if __name__ == '__main__':
    unittest.main()
