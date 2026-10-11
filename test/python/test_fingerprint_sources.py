import tempfile
import unittest
from pathlib import Path

from tilemega.fingerprint import source_fingerprint


class SourceFingerprintTests(unittest.TestCase):
    def test_sparse_source_tree_needs_no_experiment_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CMakeLists.txt').write_text('project(example)\n')
            (root / 'include').mkdir()
            header = root / 'include/runtime.h'
            header.write_text('int value = 1;\n')
            before = source_fingerprint(root)
            (root / 'docs/experiments').mkdir(parents=True)
            (root / 'docs/experiments/measurements.tsv').write_text('historical data\n')
            self.assertEqual(before, source_fingerprint(root))
            header.write_text('int value = 2;\n')
            self.assertNotEqual(before, source_fingerprint(root))

    def test_build_configuration_is_a_source_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'CMakeLists.txt'
            config.write_text('project(first)\n')
            before = source_fingerprint(root)
            config.write_text('project(second)\n')
            self.assertNotEqual(before, source_fingerprint(root))


if __name__ == '__main__':
    unittest.main()
