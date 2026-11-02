"""A diagnostic symlink cannot turn one five-process cohort into two."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'experiments/mx7/aggregate_probes.py'
spec = importlib.util.spec_from_file_location('mx7_probe_evidence', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProbeEvidenceTests(unittest.TestCase):
    def test_aliases_preserve_the_directed_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'actual.json'
            original.write_text('{}')
            alias = Path(directory) / 'alias.json'
            alias.symlink_to(original)
            key = module.canonical_evidence
            self.assertEqual(key(str(original)), key(str(alias)))
            self.assertEqual(key(str(original) + ':0->1'), key(str(alias) + ':0->1'))
            self.assertNotEqual(key(str(original) + ':0->1'), key(str(alias) + ':1->0'))


if __name__ == '__main__':
    unittest.main()
