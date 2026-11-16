"""Classic publication keeps retained cells and gates the final release ledger."""
from pathlib import Path
import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_publish_aitemplate as publisher


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.epoch = self.root / 'epoch'
        self.epoch.mkdir()
        self.args = argparse.Namespace(epoch=[self.epoch], out=self.root / 'out',
            failure_declaration=[], complete=False, workload='dnn-classic-v1')

    def publish(self, cells):
        (self.epoch / 'accepted.json').write_text(json.dumps([
            dict(model=model, batch=batch) for model, batch in cells]))
        def checked_row(value, fields):
            row = dict.fromkeys(fields, '')
            row.update(model=value['model'], batch=value['batch'], status='pass',
                system='aitemplate', precision='bf16', versions='original identity')
            return row, dict(model=value['model'], batch=value['batch'], status='pass')
        with patch.object(publisher, 'accepted_row', side_effect=checked_row), \
             contextlib.redirect_stdout(io.StringIO()):
            publisher.publish(self.args)

    def test_retained_ten_publish_as_classic_partial_without_final_ledger(self):
        self.publish(sorted(publisher.classic.RETAINED))
        path = self.args.out / 'DNN_classic_aitemplate_partial.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r['versions'] == 'original identity' for r in rows))
        self.assertFalse((self.args.out / 'DNN_classic_aitemplate_completion.json').exists())
        self.assertFalse((self.args.out / 'DNN1_aitemplate_partial.csv').exists())

    def test_missing_or_old_model_cannot_create_completion(self):
        self.args.complete = True
        for cells in (sorted(publisher.classic.RETAINED), [('nafnet', 1)]):
            with self.assertRaises(ValueError):
                self.publish(cells)
        self.assertFalse(self.args.out.exists())

    def test_exact_new_grid_creates_sha_bound_workload_ledger(self):
        self.args.complete = True
        self.publish(sorted(publisher.classic.CELLS))
        ledger = json.loads((self.args.out / 'DNN_classic_aitemplate_completion.json').read_text())
        self.assertEqual(ledger['status'], 'final_25_terminal')
        self.assertEqual(ledger['workload'], publisher.classic.WORKLOAD)
        self.assertEqual(ledger['workload_id'], 'dnn-classic-v1')
        self.assertFalse(ledger['repair_pending'])
        self.assertEqual(ledger['csv_sha256'], publisher.base.sha(
            self.args.out / 'DNN_classic_aitemplate.csv'))
        self.assertEqual(len(ledger['cells']), 25)
        self.assertTrue(all(row['final'] and row['status'] == 'pass' for row in ledger['cells']))

    def test_duplicate_epochs_cannot_select_or_mix_cells(self):
        self.args.epoch.append(self.epoch)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.publish([('resnet18', 1)])


if __name__ == '__main__':
    unittest.main()
