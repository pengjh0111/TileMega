"""CPU-only synthetic evidence tests; no CUDA import or real results written."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/dnn1_tvm_incremental_publish.py'
SPEC = importlib.util.spec_from_file_location('incremental', SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write_csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())


class PublisherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(M, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        checker = self.root / 'python/tilemega/dnn/baselines.py'
        checker.parent.mkdir(parents=True)
        checker.write_text('# synthetic checker fixture\n')
        self.epoch = self.root / 'epoch'
        job = self.epoch / 'dispatch/workers/cell/raw/attempt1/job'
        job.mkdir(parents=True)
        self.session = job / 'cell-result.json'
        self.session.write_text(json.dumps(dict(model='mbv1', batch=2, gpu=0, status='complete')))
        outer = job.parent / 'guard.jsonl'
        outer.write_text('{"fixture":"outer sampled evidence"}\n')
        (job.parent / 'guard_result.json').write_text('{"code":0}\n')
        self.paths, processes = [], []
        for index in range(5):
            path = job / 'measurements' / str(index) / 'measurements.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(system='tvm', model='mbv1', batch=2,
                precision='bf16', gpu_index=0, correctness_passed=True,
                correctness=[dict(cosine_min_vs_fp32=0.999999, top1_agreement_vs_fp32=1.0)])))
            snapshots = path.parent / 'guard.jsonl'
            snapshots.write_text('{"fixture":"process sampled evidence"}\n')
            self.paths.append(path)
            processes.append(dict(receipt=str(path), receipt_sha256=M.sha(path.read_bytes()),
                snapshots_sha256=M.sha(snapshots.read_bytes()), issues=[]))
        self.audit = dict(model='mbv1', batch=2, precision='bf16', gpu=0,
            session=str(self.session), session_sha256=M.sha(self.session.read_bytes()),
            outer_guard_sha256=M.sha(outer.read_bytes()), processes=processes,
            sampled_windows_clean=True)
        self.incoming = dict(model='mbv1', batch='2', system='tvm', precision='bf16',
            arm='tvm-bf16', gpu='0', status='pass', correctness_passes='5/5',
            versions=json.dumps(dict(tvm_recipe_schema=6)), cell_session=str(self.session),
            source=json.dumps([str(p) for p in self.paths]), median_ms='0.6', p90_ms='0.7',
            process_range_ms='0.01', process_range_pct='1.66',
            correctness_metrics_identical='True', correctness_policy='P-CORR-v2', strict_v1_status='pass')
        self.diag = dict(model='mbv1', batch='2', precision='bf16', status='pass',
            library_sha256='a' * 64, source=str(self.session))
        results = self.root / 'docs/results'
        results.mkdir(parents=True)
        self.files = [results / name for name in M.NAMES]
        primary = [dict(self.incoming, model='primary' + str(i), system='torchcompile',
                        source='multiline primary\nrecord ' + str(i)) for i in range(125)]
        old = dict(self.incoming, status='N/A', median_ms='', source='historical failure\nverbatim')
        write_csv(self.files[0], list(self.incoming), primary + [old, dict(old, precision='fp16')])
        old_diag = dict(self.diag, status='N/A')
        write_csv(self.files[1], list(self.diag), [old_diag, dict(old_diag, precision='fp16')])
        correctness = M.correctness_row(self.incoming, self.diag,
            [json.loads(p.read_text()) for p in self.paths], self.session, self.audit['session_sha256'])
        write_csv(self.files[2], list(correctness), [])

    def contents(self):
        return [path.read_bytes() for path in self.files]

    def merge(self, incoming=None, audit=None):
        return M.merge_one(self.epoch, incoming or self.incoming, self.diag,
                           audit or self.audit, 'b' * 64)

    def test_multiline_preservation_and_idempotence(self):
        before = M.load(self.files[0])
        result = self.merge()
        receipt = json.loads(Path(result).read_text())
        after = M.load(self.files[0])
        self.assertEqual(before['header'], after['header'])
        self.assertEqual(before['records'][:125], after['records'][:125])
        self.assertIn('multiline primary\nrecord', after['records'][0][1])
        self.assertEqual(len(after['records']), 126)
        self.assertEqual(len(receipt['verified_raw_sha256']), 13)
        accepted = self.contents()
        self.assertIsNone(self.merge())
        self.assertEqual(accepted, self.contents())

    def test_accepted_row_replacement_rejected(self):
        self.merge()
        before = self.contents()
        for field in ('source', 'median_ms'):
            with self.subTest(field=field):
                changed = dict(self.incoming, **{field: 'changed'})
                with self.assertRaisesRegex(ValueError, 'accepted row differs'):
                    M.merge_records(M.load(self.files[0]), changed)
                self.assertEqual(before, self.contents())

    def test_failed_and_dirty_not_published(self):
        before = self.contents()
        cases = [(dict(self.incoming, status='N/A'), self.audit),
                 (dict(self.incoming, correctness_passes='4/5'), self.audit),
                 (self.incoming, dict(self.audit, sampled_windows_clean=False))]
        dirty = copy.deepcopy(self.audit)
        dirty['processes'][0]['issues'] = ['unexpected process']
        cases.append((self.incoming, dirty))
        for incoming, audit in cases:
            with self.subTest(incoming=incoming['status'], clean=audit['sampled_windows_clean']):
                with self.assertRaises(ValueError):
                    self.merge(incoming, audit)
                self.assertEqual(before, self.contents())

    def test_each_raw_evidence_hash_rechecked(self):
        before = self.contents()
        paths = [self.session, self.session.parent.parent / 'guard.jsonl',
                 self.paths[0], self.paths[0].parent / 'guard.jsonl']
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    self.merge()
                self.assertEqual(before, self.contents())
                path.write_bytes(original)

    def test_raw_change_during_merge_prevents_writes(self):
        before = self.contents()
        original = M.correctness_row
        def mutate_after_verification(*args):
            row = original(*args)
            self.paths[0].write_bytes(self.paths[0].read_bytes() + b' ')
            return row
        with patch.object(M, 'correctness_row', side_effect=mutate_after_verification):
            with self.assertRaisesRegex(RuntimeError, 'raw evidence changed during publication'):
                self.merge()
        self.assertEqual(before, self.contents())

    def test_audit_must_cover_exact_receipts_and_cell(self):
        for edit in ('model', 'duplicate_receipt'):
            audit = copy.deepcopy(self.audit)
            if edit == 'model':
                audit['model'] = 'another_model'
            else:
                audit['processes'][-1] = audit['processes'][0]
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                self.merge(audit=audit)


if __name__ == '__main__':
    unittest.main()
