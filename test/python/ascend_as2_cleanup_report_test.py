"""Cleanup audit compaction retains every removal and replay provenance."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from as1_cleanup import compact_as2_report


class CleanupReport(unittest.TestCase):
    def test_retains_every_candidate_without_changing_the_safety_plan(self):
        kept=[dict(path=f'owned/retained{i}',action='keep',bytes=0,reason='Retained evidence') for i in range(10000)]
        candidate=dict(path='owned/completed.o',action='remove',bytes=64,stamp=dict(inode=7))
        report=dict(mode='dry-run',items=kept+[candidate],eligible_bytes=64,removed_files=0,released_bytes=0)
        prefix=b'{"path":"owned/completed.o","completed":true}\n'
        result=compact_as2_report(report,Path('owned/manifest.jsonl'),prefix)
        self.assertEqual(result['items'],[candidate]);self.assertEqual(len(report['items']),10001)
        self.assertEqual(result['kept_reason_counts'],{'Retained evidence':10000})
        self.assertEqual(result['manifest_prefix_sha256'],hashlib.sha256(prefix).hexdigest())
        self.assertEqual(result['manifest_prefix_bytes'],len(prefix))
        self.assertEqual(result['eligible_bytes'],64);self.assertEqual(result['removed_files'],0)
        self.assertLess(len(str(result)),1000)

    def test_zero_eligible_dry_run_and_apply_receipts_remain_distinguishable(self):
        for mode in ('dry-run','apply'):
            report=dict(mode=mode,items=[],eligible_bytes=0,removed_files=0,released_bytes=0)
            result=compact_as2_report(report,Path('owned/manifest.jsonl'),b'')
            self.assertEqual(result['mode'],mode);self.assertEqual(result['items'],[])
            self.assertEqual(result['inspected_entries'],0)


if __name__=='__main__':unittest.main()
