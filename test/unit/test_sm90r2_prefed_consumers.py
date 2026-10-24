#!/usr/bin/env python3
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from collect_sm90r2_prefed_consumers import (parse, RAW_FIELDS, CTA_FIELDS, launch_order, load_csv,
                            validate_rows, validate_metadata, aggregate)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'runs/sm90r2/flow-consumer-prefed-v2/gpu/raw/flow-consumer-prefed-v2-f0/attempt1/unit/probe'


class RowEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.cta = dict(fresh='0', active_ctas='1', feed_mode='prefed', observe_empty='0',
                        trial='1', cta='0', smid='124', has_work='1', full_ready_mask='3',
                        full0_confirmed='10', full1_confirmed='20', setup_acquired='30',
                        v3_arrive='100', v3_acquired='120')
        self.raw = {k: v for k, v in self.cta.items() if k != 'has_work'}
        self.raw.update(warp='0', entry='40', returned='80', empty0_wait_begin='0',
                        empty0_observed_ready='0', empty1_wait_begin='0', empty1_observed_ready='0')

    def test_valid_prefed_row(self):
        self.assertEqual(parse(self.raw, RAW_FIELDS, 0)['full_ready_mask'], 3)

    def test_reject_missing_full_slot(self):
        self.cta['full_ready_mask'] = '1'
        with self.assertRaisesRegex(ValueError, 'full-ready'):parse(self.cta, CTA_FIELDS, 0)

    def test_reject_prefeed_after_setup(self):
        self.cta['full1_confirmed'] = '31'
        with self.assertRaisesRegex(ValueError, 'full-ready'):parse(self.cta, CTA_FIELDS, 0)

    def test_concurrent_zero_confirmation_is_not_unready_claim(self):
        self.cta.update(feed_mode='concurrent', full_ready_mask='0', full0_confirmed='0', full1_confirmed='0')
        self.assertEqual(parse(self.cta, CTA_FIELDS, 0)['feed_mode'], 'concurrent')

    def test_reject_mixed_process(self):
        with self.assertRaisesRegex(ValueError, 'process'):parse(self.raw, RAW_FIELDS, 1)

    def test_reject_observer_in_off_control(self):
        self.raw['empty0_observed_ready'] = '70'
        with self.assertRaisesRegex(ValueError, 'off control'):parse(self.raw, RAW_FIELDS, 0)

    def test_reject_consume_before_setup(self):
        self.raw['entry'] = '29'
        with self.assertRaisesRegex(ValueError, 'Consume'):parse(self.raw, RAW_FIELDS, 0)

    def test_five_process_aggregate_rejects_missing_fresh(self):
        row = dict(fresh=0, active_ctas=1, feed_mode='prefed', observe_empty='0',
                   scope='cta', metric='connected', samples=16, median=32, p99=64)
        with self.assertRaisesRegex(ValueError, 'process statistic'):aggregate([row], range(5))

    def test_interleave_plan(self):
        self.assertEqual(launch_order()[:4], [(1, 'prefed', 0, 1), (1, 'prefed', 1, 1),
                                           (1, 'concurrent', 0, 1), (1, 'concurrent', 1, 1)])
        self.assertEqual(len(set(launch_order())), 256)


@unittest.skipUnless(FIXTURE.exists(), 'actual completed f0 fixture unavailable')
class ActualEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_csv(FIXTURE / 'raw.csv', RAW_FIELDS, 0)
        cls.cta = load_csv(FIXTURE / 'ctas.csv', CTA_FIELDS, 0)

    def test_complete_actual_fresh(self):
        observations, ctas = validate_rows(self.raw, self.cta)
        self.assertEqual(len(observations), 14400)
        self.assertEqual(len(ctas), 33792)

    def test_reject_missing_warp(self):
        with self.assertRaisesRegex(ValueError, 'raw cohort'):validate_rows(self.raw[:-1], self.cta)

    def test_reject_fabricated_smid(self):
        raw = list(self.raw);raw[0] = dict(raw[0], smid=999)
        with self.assertRaisesRegex(ValueError, 'metadata differs'):validate_rows(raw, self.cta)

    def test_reject_duplicate_resident_smid(self):
        ctas = list(self.cta);ctas[1] = dict(ctas[1], smid=ctas[0]['smid'])
        with self.assertRaisesRegex(ValueError, 'distinct actual SMIDs'):validate_rows(self.raw, ctas)

    def test_reject_reordered_controls(self):
        raw = list(self.raw);raw[0], raw[4] = raw[4], raw[0]
        with self.assertRaisesRegex(ValueError, 'interleaved context'):validate_rows(raw, self.cta)

    def test_reject_acquire_before_last_arrival(self):
        ctas = list(self.cta);ctas[0] = dict(ctas[0], v3_acquired=min(r['v3_arrive'] for r in ctas[:132]))
        with self.assertRaisesRegex(ValueError, 'acquire predates'):validate_rows(self.raw, ctas)


if __name__ == '__main__':unittest.main()
