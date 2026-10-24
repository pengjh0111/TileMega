#!/usr/bin/env python3
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collect_sm90r2_joint_qkv import RAW,CTAS,CONTEXTS,load,parse,validate_rows,launches
ROOT=Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'runs/sm90r2/flow-joint-qkv-v1/gpu/raw/flow-joint-qkv-v1-f0/attempt1/unit/probe'

@unittest.skipUnless(FIXTURE.exists(),'completed actual fresh0 fixture unavailable')
class JointEvidenceTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=load(FIXTURE/'raw.csv',RAW,0);cls.ctas=load(FIXTURE/'ctas.csv',CTAS,0);cls.contexts=load(FIXTURE/'contexts.csv',CONTEXTS,0)
 def test_complete_actual_cohort_and_integer_deltas(self):
  observations,ctas=validate_rows(self.raw,self.ctas,self.contexts)
  self.assertEqual((len(observations),len(ctas)),(7200,16896))
  first=next(iter(observations.values()));self.assertGreater(first['joint_entry_min'],2**53)
  self.assertIsInstance(first['producer_call_ns'],int)
  self.assertEqual(first['producer_call_ns'],first['producer_returned']-first['producer_entry'])
 def test_reject_wrong_counter(self):
  row={k:str(v) for k,v in self.contexts[0].items()};row['setup_arrivals']='131'
  with self.assertRaisesRegex(ValueError,'counters'):parse(row,CONTEXTS,0)
 def test_reject_wrong_sentinel_coverage(self):
  row={k:str(v) for k,v in self.contexts[0].items()};row['checked_sentinels']='128'
  with self.assertRaisesRegex(ValueError,'sentinel coverage'):parse(row,CONTEXTS,0)
 def test_reject_wrong_generation(self):
  row={k:str(v) for k,v in self.ctas[0].items()};row['producer_phase1']='0'
  with self.assertRaisesRegex(ValueError,'generation'):parse(row,CTAS,0)
 def test_reject_missing_consumer_warp(self):
  with self.assertRaisesRegex(ValueError,'raw cohort'):validate_rows(self.raw[:-1],self.ctas,self.contexts)
 def test_reject_duplicate_actual_smid(self):
  ctas=list(self.ctas);ctas[1]=dict(ctas[1],smid=ctas[0]['smid'])
  with self.assertRaisesRegex(ValueError,'distinct actual SMIDs'):validate_rows(self.raw,ctas,self.contexts)
 def test_reject_different_producer_copy(self):
  raw=list(self.raw);raw[0]=dict(raw[0],producer_returned=raw[0]['producer_returned']+32)
  with self.assertRaisesRegex(ValueError,'producer stamps differ'):validate_rows(raw,self.ctas,self.contexts)
 def test_reject_reordered_cache_controls(self):
  contexts=list(self.contexts);contexts[0],contexts[1]=contexts[1],contexts[0]
  with self.assertRaisesRegex(ValueError,'interleaved contexts'):validate_rows(self.raw,self.ctas,contexts)
 def test_reject_acquire_before_last_arrival(self):
  ctas=list(self.ctas);ctas[0]=dict(ctas[0],v3_acquired=min(r['v3_arrive'] for r in ctas[:132]))
  with self.assertRaisesRegex(ValueError,'acquire predates'):validate_rows(self.raw,ctas,self.contexts)

if __name__=='__main__':unittest.main()
