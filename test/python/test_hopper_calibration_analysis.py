# SPDX-License-Identifier: BSD-3-Clause
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[2]/'scripts/run_r16_hopper_calibration.py'
spec=importlib.util.spec_from_file_location('hopper_calib_driver',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class HopperCalibrationAnalysisTest(unittest.TestCase):
    def fixtures(self,root,scale=1.02):
        receipts=[]
        keys=[(name,1,-1,'GB/s') for name in ['ldg128','cp_async','bulk','tma']]
        keys += [('cluster_barrier' if peer==0 else 'dsmem_ticket_rtt',c,peer,'ns/op') for c in [2,4,8] for peer in range(c)]
        for gpu in [1,2]:
            for fresh in range(5):
                d=root/f'{gpu}-{fresh}';d.mkdir()
                records=[dict(name=n,cluster_dim=c,peer_rank=p,unit=u,median=100*(scale if gpu==2 else 1)) for n,c,p,u in keys]
                fragment=d/'fragment.json';m.write(fragment,{'records':records})
                raw=d/'fragment.json.tsv';raw.write_text('checksum_pass\n1\n')
                value=dict(passed=True,exit_code=0,child_pid=gpu*1000+fresh,fresh_index=fresh,fragment=str(fragment),raw=str(raw),fragment_sha256=m.sha(fragment),raw_sha256=m.sha(raw),records=records,config=dict(gpu=gpu,cli_sha256='cli',source_head='source',base_sha256='base'))
                m.write(d/'process.json',value);m.write(d/'guard_result.json',dict(code=0,ownership_verified=False,acceptance='provisional_container_pid_namespace'));receipts.append(d/'process.json')
        return receipts
    def test_original_three_percent_gate_and_provisional_ownership(self):
        with tempfile.TemporaryDirectory() as t:
            ps=self.fixtures(Path(t),1.03);v=m.analyze_receipts(ps)
            self.assertTrue(v['passed']);self.assertFalse(v['ownership_verified']);self.assertEqual(v['acceptance'],'provisional_container_pid_namespace');self.assertEqual(len(v['rows']),18)
    def test_excess_difference_is_retained_without_changing_threshold(self):
        with tempfile.TemporaryDirectory() as t:
            ps=self.fixtures(Path(t),1.0301);v=m.analyze_receipts(ps)
            self.assertFalse(v['passed']);self.assertGreater(v['rows'][0]['relative_difference'],.03)
    def test_failed_guard_and_changed_raw_are_rejected(self):
        for mode in ['guard','raw']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                ps=self.fixtures(Path(t));d=ps[0].parent
                if mode=='guard':m.write(d/'guard_result.json',dict(code=75))
                else:(d/'fragment.json.tsv').write_text('checksum_pass\n0\n')
                v=m.analyze_receipts(ps);self.assertFalse(v['passed']);self.assertEqual(len(v['rejected']),1)
    def test_five_distinct_processes_and_same_binary_required(self):
        for field in ['child_pid','cli_sha256']:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
                ps=self.fixtures(Path(t));v=m.read(ps[0])
                if field=='child_pid':v[field]=m.read(ps[1])[field]
                else:v['config'][field]='other'
                m.write(ps[0],v);self.assertFalse(m.analyze_receipts(ps)['passed'])
    def test_nonpositive_subtraction_and_missing_process_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            ps=self.fixtures(Path(t));self.assertFalse(m.analyze_receipts(ps[:-1])['passed'])
            for p in ps[:5]:
                v=m.read(p);v['records'][0]['median']=-.2;m.write(v['fragment'],{'records':v['records']});v['fragment_sha256']=m.sha(v['fragment']);m.write(p,v)
            self.assertFalse(m.analyze_receipts(ps)['passed'])

if __name__=='__main__':unittest.main()
