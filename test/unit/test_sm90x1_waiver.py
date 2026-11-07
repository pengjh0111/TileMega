"""A user waiver enables timing while preserving the strict paper-data gap."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90d1_common import read, write, sha
import sm90x1 as base
import sm90x1_experiment as experiment
import sm90x1_waiver as waiver
import sm90x1_watch as watch

CELL='qwen3_30b_a3b-B16'
AUTH=dict(authorized_by_user=True, skip_remaining_pcorr=True, reason='explicit user instruction')


class WaiverTest(unittest.TestCase):
    def test_authorization_is_explicit_and_scoped_to_one_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            policy=Path(temp)/'policy.json'
            with patch.object(waiver, 'POLICY', policy):
                self.assertIsNone(waiver.authorization(dict(cell=CELL)))
                for flags in ({}, dict(authorized_by_user=True), dict(skip_remaining_pcorr=True), AUTH):
                    write(policy, dict(correctness_waivers={CELL:flags}))
                    self.assertEqual(bool(waiver.authorization(dict(cell=CELL))), flags==AUTH)
                    self.assertIsNone(waiver.authorization(dict(cell='other')))

    def test_marker_cannot_manufacture_a_correctness_receipt(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(waiver, 'authorization', return_value=AUTH):
            root=Path(temp);waiver.record(dict(cell=CELL), root)
            marker=read(root/'correctness-waiver.json')
            self.assertIs(marker['pcorr_passed'], False)
            self.assertIs(marker['performance_eligible'], False)
            self.assertFalse(waiver.passed(root))
            self.assertFalse((root/'result.json').exists())
            write(root/'result.json', dict(c1_pass=True, c2_pass=True, **{'pass':True}))
            self.assertFalse(waiver.passed(root))
            write(root/'unit.json', {})
            self.assertTrue(waiver.passed(root))

    def test_only_authorized_timing_bypasses_missing_pcorr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'decode.so';binary.write_bytes(b'native fixture')
            cell=dict(cell=CELL, processes=5, model='qwen3_30b_a3b', batch=16,
                      prefill=str(binary), prefill_sha256=sha(binary), **{'class':'trial'})
            binding=dict(head='cohort', host_abi_status=0, trace=False,
                         decode=str(binary), decode_sha256=sha(binary))
            write(root/'W/binding.json', binding)
            write(root/'arm-plan.json', [dict(arm=a,id='W') for a in ('W','L','Wstar')])
            args=argparse.Namespace(phase=2, tag='phase2', action='time')
            with patch.object(experiment, 'gate'), patch.object(experiment, 'rows', return_value=[cell]), \
                 patch.object(experiment, 'cell_dir', return_value=root), \
                 patch.object(waiver, 'authorization', return_value=None), \
                 patch.object(experiment, 'run_unit') as run:
                with self.assertRaisesRegex(ValueError, 'P-CORR must pass'):experiment.measurement(args)
                run.assert_not_called()
            with patch.object(experiment, 'gate'), patch.object(experiment, 'rows', return_value=[cell]), \
                 patch.object(experiment, 'cell_dir', return_value=root), \
                 patch.object(waiver, 'authorization', return_value=AUTH), \
                 patch.object(experiment, 'run_unit') as run:
                args.action='corr';experiment.measurement(args);run.assert_not_called()
                args.action='time';experiment.measurement(args);run.assert_called_once()
            spec=read(root/'time-spec.json')
            self.assertEqual(spec['order'], ['W','L','Wstar','Wstar','L','W']*2+['W','L','Wstar'])
            self.assertEqual(spec['correctness_waiver'], AUTH)
            self.assertIs(spec['performance_eligible'], False)
            self.assertFalse((root/'W/corr/result.json').exists())

    def test_tables_keep_waived_timings_out_of_strict_aggregates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';results=root/'results';results.mkdir()
            cells=base.cells('all')
            for cell in cells:
                d=run/'cells'/cell['cell']/('phase1' if cell['pilot'] else 'phase2')
                coords=d/'coordinates.json';write(coords, cell['coordinates'])
                write(d/'arm-plan.json', [dict(arm=a,id='W',coordinates=str(coords),coordinates_sha256='fixture')
                                         for a in ('W','L','Wstar')])
                write(d/'W/binding.json', dict(decode_sha256='fixture',head='fixture',trace=False))
                if cell['cell']!=CELL:
                    write(d/'W/corr/result.json',dict(c1_pass=True,c2_pass=True,**{'pass':True}))
                arms={arm:dict(tpot_ms=ms,range_pct=.1,processes=cell['processes'])
                      for arm,ms in [('W',10),('L',12),('Wstar',8)]}
                protocol=dict(gpu=0)
                if cell['cell']==CELL:
                    for arm,gpus in [('W',[0,1]),('L',[1,2]),('Wstar',[0,3])]:arms[arm]['gpus']=gpus
                    protocol=dict(gpu='0;1;2;3',gpus=[0,1,2,3],mode='parallel_processes')
                write(d/'time/result.json',dict(arms=arms));write(d/'time/unit.json',protocol)
            args=argparse.Namespace(cell='all',tag='phase2',phase=2,k=4)
            authorize=lambda c: AUTH if c['cell']==CELL else None
            with patch.object(experiment,'RUN',run), patch.object(experiment,'RESULTS',results), \
                 patch.object(experiment,'pilot_gate',return_value={}), \
                 patch.object(experiment,'audit_unit',return_value=dict(passed=True)), \
                 patch.object(waiver,'authorization',side_effect=authorize), \
                 contextlib.redirect_stdout(io.StringIO()):experiment.tables(args)
            def rows(name):
                with (results/('SM90X1_'+name+'.csv')).open() as stream:return list(csv.DictReader(stream))
            strict=rows('arms');extra=rows('uncertified_arms');summary=rows('summary');supplement=rows('summary_user_waived')
            self.assertEqual(len(strict),42)
            self.assertEqual(sum(r['status']=='measured' for r in strict),39)
            self.assertEqual(len(rows('figure_arms')),39)
            for row in [r for r in strict if r['cell']==CELL]:
                self.assertEqual(row['status'],'correctness_waived')
                self.assertEqual(row['c1_pass'],row['c2_pass'])
                self.assertEqual(row['c1_pass'],row['tpot_ms'])
                self.assertEqual(row['tpot_ms'],'')
            self.assertEqual(len(extra),3)
            for row in extra:
                self.assertEqual(row['status'],'measured_under_user_waiver')
                self.assertEqual(row['c1_pass'],row['c2_pass'])
                self.assertEqual(row['c1_pass'],'')
                self.assertEqual(row['performance_eligible'],'False')
                self.assertGreater(float(row['tpot_ms']),0)
                self.assertEqual(row['execution_mode'],'parallel_processes')
                self.assertEqual(row['gpu'],{'W':'0;1','L':'1;2','Wstar':'0;3'}[row['arm']])
            metrics=[r for r in summary if r['metric'] in ('X','Xprime','Y','Yprime')]
            for row in metrics:
                self.assertEqual(row['status'],'complete' if row['batch']=='1' else 'pending')
                if row['batch']!='1':self.assertEqual(row['value'],'')
            overall={r['metric']:r for r in supplement if r['batch']=='overall'}
            self.assertAlmostEqual(float(overall['X']['value']),1.5)
            self.assertAlmostEqual(float(overall['Xprime']['value']),1.2)
            self.assertEqual(overall['Y']['denominator'],'82')
            self.assertEqual(overall['Y']['value'],'0.0')
            self.assertTrue(all(r['correctness_waived_cells']==CELL and r['performance_eligible']=='False'
                                for r in overall.values()))
            complete=read(run/'main-complete.json')
            self.assertTrue(complete['complete']);self.assertFalse(complete['strict_complete'])
            self.assertEqual(complete['timed'],14);self.assertEqual(complete['measured'],13)
            by_cell={c['cell']:{r['arm']:r for r in strict+extra if r['cell']==c['cell'] and r['tpot_ms']} for c in cells}
            with patch.object(watch,'RUN',run):
                watch.check_aggregates(summary,cells,{k:v for k,v in by_cell.items() if k!=CELL})
                watch.check_aggregates(supplement,cells,by_cell,{CELL})
                bad=[dict(r) for r in supplement];bad[-1]['performance_eligible']='True'
                with self.assertRaises(ValueError):watch.check_aggregates(bad,cells,by_cell,{CELL})


if __name__=='__main__':
    unittest.main()
