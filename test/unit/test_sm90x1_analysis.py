"""CPU fixtures for own-time attribution, release ledgers and noisy selections."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import sm90x1_analysis as x


class AnalysisTest(unittest.TestCase):
    def test_latest_arriver_own_excludes_other_cta_wait(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            (d/'runtime_stages.tsv').write_text('stage\tkind_name\tname\n0\tkGemm\tl0.qkv.weight\n1\tkGemm\tlm_head.weight\n')
            with (d/'stage_trace.tsv').open('w') as f:
                writer=csv.writer(f,delimiter='\t')
                writer.writerow(['worker','iteration','past','stage','t_begin','t_tasks_end','t_release'])
                for past in range(567,583):
                    base=past*100000
                    for g in (0,1):
                        # CTA 0 runs longer; CTA 1 arrives last after a later group wait.
                        writer.writerow([0,past,past,g,base+g*30000,base+g*30000+10000,base+g*30000+25000])
                        writer.writerow([1,past,past,g,base+g*30000+19000,base+g*30000+21000,base+g*30000+26000])
            result=x.trace_process(d,False)
            self.assertEqual(result['own']['qkv'],2)
            self.assertEqual(result['step_us'],55)
            self.assertEqual([s['ledger_us'] for s in result['stages']],[25,30])

    def test_range_tie_prefers_reference(self):
        w=dict(variant='W',reference=True,own_us=100,own_us_range_pct=1)
        a=dict(variant='v1',reference=False,own_us=99,own_us_range_pct=0)
        b=dict(variant='v2',reference=False,own_us=95,own_us_range_pct=0)
        self.assertIs(x.criterion_order([a,w],'own_us')[0],w)
        self.assertIs(x.criterion_order([a,w,b],'own_us')[0],b)

    def test_reversal_excludes_noise_and_tau(self):
        rows=[dict(variant=str(i),own_us=10+i,step_us=100-5*i,
                   own_us_range_pct=0,step_us_range_pct=0) for i in range(3)]
        self.assertEqual(x.kendall(rows),-1)
        self.assertEqual(len(x.reversals(rows)),3)
        rows[1]['step_us']=rows[0]['step_us']*0.999
        self.assertEqual(len(x.reversals(rows)),2)


class ProtocolTest(unittest.TestCase):
    def test_unresolved_extra_owners_are_rejected(self):
        from sm90x1_hardware import audit_samples
        def row(phase, *owners):return dict(phase=phase, owners={str(p):100 for p in owners})
        clean=[row('preflight'),row('running',100),row('after')]
        self.assertTrue(audit_samples(clean)['passed'])
        # A transient second owner must reject even if it exits before the guard.
        extra=clean[:2]+[row('running',100,200)]+clean[2:]
        self.assertFalse(audit_samples(extra)['passed'])
        self.assertTrue(audit_samples(extra,hf=True)['passed'])
        switched=clean[:2]+[row('running',200)]+clean[2:]
        self.assertFalse(audit_samples(switched)['passed'])

    def test_frozen_cells_and_hashes(self):
        import hashlib
        import json
        import sm90x1
        rows=sm90x1.cells('all')
        self.assertEqual(len(rows),14)
        self.assertEqual(sum(c['pilot'] for c in rows),2)
        self.assertEqual(sum(5 if c['moe'] else 6 for c in rows),82)
        for c in rows:
            self.assertEqual(hashlib.sha256(sm90x1.coordinate_bytes(c['coordinates'])).hexdigest(),
                             c['w_coordinates_sha256'])
            self.assertEqual(c['class'],'machine' if c['moe'] else 'trial')

    def test_incomplete_gate_blocks_sweeps(self):
        import sm90x1_experiment as experiment
        from unittest.mock import patch
        with patch.object(experiment,'read',return_value=dict(stop=False,complete=False)):
            with self.assertRaises(ValueError):experiment.gate(1)
        with patch.object(experiment,'read',return_value=dict(stop=True,complete=True)):
            with self.assertRaises(ValueError):experiment.gate(1)

    def test_pending_report_keeps_every_cell(self):
        import argparse
        import contextlib
        import io
        import sm90x1_experiment as experiment
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'run').mkdir();(root/'results').mkdir()
            args=argparse.Namespace(cell='all',tag='phase1',phase=1,k=None)
            with patch.object(experiment,'RUN',root/'run'),patch.object(experiment,'RESULTS',root/'results'):
                with contextlib.redirect_stdout(io.StringIO()):experiment.tables(args)
            with (root/'results/SM90X1_arms.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),42)
            with (root/'results/SM90X1_summary.csv').open() as f:
                metrics=[r for r in csv.DictReader(f) if r['metric'] in ('X','Xprime','Y','Yprime')]
            self.assertEqual(len(metrics),12)
            self.assertTrue(all(r['value']=='' and r['status']=='pending' for r in metrics))

    def test_hardware_rejected_finals_do_not_complete_phase0(self):
        import argparse
        import contextlib
        import io
        import sm90x1 as base
        from sm90d1_common import write
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'docs/results').mkdir(parents=True)
            for c in base.cells('pilot'):
                d=root/'run/cells'/c['cell']/'phase0'
                write(d/'time/result.json',dict(arms=dict(W=dict(tpot_ms=c['accepted_tpot_ms']))))
                write(d/'corr/result.json',dict(c1_pass=True,c2_pass=True,**{'pass':True}))
                write(d/'W/binding.json',dict(trace=False))
            args=argparse.Namespace(tag='phase0')
            with patch.object(base,'ROOT',root),patch.object(base,'RUN',root/'run'), \
                 patch.object(base,'head',return_value='fixture'), \
                 patch.object(base,'audit_unit',return_value=dict(passed=False)):
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(base.tables(args),3)
            self.assertFalse(base.read(root/'run/phase0.json')['complete'])

    def test_complete_arm_ratios_and_selection_denominator(self):
        import argparse
        import contextlib
        import io
        import json
        import sm90x1 as base
        import sm90x1_experiment as experiment
        from sm90d1_common import write
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';results=root/'results';results.mkdir()
            for c in base.cells('all'):
                d=run/'cells'/c['cell']/('phase1' if c['pilot'] else 'phase2')
                coordinates=d/'coordinates.json';write(coordinates,c['coordinates'])
                plan=[dict(arm=arm,id='W',coordinates=str(coordinates),coordinates_sha256='fixture')
                      for arm in ['W','L','Wstar']]
                write(d/'arm-plan.json',plan)
                write(d/'W/binding.json',dict(decode_sha256='fixture',head='fixture',trace=False))
                write(d/'W/corr/result.json',dict(c1_pass=True,c2_pass=True,**{'pass':True}))
                arms={arm:dict(tpot_ms=ms,range_pct=0.1,processes=c['processes'])
                      for arm,ms in [('W',10),('L',12),('Wstar',8)]}
                write(d/'time/result.json',dict(arms=arms))
                write(d/'time/unit.json',dict(gpu=0))
            args=argparse.Namespace(cell='all',tag='phase2',phase=2,k=None)
            with patch.object(experiment,'RUN',run),patch.object(experiment,'RESULTS',results), \
                 patch.object(experiment,'pilot_gate',return_value=dict(continue_=True)), \
                 patch.object(experiment,'audit_unit',return_value=dict(passed=True)):
                with contextlib.redirect_stdout(io.StringIO()):experiment.tables(args)
            with (results/'SM90X1_summary.csv').open() as f:
                summary=list(csv.DictReader(f))
            overall={r['metric']:r for r in summary if r['batch']=='overall' and not r['family']}
            self.assertAlmostEqual(float(overall['X']['value']),1.5)
            self.assertAlmostEqual(float(overall['Xprime']['value']),1.2)
            self.assertEqual(int(overall['Y']['denominator']),82)
            self.assertEqual(float(overall['Y']['value']),0)
            self.assertTrue(json.loads((run/'main-complete.json').read_text())['complete'])


if __name__=='__main__':
    unittest.main()
