#!/usr/bin/env python3
"""Reject partial/mixed checkpoints and invalid timing evidence without CUDA."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dnn2_scoreboard', ROOT/'scripts/dnn2_scoreboard.py')
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


class ScoreboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def baseline(self):
        return [dict(model=m, batch=str(b), system='torchcompile', precision='bf16',
                     status='pass', arm=f'arm{i}', median_ms=str(i+1))
                for m, b in sb.CELLS for i in range(4)]

    def records(self, head='a'):
        return [dict(head=head, promoted_change='CP-0', model=m, batch=b, gpu=0,
                     median_ms=2., p90_ms=3., process_range_ms=.01,
                     correctness='pass:5/5', source='immutable-receipt') for m, b in sb.CELLS]

    def scan(self, head='a'):
        return dict(head=head, promoted_change='CP-0', baselines=sb.baseline_lookup(self.baseline()))

    def windows(self):
        return [dict(label='before-initialization', time=1., pids=[],
                     telemetry={'memory.used':'0'}, accepted=True),
                dict(label='before-measurement', time=2., pids=[71],
                     telemetry={'memory.used':'100'}, accepted=True),
                dict(label='after-measurement', time=3., pids=[71],
                     telemetry={'memory.used':'100'}, accepted=True)]

    def window(self, samples, outer=()):
        f=self.root/'guard.jsonl'
        f.write_text(''.join(json.dumps(x)+'\n' for x in samples))
        return sb.timing_window(dict(guard_log=str(f)), outer)

    def test_exact_grid_and_primary_baseline(self):
        sb.check_grid([dict(model=m,batch=b) for m,b in sb.CELLS])
        with self.assertRaises(ValueError): sb.check_grid(self.records()[:-1])
        bad=self.records();bad[-1]=bad[0]
        with self.assertRaises(ValueError): sb.check_grid(bad)
        rows=self.baseline();rows.append(dict(rows[0],precision='fp16',median_ms='.001'))
        self.assertEqual(sb.baseline_lookup(rows)['resnet18_b1']['ms'],1.)
        with self.assertRaises(ValueError): sb.baseline_lookup(rows[1:])

    def test_publish_only_complete_same_head(self):
        out=self.root/'scoreboard.csv';records=self.records()
        for bad in [records[:-1], [dict(r,head='b') if i==1 else r for i,r in enumerate(records)]]:
            with self.assertRaises(ValueError): sb.publish_rows(self.scan(),bad,out)
            self.assertFalse(out.exists())
        sb.publish_rows(self.scan(),records,out)
        first=out.read_bytes();self.assertEqual(len(list(csv.DictReader(io.StringIO(out.read_text())))),10)
        self.assertNotIn(b'\r',first)
        with self.assertRaises(ValueError): sb.publish_rows(self.scan(),records,out)
        self.assertEqual(out.read_bytes(),first)
        sb.publish_rows(self.scan('b'),self.records('b'),out)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(out.read_text())))),20)

    def test_freeze_rejects_head_or_source_changes(self):
        from unittest.mock import patch
        source=self.root/'source';source.mkdir();payload=source/'forward.cuh';payload.write_text('fixed')
        compiler=self.root/'compiler';compiler.write_text('fixedcompiler')
        scan=dict(head='a',source_root=str(source),compiler_sha256=sb.sha(compiler),
                  source_snapshot=dict(source_digest='digest',source_files={'forward.cuh':sb.sha(payload)}),
                  files={str(compiler):sb.sha(compiler)})
        f=self.root/'freeze.json';sb.write(f,dict(frozen=True,head='a',source_digest='digest',compiler_sha256=sb.sha(compiler)))
        with patch.object(sb,'git',return_value='a'):sb.require_freeze(scan,f)
        with patch.object(sb,'git',return_value='b'),self.assertRaises(ValueError):sb.require_freeze(scan,f)
        scan['out']=str(self.root);manifest=self.root/'scan.json';sb.write(manifest,scan)
        freeze=sb.read(f);freeze['scan_sha256']=sb.sha(manifest);sb.write(f,freeze)
        with patch.object(sb,'git',return_value='a'):sb.require_freeze(scan,f)
        sb.write(manifest,dict(scan,baselines='modified'))
        with patch.object(sb,'git',return_value='a'),self.assertRaises(ValueError):sb.require_freeze(scan,f)
        sb.write(manifest,scan)
        payload.write_text('newkernel')
        with patch.object(sb,'git',return_value='a'),self.assertRaises(ValueError):sb.require_freeze(scan,f)

    def test_initialization_and_inner_window(self):
        self.assertEqual(self.window(self.windows())['outer_samples'],0)
        for change in ['initial_pid','initial_memory','end_pid','intermediate_owner','missing_boundary']:
            samples=self.windows()
            if change=='initial_pid':samples[0]['pids']=[8]
            if change=='initial_memory':samples[0]['telemetry']['memory.used']='4'
            if change=='end_pid':samples[-1]['pids']=[72]
            if change=='intermediate_owner':samples.insert(2,dict(samples[1],time=2.5,pids=[71,72]))
            if change=='missing_boundary':samples.pop()
            with self.subTest(change=change), self.assertRaises(ValueError):self.window(samples)

    def test_outer_window_evidence_is_not_fabricated(self):
        result=self.window(self.windows(),[dict(time=2.5,owners={'71':100})])
        self.assertEqual(result['outer_samples'],1)
        self.assertEqual(result['outer_observation'],'sampled')
        with self.assertRaises(ValueError):self.window(self.windows(),[dict(time=2.5,owners={'71':100,'72':4})])
        # A short event window can have no periodic outer sample; disclose that
        # limitation while still requiring both inner boundary observations.
        self.assertIn('no outer sample',self.window(self.windows())['outer_observation'])

    def test_memory_or_owner_resets_the_consecutive_idle_boundary(self):
        class Clock:
            def __init__(self):self.t=0.
            def now(self):return self.t
            def sleep(self,seconds):self.t+=seconds
        cases=[([0,4,0,1],4,'empty_observed'),([0,4,4,4,4],5,'timeout')]
        for memories,count,status in cases:
            clock=Clock();rows=iter(dict(owners={},used_mib=x) for x in memories)
            d=sb.idle_boundary({},timeout_s=1.,interval_s=.25,observe=lambda _:next(rows),
                               clock=clock.now,sleep=clock.sleep)
            self.assertEqual(len(d['samples']),count);self.assertEqual(d['status'],status)
            if status=='empty_observed':self.assertEqual(sb.final_boundary(d),{'owners':{},'used_mib':1})
            else:
                with self.assertRaises(sb.ResourceRejection):sb.final_boundary(d)
        clock=Clock();rows=iter([dict(owners={},used_mib=0),dict(owners={99:1},used_mib=1),
                               dict(owners={},used_mib=0),dict(owners={},used_mib=0)])
        d=sb.idle_boundary({},observe=lambda _:next(rows),clock=clock.now,sleep=clock.sleep)
        self.assertEqual(len(d['samples']),4)
        # Reject the historical owner-only [0,4] result; do not look backward.
        d['samples'][-1]['telemetry']['used_mib']=4
        with self.assertRaises(sb.ResourceRejection):sb.final_boundary(d)

    def fixture(self):
        inputs=self.root/'inputs';inputs.write_bytes(b'fixedinput')
        cell=dict(model='resnet18',batch=1,inputs=str(inputs))
        scan=dict(head='a',_sha256='scan',promoted_change='CP-0')
        session=dict(head='a',scan_sha256='scan',cell=dict(cell,gpu=0),artifact_id='artifact',
                     library_sha256='binary',status='complete_pending_outer_audit',correctness=[],timing=[])
        for role,count in [('correctness',5),('timing',3)]:
            for i in range(count):
                pid=i+10+(10 if role=='timing' else 0)
                value=dict(artifact_id='artifact',pid=pid,batch=1,model='resnet18',passed=True,
                           library_sha256='binary',input_tensors_sha256=sb.sha(inputs),metrics=[dict(passed=True)])
                if role=='timing':
                    log=self.root/f'inner{i}.jsonl'
                    log.write_text(''.join(json.dumps(x)+'\n' for x in self.windows()))
                    value.update(selected_source_head='a',round_index=i,
                        identity=dict(source=dict(head='a'),binary_sha256='binary'),
                        input_sha256=sb.sha(inputs),gpu_index=0,protocol='P-DNN',
                        measurement=dict(warmup=50,iterations=200),samples_ms=[1.]*200,
                        median_ms=1.,p90_ms=1.,guard_log=str(log))
                p=self.root/f'{role}{i}.json';sb.write(p,value)
                boundary=dict(owners={},used_mib=0)
                drain=dict(status='empty_observed',samples=[dict(telemetry=boundary),dict(telemetry=boundary)])
                session[role].append(dict(pid=pid,returncode=0,receipt=str(p),receipt_sha256=sb.sha(p),
                    before=boundary,after=boundary,drain=drain,before_drain=drain))
        guard=self.root/'guard_result.json';sb.write(guard,dict(code=0))
        guard.with_name('guard.jsonl').write_text('')
        result=self.root/'result.json';sb.write(result,session)
        return scan,cell,result,guard

    def test_fresh_process_protocol_and_head_binding(self):
        scan,cell,result,guard=self.fixture()
        record=sb.collect_cell(scan,cell,result,guard)
        self.assertEqual(record['median_ms'],1.)
        session=sb.read(result)
        for mutation in ['head','pid','samples','statistics','source_hash']:
            changed=copy.deepcopy(session)
            if mutation=='head':changed['head']='b'
            elif mutation=='pid':changed['timing'][0]['pid']=changed['correctness'][0]['pid']
            else:
                p=Path(changed['timing'][0]['receipt']);value=sb.read(p)
                if mutation=='samples':value['samples_ms']=value['samples_ms'][:-1]
                if mutation=='statistics':value['median_ms']=.001
                if mutation=='source_hash':value['selected_source_head']='b'
                sb.write(p,value);changed['timing'][0]['receipt_sha256']=sb.sha(p)
            sb.write(result,changed)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):sb.collect_cell(scan,cell,result,guard)
            # Restore all fixture files before trying an independent defect.
            scan,cell,result,guard=self.fixture();session=sb.read(result)


if __name__=='__main__':unittest.main()
