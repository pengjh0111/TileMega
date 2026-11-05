import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90d1_common import sha,write
from sm90r6_local_theta import measured_choice,traffic_choice
from sm90r6_merge_theta import family_rows,process_key


class ThetaEvidence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.directory=self.root/'ab1/measurements-v1/model-B1/qkv'
        self.directory.mkdir(parents=True)
        self.cell=dict(input=dict(id='model-B1',model='model',batch=1),layers=2)
        options=[dict(geometry=dict(tile_n=n,tile_k=64,slot_bytes=n*128,warps_n=2,warps_k=2,
                                  slots=2,mma='wgmma'),tasks=tasks,traffic_bytes=traffic)
                 for n,tasks,traffic in [(64,500,100),(128,300,200),(256,280,300)]]
        self.domain=dict(family='qkv',grid=132,geometries=options)
        self.a0=dict(gemms=dict(qkv=options[0]['geometry']))
        self.records=[]
        for process in range(3):
            attempt=self.directory/f'p{process}'
            guard=attempt/'guard/guard_result.json';guard.parent.mkdir(parents=True);write(guard,dict(code=0))
            for index,option in enumerate(options):
                part=attempt/'trace'/f'part-{index}';part.mkdir(parents=True)
                raw=part/'stage_trace.tsv';raw.write_text('retained ledger\n')
                self.records.append(dict(family='qkv',geometry=option['geometry'],gpu=7,
                    reference_sha256='reference',guard_code=0,fresh_process=True,development=False,
                    input_state='A0 prompt-prefilled KV',process_id=f'{100+process}:{part}',
                    stage_ns=[(4-index)*1000,(4-index)*1000],guard=str(guard),raw=str(raw),raw_sha256=sha(raw)))
        write(self.directory/'result.json',dict(passed=True,development=False,same_card=True,processes=[0,1,2]))
        self.save()
        measured,evidence=measured_choice(options,self.records,2)
        traffic,rule=traffic_choice(options,132)
        self.choices={'AB1a':dict(qkv=dict(family='qkv',grid=132,**measured,**evidence)),
                      'AB1b':dict(qkv=dict(family='qkv',grid=132,**traffic,**rule))}

    def save(self):
        (self.directory/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))

    def run_family(self):return family_rows(self.root,self.cell,self.domain,self.a0,self.choices)

    def test_reports_selected_and_reference_durations_and_window_exception(self):
        rows=self.run_family()
        self.assertEqual([r['isolated_us'] for r in rows],[4,2,2])
        self.assertEqual(rows[-1]['task_window_exception'],'nearest_task_window_shell')
        self.assertEqual(rows[-1]['window_distance'],16)

    def test_missing_a0_requires_opt_in_and_keeps_a_blank_row(self):
        self.a0['gemms']['qkv']=dict(self.a0['gemms']['qkv'],tile_n=96)
        with self.assertRaisesRegex(ValueError,'absent'):self.run_family()
        rows=family_rows(self.root,self.cell,self.domain,self.a0,self.choices,True)
        self.assertEqual(rows[0]['measurement_status'],'missing_a0_geometry')
        self.assertEqual(rows[0]['isolated_us'],'')
        self.assertEqual(rows[0]['processes'],0)
        self.assertEqual(rows[1]['measurement_status'],'verified')

    def test_changed_raw_or_rejected_guard_is_not_reported(self):
        guard=Path(self.records[0]['guard']);write(guard,dict(code=75))
        with self.assertRaisesRegex(ValueError,'unaccepted'):self.run_family()
        write(guard,dict(code=0));Path(self.records[0]['raw']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'):self.run_family()

    def test_mixing_three_different_process_sets_is_rejected(self):
        self.records[0]['process_id']=self.records[0]['process_id'].replace('100:', '999:');self.save()
        with self.assertRaisesRegex(ValueError,'share three fresh'):self.run_family()

    def test_part_cannot_claim_another_guard_or_raw_ledger(self):
        row=dict(self.records[0]);row['guard']=self.records[-1]['guard']
        with self.assertRaisesRegex(ValueError,'bound'):process_key(row)
        row=dict(self.records[0]);row['raw']=self.records[1]['raw']
        with self.assertRaisesRegex(ValueError,'bound'):process_key(row)


if __name__=='__main__':unittest.main()
