"""Idle queue admission and full-request timing contracts, with no NPU work."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from tilemega.bench import as2_queue as queue
from tilemega.bench.as1_run import devices_from_snapshot
from tilemega.bench.as2_ablation import paired_interval,validate_qk_stages


class As2Queue(unittest.TestCase):
    def test_body_ablation_rejects_math_or_buffer_changes(self):
        original=tuple([0]*17+[13]+[0]*13+[6,0])
        candidate=original[:31]+(8,)+original[32:]
        validate_qk_stages([original],[candidate],'decode')
        for bad in (candidate[:24]+(128,)+candidate[25:],original):
            with self.assertRaises(ValueError):validate_qk_stages([original],[bad],'decode')
        with self.assertRaises(ValueError):validate_qk_stages([original],[candidate],'prefill')

    def test_confidence_uses_five_process_pairs_not_fifteen_requests(self):
        result=paired_interval([1.02,1.01,1.03,1.02,1.01])
        self.assertGreater(result['lower95'],1)
        self.assertLess(result['lower95'],result['geometric_mean'])
        self.assertGreater(result['upper95'],result['geometric_mean'])
        self.assertEqual(paired_interval([1.]*5)['lower95'],1)
        for ratios in ([1.02]*15,[1.02]*4,[1.,1.,1.,1.,float('nan')]):
            with self.assertRaises(ValueError):paired_interval(ratios)

    def test_idle_uses_process_table_not_health_or_utilization(self):
        output='''| 0 | Ascend950PR | OK | 55 | 41 |
| | | 0000:D7:00.0 | 0 0 / 0 7667 / 131072 |
| 1 | Ascend950PR | Alarm | 55 | 41 |
| | | 0000:55:00.0 | 100 0 / 0 5000 / 131072 |
| NPU ID | Process id | Process name | Process memory(MB) |
 +------------------+
| 0 | 123456 | python | 332 |
+------------------+
'''
        state=dict(exit_code=0,output=output)
        self.assertEqual(set(queue.idle_devices(state,16000)),{1})
        self.assertEqual(set(queue.idle_devices(state,16000,0)),{1})
        self.assertEqual(set(queue.idle_devices(dict(state,output=output.replace('+------------------+','+==================+')),16000)),{1})
        self.assertEqual(devices_from_snapshot(state)[0]['processes'],[123456])
        self.assertEqual(devices_from_snapshot(state)[0]['pci_bus_id'],'0000:d7:00.0')
        self.assertEqual(queue.idle_devices(dict(state,exit_code=1),0),{})
        for text in (output.split('| NPU ID | Process id')[0],output.rstrip().rsplit('\n',1)[0],
                     output.replace('| 0 | 123456 |','| unknown | 123456 |')):
            self.assertEqual(queue.idle_devices(dict(state,output=text),0),{})
        self.assertEqual(set(queue.idle_devices(dict(state,output=output.replace('| 0 | 123456 |','| 1 | 123456 |')),0)),{0})
        self.assertEqual(queue.idle_devices(state,130000),{})
        empty=output.split('| NPU ID | Process id')[0]+'No running processes found\n'
        self.assertEqual(set(queue.idle_devices(dict(state,output=empty),16000)),{0,1})

    def test_full_generation_timing_rejects_truncation_and_wrong_denominator(self):
        root=queue.RUNS/'probes';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as raw:
            folder=Path(raw);job=dict(output=raw,batch=16)
            rows=[dict(repetition=i,output_count=1024,decode_forwards=1023,
                       ttft_ms=50,e2e_ms=1073,tpot_ms=1) for i in (-1,0,1,2)]
            path=folder/'samples.jsonl'
            def save():path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save();self.assertEqual(queue.collect_samples(job)['tpot_ms'],1)
            rows[-1]['output_count']=1023;save()
            with self.assertRaisesRegex(ValueError,'generation'):queue.collect_samples(job)
            rows[-1]['output_count']=1024;rows[-1]['tpot_ms']=1023/1024;save()
            with self.assertRaisesRegex(ValueError,'boundaries'):queue.collect_samples(job)

    def test_fail_closed_resource_and_path_preflight(self):
        job=dict(id='sample',environment='as2',timeout_s=600,required_free_mib=100,
                 argv=['python','-c','pass'],output=str(queue.RUNS/'bench/test'))
        queue.validate_job(job)
        for field,value in [('timeout_s',0),('timeout_s',901),('physical_device',2),
                            ('output','/tmp/no'),('argv','python -c pass'),
                            ('expected_output_bytes',-1),('storage_family','invalid/request')]:
            with self.assertRaises(ValueError):queue.validate_job(dict(job,**{field:value}))
        disk=dict(generated_bytes=950,generated_soft_budget_bytes=1000)
        candidate=dict(job,expected_output_bytes=100)
        self.assertFalse(queue.output_budget_available(disk,candidate))
        self.assertTrue(queue.output_budget_available(dict(disk,generated_bytes=850),candidate))

    def test_output_parent_is_prepared_without_reusing_existing_results(self):
        with tempfile.TemporaryDirectory(dir=queue.RUNS/'probes') as raw:
            root=Path(raw);out=root/'new_parent/results/request'
            self.assertEqual(queue.prepare_output(dict(output=str(out))),out)
            self.assertTrue(out.parent.is_dir());self.assertFalse(out.exists())
            out.mkdir();saved=out/'evidence';saved.write_text('retained')
            with self.assertRaises(FileExistsError):queue.prepare_output(dict(output=str(out)))
            self.assertEqual(saved.read_text(),'retained')
            link=root/'alias';link.symlink_to(out.parent,target_is_directory=True)
            with self.assertRaises(ValueError):queue.prepare_output(dict(output=str(link/'other')))
            with self.assertRaises(ValueError):queue.prepare_output(dict(output='/tmp/as2-not-authorized'))

    def test_device_init_failure_pauses_later_submissions(self):
        with tempfile.TemporaryDirectory(dir=queue.RUNS/'probes') as raw:
            folder=Path(raw);(folder/'stderr.log').write_text('aclrtSetDevice: 507033; TsdOpen failed')
            self.assertIsNotNone(queue.serious_failure(folder,dict(status='failed')))
            self.assertIsNotNone(queue.serious_failure(folder,dict(status='timeout')))
            (folder/'stderr.log').write_text('Unsupported graph configuration')
            self.assertIsNone(queue.serious_failure(folder,dict(status='failed')))
            (folder/'after.json').write_text(json.dumps(dict(exit_code=0,output='| 1 | Ascend950PR | Alarm |\n| 0000:55:00.0 | 100 | 0 / 0 | 5000 / 131072 |')))
            self.assertIsNone(queue.serious_failure(folder,dict(status='passed',physical_device_id=1)))

    def test_process_arrival_is_requeued_only_when_no_child_started(self):
        item=dict(status='blocked-busy',job_directory='retained_attempt',receipt=dict(status='blocked-busy'))
        state=dict(jobs=dict(sample=item))
        self.assertTrue(queue.defer_busy_admission(state,'sample'))
        self.assertNotIn('sample',state['jobs'])
        self.assertEqual(state['admission_attempts'][0]['job_directory'],'retained_attempt')
        for receipt in (dict(status='blocked-busy',pid=123),dict(status='timeout'),dict(status='failed')):
            state=dict(jobs=dict(sample=dict(receipt=receipt)))
            self.assertFalse(queue.defer_busy_admission(state,'sample'))
            self.assertIn('sample',state['jobs']);self.assertNotIn('admission_attempts',state)

    def test_row_admission_uses_complete_baselines_and_retry_identity(self):
        job=dict(wait_for_baselines=True,model_id='model',batch=16)
        policy=dict(systems=['eager','engine'])
        state=dict(jobs={'model_b16_eager':dict(status='collected'),
                        'engine_retry':dict(status='failed',request=dict(cell_id='model_b16_engine'))})
        self.assertFalse(queue.ready(job,state,policy))
        state['jobs']['engine_retry']['status']='collected'
        self.assertTrue(queue.ready(job,state,policy))

    def test_timed_tilemega_outputs_match_prior_numerics_without_baseline_checks(self):
        with tempfile.TemporaryDirectory(dir=queue.RUNS/'probes') as raw:
            root=Path(raw);data=bytes(range(256))*16;sha=hashlib.sha256(data).hexdigest()
            proof=dict(status='pass',model_id='fixture',batch=1,isl=64,osl=1024,
                rows=[dict(greedy=dict(passed=True,native_sha256=sha,reference_sha256=sha))])
            p=root/'prior.json';p.write_text(json.dumps(proof))
            for i in (-1,0,1,2):
                folder=root/f'request_{i}';folder.mkdir();(folder/'tokens.bin').write_bytes(data)
            actual=dict(performance_only=True,priced_plans_executed=True,batch=1,isl=64,osl=1024,
                warmup=1,repetitions=3,host_kernel_launches=4096,binding_directory=str(root/'binding'))
            (root/'result.json').write_text(json.dumps(actual));job=dict(id='fixture',kind='tilemega',
                model_id='fixture',batch=1,output=str(root),prior_correctness=str(p),argv=['native','bench',str(root/'binding')])
            self.assertTrue(queue.audit_tilemega_tokens(job)['passed'])
            (root/'request_2/tokens.bin').write_bytes(data[:-1]+b'X')
            self.assertFalse(queue.audit_tilemega_tokens(job)['passed'])
            (root/'request_2/tokens.bin').write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError,'extent'):queue.audit_tilemega_tokens(job)
            with self.assertRaisesRegex(ValueError,'TileMega-only'):queue.audit_tilemega_tokens(dict(job,kind='baseline'))


if __name__=='__main__':unittest.main()
