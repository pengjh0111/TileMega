"""Idle queue admission and full-request timing contracts, with no NPU work."""
import json
from pathlib import Path
import tempfile
import unittest

from tilemega.bench import as2_queue as queue
from tilemega.bench.as1_run import devices_from_snapshot


class As2Queue(unittest.TestCase):
    def test_idle_requires_healthy_observation_not_empty_process_list(self):
        output='''| 0 | Ascend950PR | OK | 55 | 41 |
| | | 0000:D7:00.0 | 0 0 / 0 7667 / 131072 |
| 1 | Ascend950PR | Critical | 55 | 41 |
| | | 0000:55:00.0 | 0 0 / 0 5000 / 131072 |
| 0 | 123456 | python | 332 |
'''
        state=dict(exit_code=0,output=output)
        self.assertEqual(set(queue.idle_devices(state,16000)),{0})
        self.assertEqual(devices_from_snapshot(state)[0]['pci_bus_id'],'0000:d7:00.0')
        self.assertEqual(queue.idle_devices(dict(state,exit_code=1),0),{})
        self.assertEqual(queue.idle_devices(dict(state,output=output.replace('| 0 0 / 0','| 9 0 / 0')),0),{})
        self.assertEqual(queue.idle_devices(state,130000),{})

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
                            ('output','/tmp/no'),('argv','python -c pass')]:
            with self.assertRaises(ValueError):queue.validate_job(dict(job,**{field:value}))

    def test_device_init_failure_pauses_later_submissions(self):
        with tempfile.TemporaryDirectory(dir=queue.RUNS/'probes') as raw:
            folder=Path(raw);(folder/'stderr.log').write_text('aclrtSetDevice: 507033; TsdOpen failed')
            self.assertIsNotNone(queue.serious_failure(folder,dict(status='failed')))
            self.assertIsNotNone(queue.serious_failure(folder,dict(status='timeout')))
            (folder/'stderr.log').write_text('Unsupported graph configuration')
            self.assertIsNone(queue.serious_failure(folder,dict(status='failed')))

    def test_row_admission_uses_complete_baselines_and_retry_identity(self):
        job=dict(wait_for_baselines=True,model_id='model',batch=16)
        policy=dict(systems=['eager','engine'])
        state=dict(jobs={'model_b16_eager':dict(status='collected'),
                        'engine_retry':dict(status='failed',request=dict(cell_id='model_b16_engine'))})
        self.assertFalse(queue.ready(job,state,policy))
        state['jobs']['engine_retry']['status']='collected'
        self.assertTrue(queue.ready(job,state,policy))


if __name__=='__main__':unittest.main()
