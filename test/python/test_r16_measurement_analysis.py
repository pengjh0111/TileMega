"""CPU evidence checks: failed receipts and telemetry overhead cannot pass M0."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    import analyze_r16_m0 as m0
    import r16_measurement_validation as validation
    import summarize_r16 as summary
finally:
    sys.path.remove(str(SCRIPTS))


class MeasurementAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/'models/llama3_2_1b'; self.model.mkdir(parents=True)
        (self.model/'config.json').write_text('{"vocab_size":32}')
        self.state = {6:{},7:{}}
        self.receipts = []

    def receipt(self, phase, card, rnd, base, duration=10):
        phase_name = 'm0_solo' if phase == 'solo' else 'm0_concurrent_v4'
        name = f'{phase_name}_llama3_2_1b_vllm_B16_r{rnd}_gpu{card}'
        directory = self.root/'runs/r16/scheduler/raw'/name/'attempt1'
        out = directory/'measurements/B16'; out.mkdir(parents=True)
        own = dict(ownership_verified=False, acceptance='provisional_container_pid_namespace')
        (directory/'guard_result.json').write_text(json.dumps(dict(code=0,numa_concurrency_trial=phase=='concurrent',**own)))
        rows, samples = [], []
        for count in (1,1024):
            for run in range(4):
                seconds = 1 if count == 1 else duration
                token_file = f'tokens_N{count}_run{run}.json'
                (out/token_file).write_text(json.dumps([[7]*count for _ in range(16)]))
                rows.append(dict(N=count,run=run,warmup=run==0,wall_seconds=seconds,tokens_file=token_file))
                start = base+(run*20) if count == 1024 else base-100+run*3
                for suffix, clock in (('before',start-.2),('after',start+seconds+.2)):
                    samples.append(dict(time=clock,label=f'N{count}-run{run}-{suffix}',accepted=True,
                        unresolved_host_pids=[99999999],**own,telemetry={
                            'power.draw':'120','clocks.sm':'1800','clocks.mem':'2600','temperature.gpu':'40',
                            'clocks_throttle_reasons.active':'0x0'}))
        (out/'guard.jsonl').write_text('\n'.join(json.dumps(s) for s in samples)+'\n')
        value = dict(evidence='verified',model=str(self.model),batch=16,max_tokens=1024,prompt_length=64,
            model_config_sha256=hashlib.sha256((self.model/'config.json').read_bytes()).hexdigest(),
            prompt_ids_sha256='frozenprompt',system='vllm',round_index=rnd,gpu_index=card,pid=card*100+rnd,
            versions={'vllm':'0.11.0','torch':'2.8.0'},load_format='dummy',
            configuration={'seed':0,'dtype':'bfloat16','kv_cache_dtype':'auto'},
            generation_runs=rows,e2e_seconds=duration,ttft_seconds=1,tpot_seconds=(duration-1)/1023,
            output_tokens_per_second=16*1024/duration,guard_log=str(out/'guard.jsonl'),**own)
        path = out/'measurements.json';path.write_text(json.dumps(value))
        self.state[card][name] = dict(status='done',exit_code=0,out=str(directory.relative_to(self.root)))
        self.receipts.append(path)
        return path

    def matrix(self, delta=.005, concurrent=True):
        for card in (6,7):
            for rnd in range(3):
                self.receipt('solo',card,rnd,(1000 if card==6 else 2000)+100*rnd)
                self.receipt('concurrent',card,rnd,(3000 if card==6 or concurrent else 4000)+100*rnd+(card==7)*.1,10*(1+delta))
        self.persist()

    def persist(self):
        for card,state in self.state.items():
            path = self.root/f'runs/r16/scheduler/gpu{card}';path.mkdir(parents=True,exist_ok=True)
            (path/'state.json').write_text(json.dumps(state))

    def test_valid_legacy_solo_and_new_concurrent_pass_with_provisional_ownership(self):
        self.matrix()
        report = m0.analyze(self.root,'v4','m0_solo')
        self.assertTrue(report['passed'],report['errors'])
        self.assertEqual(len(report['interval_coverage']),18)
        self.assertFalse(report['ownership_verified'])
        self.assertFalse(report['weight_identity']['checkpoint_hash_verified'])
        self.assertEqual(report['weight_identity']['evidence'],'inferred')

    def test_no_actual_overlap_and_performance_regression_fail(self):
        self.matrix(delta=.02,concurrent=False)
        report = m0.analyze(self.root,'v4','m0_solo')
        self.assertFalse(report['passed'])
        self.assertTrue(any('exceeds 1%' in error for error in report['errors']))
        self.assertTrue(any('no guaranteed actual' in error for error in report['errors']))

    def test_outer_overlap_does_not_prove_actual_timing_overlap(self):
        left = validation.request_intervals({'generation_runs':[dict(N=1024,run=1,warmup=False,wall_seconds=4)]},[
            dict(label='N1024-run1-before',time=10),dict(label='N1024-run1-after',time=16)])[0]
        right = validation.request_intervals({'generation_runs':[dict(N=1024,run=1,warmup=False,wall_seconds=4)]},[
            dict(label='N1024-run1-before',time=14),dict(label='N1024-run1-after',time=20)])[0]
        self.assertLess(right['outer_start'],left['outer_end'])
        self.assertEqual(m0.overlap(left,right),0)
        # Enumerating admissible request starts checks the inner interval proof.
        for start in (10,10.5,11,11.5,12):
            self.assertGreaterEqual(left['inner_start'],start)
            self.assertLessEqual(left['inner_end'],start+4)

    def test_nan_wrong_tokens_thermal_and_failed_job_are_not_collected(self):
        self.matrix()
        for path, damage in zip(self.receipts[:3],('nan','tokens','thermal')):
            value=json.loads(path.read_text())
            if damage=='nan':value['e2e_seconds']=float('nan');path.write_text(json.dumps(value))
            elif damage=='tokens':
                (path.parent/'tokens_N1024_run1.json').write_text('[[7]]')
            else:
                log=path.parent/'guard.jsonl';samples=[json.loads(line) for line in log.read_text().splitlines()]
                samples[0]['telemetry']['clocks_throttle_reasons.active']='0x20'
                log.write_text('\n'.join(map(json.dumps,samples))+'\n')
        next(iter(self.state[7].values()))['status']='failed'
        self.persist()
        records,rejected=summary.collect(self.root)
        self.assertEqual(len(records),8)
        self.assertEqual(len(rejected),3)
        self.assertFalse(m0.analyze(self.root,'v4','m0_solo')['passed'])

    def test_dnn_ownership_is_derived_and_invalid_rows_are_excluded(self):
        name='dnn-resnet18-tensorrt-bf16-r0-gpu6'
        directory=self.root/'runs/r16/scheduler/raw'/name/'attempt1'
        out=directory/'measurements';out.mkdir(parents=True)
        own=dict(ownership_verified=False,acceptance='provisional_container_pid_namespace')
        (directory/'guard_result.json').write_text(json.dumps(dict(code=0,**own)))
        samples=[dict(time=100+i,label=label,accepted=True,unresolved_host_pids=[99999999],**own,
            telemetry={'power.draw':'120','clocks.sm':'1800','clocks.mem':'2600','temperature.gpu':'40',
                       'clocks_throttle_reasons.active':'0x0'})
            for i,label in enumerate(('before-initialization','before-measurement','after-measurement'))]
        (out/'guard.jsonl').write_text('\n'.join(map(json.dumps,samples))+'\n')
        value=dict(evidence='verified',measurement=dict(warmup=50,iterations=200),samples_ms=[1.0]*200,
            median_ms=1,p90_ms=1,guard_log=str(out/'guard.jsonl'),model='resnet18',system='tensorrt',
            batch=1,precision='bf16',round_index=0,gpu_index=6,exported_program_sha256='f'*64,random_weights=True)
        path=out/'measurements.json';path.write_text(json.dumps(value))
        self.state[6][name]=dict(status='done',exit_code=0,out=str(directory.relative_to(self.root)))
        self.persist()
        rows,rejected=summary.collect_dnn(self.root)
        self.assertEqual(len(rows),1);self.assertFalse(rejected)
        self.assertFalse(rows[0]['ownership_verified'])
        self.assertEqual(rows[0]['unresolved_host_pids'],'99999999')
        self.assertEqual(rows[0]['acceptance'],'provisional_container_pid_namespace')
        value['valid']=False;path.write_text(json.dumps(value))
        rows,rejected=summary.collect_dnn(self.root)
        self.assertFalse(rows);self.assertEqual(len(rejected),1)

    def test_versions_must_match_and_incomplete_is_not_passed(self):
        self.matrix()
        path=self.receipts[-1];value=json.loads(path.read_text());value['versions']['torch']='different'
        path.write_text(json.dumps(value))
        self.assertFalse(m0.analyze(self.root,'v4','m0_solo')['identity_consistent'])
        self.assertEqual(m0.analyze(self.root,'another','m0_solo')['status'],'incomplete')


if __name__ == '__main__':
    unittest.main()
