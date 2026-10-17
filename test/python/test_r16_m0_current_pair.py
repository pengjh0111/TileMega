# SPDX-License-Identifier: BSD-3-Clause
"""A bounded M0 trial cannot release concurrency from stale peer metadata."""
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from advance_r16_m0_current_pair import Trial,write,sha,validate_barriers
from analyze_r16_m0 import analyze

class CurrentPairTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
    def test_phase_gate_requires_accepted_solo_plan_hash(self):
        trial=Trial.__new__(Trial);trial.out=self.out;write(self.out/'plan.json',dict(fixed=True));receipt=self.out/'receipt.json'
        write(self.out/'phase.json',dict(phase='concurrent',solo_accepted=5,plan_sha256=sha(self.out/'plan.json')))
        self.assertEqual(trial.phase_gate(4,receipt),2)
        write(self.out/'phase.json',dict(phase='concurrent',solo_accepted=6,plan_sha256=sha(self.out/'plan.json')))
        self.assertEqual(trial.phase_gate(4,receipt),0)
        write(self.out/'phase.json',dict(phase='abort'));self.assertEqual(trial.phase_gate(7,receipt),2)
    def test_explicit_clean_rows_do_not_scan_legacy_results(self):
        with patch('analyze_r16_m0.collect',side_effect=AssertionError('legacy collector called')):
            result=analyze(self.out,'current47_v1',rows=[],rejected=[])
        self.assertEqual(result['process_count'],0);self.assertFalse(result['passed'])
    def test_peer_pid_and_nonce_must_match_real_measurements(self):
        rows=[]
        for gpu,peer in [(4,7),(7,4)]:
            generation=[]
            for run in [1,2,3]:
                directory=self.out/f'run{run}';directory.mkdir(exist_ok=True)
                own=dict(gpu=str(gpu),pid=gpu*100,process_start=str(gpu),nonce=f'{gpu}-{run}',ready_against={str(peer):f'{peer}-{run}'})
                write(directory/f'gpu{gpu}.json',own)
                generation.append(dict(N=1024,run=run,m0_barrier=dict(directory=str(directory),participants=['4','7'],round=0,run=run,peer_pid=peer*100,nonce=f'{gpu}-{run}',peer_nonce=f'{peer}-{run}',peer_process_start=str(peer))))
            path=self.out/f'measurement{gpu}.json';write(path,dict(pid=gpu*100,generation_runs=generation));rows.append(dict(gpu=gpu,round=0,phase='concurrent',absolute_measurement=str(path)))
        validate_barriers(rows)
        path=Path(rows[0]['absolute_measurement']);value=json.loads(path.read_text());value['generation_runs'][0]['m0_barrier']['peer_pid']=123;write(path,value)
        with self.assertRaisesRegex(ValueError,'peer/nonce/round'):validate_barriers(rows)

if __name__=='__main__':unittest.main()
