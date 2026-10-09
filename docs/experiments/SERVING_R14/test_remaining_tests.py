"""CPU-only contracts for the isolated R14 continuation."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import make_remaining_tests as make
import phase_d_r14 as phase
import remaining_tests as driver

class RemainingTests(unittest.TestCase):
    def test_graph_and_independent_models(self):
        rows=make.definitions(24564);by={r['name']:r for r in rows}
        self.assertEqual(len(rows),len(by))
        seen=set()
        for row in rows:
            self.assertTrue(row['name'].startswith(make.PREFIX))
            self.assertTrue(set(row['after']+row['after_any'])<=seen)
            self.assertNotIn('--allow-shared-gpu',row['command'])
            self.assertGreater(row['timeout_s'],0)
            seen.add(row['name'])
        for model in ('llama','qwen3'):
            self.assertEqual(by[make.PREFIX+'build_'+model]['after'],[make.PREFIX+'calibrate'])
            deps=by[make.PREFIX+'register_'+model]['after']
            self.assertEqual(set(deps),{make.PREFIX+'smoke_'+model,make.PREFIX+'family_'+model})
        self.assertFalse(by[make.PREFIX+'analyze']['after'])
        self.assertTrue(by[make.PREFIX+'analyze']['after_any'])

    def test_full_matrix_and_guards(self):
        rows=make.definitions(24564)
        for cell in make.CELLS:
            paired=[r for r in rows if r['name'].startswith(make.PREFIX+'d2_'+cell+'_r')]
            self.assertEqual(len(paired),3)
            for row in paired:
                self.assertTrue(row['gpu']);self.assertGreater(row['needs_free_mib'],.85*24564)
            protocol=next(r for r in rows if r['name']==make.PREFIX+'protocol_'+cell)
            self.assertEqual(protocol['retry_args'],['--resume'])
        prepare=next(r for r in rows if r['name']==make.PREFIX+'prepare')
        self.assertEqual(prepare['command'][:2],['flock',make.LOCK])
        self.assertEqual(prepare['env']['TILEMEGA_GPU_LOCK_HELD'],'1')

    def test_no_retired_choices_or_run_directories(self):
        for row in make.definitions(24564):
            command=' '.join(row['command'])
            self.assertNotIn('_v2',command);self.assertNotIn('_v3',command)
            self.assertNotIn('phase_d_final_arms.json',command)
            if 'phase_d_r14.py' in command:
                self.assertIn('--run-prefix r14-remaining',command)
                self.assertIn('--tag _remaining',command)
                self.assertIn(str(make.INPUTS),command)

    def test_phase_parameters_preserve_defaults(self):
        self.assertEqual(phase.run_dir('llama'),make.ROOT/'runs/r14-llama')
        with patch.object(phase,'RUN_PREFIX','r14-remaining'),patch.object(phase,'CONFIG_DIR',Path('/tmp/config')),patch.object(phase,'TAG','_remaining'):
            self.assertEqual(phase.run_dir('qwen3'),make.ROOT/'runs/r14-remaining-qwen3')
            self.assertEqual(phase.config('qwen3'),Path('/tmp/config/qwen3_r14.json'))
            self.assertEqual(phase.evidence_path('D0'),make.HERE/'raw/D0_remaining')

    def test_status_shows_unstarted_queue_without_old_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q=root/'q';q.mkdir()
            (q/'queue_x.json').write_text(json.dumps([dict(name='fresh')]))
            output=subprocess.check_output([sys.executable,str(make.HERE/'status.py'),'--queue-dir',str(q),
                '--state-dir',str(root/'state'),'--queue','queue_x.json'],text=True)
            self.assertIn('alive=False',output);self.assertIn('pending=1',output);self.assertIn('fresh',output)

    def register_case(self,pg):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';run=root/'run';run.mkdir()
            def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            arm=dict(model='llama',model_path='/model',batch=1,decode='/control',prefill='/old_prefill',mode='L1',decode_loop=0)
            save(root/'phase_a_arms.json',dict(llama_B1=[dict(arm,label='R13D'),dict(arm,label='vllm')]))
            save(run/'plans.json',{'1':dict(prefill='/prefill',decode='/winner',decode_pg_choice=dict(selected=dict(mode='L1',loop=0)))})
            identity=dict(source=dict(source_digest='source',compiler_sha256='compiler'),plan=dict(pg=pg),artifact_id='artifact')
            with patch.object(driver,'HERE',root),patch.object(driver,'INPUTS',inputs),patch.object(driver,'baseline_arms',return_value={'llama_B1':arm}),patch.object(driver,'run_dir',return_value=run),patch.object(driver,'verify',return_value=identity),patch.object(driver,'sha',return_value='sha'):
                driver.register('llama')
            case=json.loads((inputs/'protocol_llama_B1.json').read_text())[0]
            registered=json.loads((inputs/'arms_llama_B1.json').read_text())['llama_B1']
            self.assertEqual({a['label'] for a in registered},{'vllm','R13D','baseline','R14F'})
            self.assertTrue(all(a['prefill']=='/prefill' for a in registered if a['label']!='vllm'))
            self.assertEqual(case['steps'],64)
            return case

    def test_protocol_tracks_actual_paged_path(self):
        case=self.register_case('pages')
        self.assertEqual([a['label'] for a in case['arms']],['L1_separate','L2_separate','L2_loop','L2_loop_no_phase'])
        self.assertEqual(case['arms'][-1]['env'],dict(TILEMEGA_KPHASE_MASK='0'))

    def test_nonpaged_protocol_has_no_fictitious_kphase_arm(self):
        case=self.register_case('l2')
        self.assertEqual([a['label'] for a in case['arms']],['L1_separate','L2_separate','L1_loop'])

    def test_missing_family_or_correctness_cannot_publish_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'scheduler_remaining';state.mkdir()
            (state/'state.json').write_text(json.dumps({'prior':dict(status='failed')}))
            with patch.object(driver,'HERE',root),patch.object(driver,'observations',side_effect=FileNotFoundError('missing')):
                driver.analyze()
            result=json.loads((root/'results/remaining_acceptance.json').read_text())
            self.assertEqual(result['status'],'requires_review');self.assertIsNone(result['selection'])
            self.assertEqual(len(result['missing']),6);self.assertEqual(result['failed_steps'],{'prior':'failed'})

if __name__=='__main__':unittest.main()
