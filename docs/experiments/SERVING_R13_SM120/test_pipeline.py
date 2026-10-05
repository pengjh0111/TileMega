import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline
import protocol
import report


class PipelineTests(unittest.TestCase):
    def queue(self):
        with patch.object(pipeline,'read',return_value={}):return pipeline.make_queue()

    def test_complete_domain_is_registered(self):
        rows=self.queue()
        self.assertEqual(len(rows),74)
        self.assertEqual(len({row['name'] for row in rows}),len(rows))
        self.assertEqual(sum(row['name'].startswith('E4') and '_r0_' in row['name'] for row in rows),16)
        self.assertEqual(sum(row['name'].startswith('E5d') for row in rows),2)
        self.assertEqual(len(pipeline.MATRICES['E4c']),9)

    def test_gpu_tests_follow_automatic_dependencies(self):
        rows={row['name']:row for row in self.queue()}
        self.assertIn('E3_catalog_r5',rows['E2c_bulk_r5']['after'])
        self.assertEqual(set(rows['E4a_llama_B1_r0_r5']['after_any']),{f'E2c_{group}_r5' for group in pipeline.GROUPS})
        self.assertIn('E4d_qwen3_B16_r2_r5',rows['E4_canary_r5']['after_any'])
        self.assertEqual(rows['E4f_llama_B1_r5']['after_any'],['E4_canary_r5'])
        self.assertIn('E5c_llama_B1_r5',rows['E5d_llama_B16_r5']['after_any'])
        self.assertIn('E3_codegen_r5',rows['E6_collect_r5']['after_any'])
        self.assertTrue(all(row['gpu'] for row in rows.values() if row['name'].startswith(('E2c','E4','E5'))))

    def test_all_dependencies_resolve(self):
        rows=self.queue()
        names={row['name'] for row in rows}|{'E3_fixed_r4','E2b_smoke_r4','E3_R13F_llama_r4','E3_R13F_qwen3_r4'}
        for row in rows:
            self.assertTrue(set(row['after']+row['after_any'])<=names)
        seen=set()
        finished=set()
        def visit(name):
            if name in finished:return
            self.assertNotIn(name,seen,'dependency cycle')
            seen.add(name)
            row=next((row for row in rows if row['name']==name),None)
            if row:
                for dep in row['after']+row['after_any']:visit(dep)
            seen.remove(name)
            finished.add(name)
        visit('E6_collect_r5')

    def test_pdl_and_executor_mapping(self):
        self.assertEqual(pipeline.arm_recipe('B0p')[0],'N-R12b-120-pdl0')
        self.assertEqual(pipeline.arm_recipe('NL2gp1')[0],'N-R12b-120-pdl1')
        self.assertEqual(pipeline.arm_recipe('NL2g')[3],{'TILEMEGA_PLACEMENT_ABLATION':'grid_stride'})
        self.assertEqual(pipeline.arm_recipe('NL2r')[3],{'TILEMEGA_PLACEMENT_ABLATION':'rotate'})
        self.assertEqual(pipeline.arm_recipe('PR_L2l')[1:3],('L2',True))

    def test_completed_smoke_mode_survives_an_unrelated_mode_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            location=Path(directory)
            binary=location/'plan.so';binary.write_bytes(b'fixture')
            pipeline.write(location/'llama_r13_sm120_r3.json',{'model':{'path':str(location)}})
            row={'exit_code':0,'so':str(binary)}
            rows={('llama_B16','N-R12b-120'):row,('llama_B16','PF-120'):row}
            smoke={('llama_B16','N-R12b-120'):{'report':{'error':'later loop failed','arms':{'L2_separate':{}},'pass':False}}}
            with patch.object(pipeline,'HERE',location):
                self.assertTrue(pipeline.frozen_arm('NL2e','llama_B16',rows,smoke)['available'])
                self.assertFalse(pipeline.frozen_arm('B0l','llama_B16',rows,smoke)['available'])

    def test_numeric_protocol_failure_is_preserved_without_cancelling_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'plan.so';binary.write_bytes(b'fixture')
            arm=dict(available=True,kind='tm',prefill=str(binary),decode=str(binary),
                     binaries={phase:{'sha256':pipeline.sha(binary)} for phase in ('prefill','decode')})
            status=dict(complete=True,covered_labels=['B0p1'],pass_rate=.98,passed=49)
            with patch.object(pipeline,'original_arm',return_value=arm),patch.object(pipeline,'read',return_value=status):
                result=pipeline.runnable('llama_B16','B0p1')
                self.assertTrue(result['available'])
                self.assertFalse(result['protocol_correctness_pass'])
                self.assertFalse(pipeline.runnable('llama_B16','B0p')['available'])

    def test_trace_domain_matches_registration(self):
        self.assertEqual(len(pipeline.trace_requests('E5a','llama_B1')),2)
        self.assertEqual(len(pipeline.trace_requests('E5b','llama_B1')),2)
        self.assertEqual(len(pipeline.trace_requests('E5c','llama_B1')),5)
        self.assertEqual(len(pipeline.trace_requests('E5d','llama_B16')),3)
        self.assertEqual(pipeline.trace_requests('E5c','llama_B1')[1][1],'N-R12b-120-pdl1-step')

    def test_protocol_requires_50_unique_pids(self):
        rows=[dict(execution_complete=True,pid=number,passed=True) for number in range(50)]
        self.assertTrue(protocol.aggregate(rows)['complete'])
        self.assertEqual(protocol.aggregate(rows)['pass_rate'],1)
        rows[-1]['pid']=0
        self.assertFalse(protocol.aggregate(rows)['complete'])
        rows[-1]['pid']=49;rows[-1]['passed']=False
        value=protocol.aggregate(rows)
        self.assertTrue(value['complete'])
        self.assertEqual(value['passed'],49)
        self.assertEqual(value['pass_rate'],.98)

    def test_mechanisms_pair_only_same_group_and_rounds(self):
        rows=[dict(matrix='E4c',cell='llama_B1',arm=arm,round=number,tpot_s=value)
              for number in range(3) for arm,value in [('B0-noev',2),('B0p',1.98)]]
        rows.append(dict(matrix='E4a',cell='llama_B1',arm='B0p',round=0,tpot_s=.01))
        value=report.paired(rows,'E4c','llama_B1','B0p','B0-noev')
        self.assertTrue(value['complete'])
        self.assertAlmostEqual(value['relative'],-.01)
        rows=[row for row in rows if not(row['matrix']=='E4c' and row['arm']=='B0p' and row['round']==2)]
        value=report.paired(rows,'E4c','llama_B1','B0p','B0-noev')
        self.assertFalse(value['complete'])
        self.assertIsNone(value['relative'])

    def test_anchor_uses_explicit_local_policy(self):
        module_path=pipeline.FRAME/'anchor.py'
        spec=importlib.util.spec_from_file_location('anchor_test',module_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json';path.write_text('{}')
            with patch.dict('os.environ',{'TILEMEGA_MEASUREMENT_POLICY':str(path)}):
                self.assertEqual(module.measurement_policy(Path(directory)),path)
            with patch.dict('os.environ',{'TILEMEGA_MEASUREMENT_POLICY':str(path)+'missing'}):
                with self.assertRaises(RuntimeError):module.measurement_policy(Path(directory))

    def test_final_sm89_effect_needs_exactly_three_paired_rounds(self):
        rows=[dict(matrix='B2',cell='llama_B1',arm=arm,round=number,tpot_s=value)
              for number in range(3) for arm,value in [('B0',2),('NL2g',2.1)]]
        value,note=report.old_effect(rows,'B2','llama_B1','NL2g','B0')
        self.assertAlmostEqual(value,.05)
        self.assertIn('final sm89',note)
        rows=[row for row in rows if not(row['arm']=='NL2g' and row['round']==2)]
        self.assertIsNone(report.old_effect(rows,'B2','llama_B1','NL2g','B0')[0])

    def test_speedup_matches_r13_paired_e2e_and_keeps_tpot_separate(self):
        rows=[dict(matrix='E4a',cell='llama_B1',arm=arm,round=number,tpot_s=value,e2e_s=elapsed)
              for number in range(3) for arm,value,elapsed in [('B0',1,2),('vllm',1.1,2.4)]]
        pair=report.paired(rows,'E4a','llama_B1','B0','vllm')
        self.assertAlmostEqual(pair['speedup_tpot'],1.1)
        self.assertAlmostEqual(pair['speedup_e2e'],1.2)

    def test_final_configuration_ignores_paths_but_not_retained_features(self):
        from acceptance import configuration_matches
        local=dict(features={'pdl':'auto'},solver={'jobs':3,'prefill_pins':{'1':'local'}})
        remote=dict(features={'pdl':'auto'},solver={'jobs':3,'prefill_pins':{'1':'sm89'}})
        self.assertTrue(configuration_matches(local,remote))
        remote['features']['pdl']='off'
        self.assertFalse(configuration_matches(local,remote))

    def test_validation_replay_preserves_and_checks_the_original_round(self):
        with tempfile.TemporaryDirectory() as directory:
            location=Path(directory)
            original=location/'original.json';pipeline.write(original,{'arms':{}})
            replay=location/'replayed.json';pipeline.write(replay,{'arms':{'vllm':{'exit_code':0}}})
            pipeline.write(location/'raw/acceptance_03/anchor_replacements.json',{
                'E4a:llama_B1:0':{'path':str(replay),'original_path':str(original),'original_sha256':pipeline.sha(original)}})
            with patch.object(pipeline,'HERE',location):
                self.assertEqual(pipeline.round_path('E4a','llama_B1',0),replay)
                pipeline.write(original,{'changed':True})
                with self.assertRaises(RuntimeError):pipeline.round_path('E4a','llama_B1',0)

    def test_negative_boundary_residual_is_not_a_latency_measurement(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            location=Path(directory)
            pipeline.write(location/'start.json',{'prompt_sha256':'fixture'})
            pipeline.write(location/'predictions_sm120.json',{'predictions':[{'id':'step_boundary','range':[5,20]}]})
            pipeline.write(location/'raw/E2a_MB-1e_r3/loadbench.json',{'points':[
                {'suite':'MB-1e','mode':'separate','median_ms':1.5,'overhead_ns_per_step':-12244}]})
            with patch.object(report,'HERE',location),patch.object(pipeline,'HERE',location):report.collect()
            with (location/'results/S10.tsv').open() as stream:row=next(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(row['matches'],'')
            self.assertIsNone(__import__('json').loads(row['observation'])['value'])

    def test_missing_results_remain_missing_in_all_ten_tables(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            location=Path(directory)
            (location/'raw').mkdir()
            pipeline.write(location/'start.json',{'prompt_sha256':'fixture'})
            pipeline.write(location/'predictions_sm120.json',{'predictions':[{'id':'step_boundary','range':[5,20]}]})
            with patch.object(report,'HERE',location),patch.object(pipeline,'HERE',location):
                self.assertEqual(report.collect(),0)
            self.assertTrue(all((location/f'results/S{number}.tsv').exists() for number in range(1,11)))
            with (location/'results/S10.tsv').open() as stream:
                records=list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['matches'],'')
            self.assertIn('unavailable',records[0]['status'])

    def test_post_exit_transient_is_not_called_a_persistent_device_fault(self):
        import sys
        import device_health
        sys.path.insert(0,str(pipeline.FRAME))
        import gpu_guard
        with tempfile.TemporaryDirectory() as directory:
            location=Path(directory)
            pipeline.write(location/'guard_policy.json',{'max_util_pct':5,'samples':6,'interval_s':5})
            hot={'utilization_pct':100,'owners':{}}
            cold={'utilization_pct':0,'owners':{}}
            with patch.object(device_health,'HERE',location),patch.object(device_health.time,'sleep') as sleep:
                with patch.object(gpu_guard,'gpu',side_effect=[hot,cold]):
                    self.assertEqual(device_health.check(),cold)
                    self.assertEqual(sleep.call_count,1)
                with patch.object(gpu_guard,'gpu',return_value=hot):
                    with self.assertRaises(RuntimeError):device_health.check()


if __name__=='__main__':
    unittest.main()
