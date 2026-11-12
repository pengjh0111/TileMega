"""CPU contracts for frozen-reference, stage-boundary and payload experiments."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]


def module(name):
    spec=importlib.util.spec_from_file_location('mx8_'+name,ROOT/'experiments/mx8'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class PayloadTest(unittest.TestCase):
    def setUp(self):
        self.bw=module('bandwidth')
        self.config=dict(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,intermediate_size=64,vocab_size=128)

    def test_dense_tp_divides_weights_and_kv_once(self):
        full=self.bw.logical_payload(self.config,4)
        half=self.bw.logical_payload(self.config,4,tp=2)
        expected_parameters=2*(32*(4+2*2)*8+32*4*8+3*32*64)+32*128
        self.assertEqual(full['weight_bytes_per_rank'],2*expected_parameters)
        self.assertEqual(full['kv_bytes_per_rank'],4*4*2*2*8*576)
        self.assertEqual(half['logical_bytes_per_rank'],full['logical_bytes_per_rank']/2)

    def test_moe_uses_per_layer_distinct_experts(self):
        config=dict(self.config,num_experts=64,num_experts_per_tok=8)
        low=self.bw.logical_payload(config,4,active_per_layer=[8,16])
        high=self.bw.logical_payload(config,4,active_per_layer=[16,16])
        self.assertEqual(high['weight_bytes_per_rank']-low['weight_bytes_per_rank'],2*3*32*64*8)
        with self.assertRaises(ValueError):self.bw.logical_payload(config,4)
        with self.assertRaises(ValueError):self.bw.logical_payload(config,4,active_per_layer=[64,64])

    def test_inventory_reuses_four_systems_and_latest_tp8(self):
        rows=self.bw.frozen_timings()
        self.assertEqual(len(rows),220)
        self.assertEqual({r['system'] for r in rows},{'tilemega','vllm','sglang','torch.compile'})
        self.assertFalse(any(r['timing_remeasured'] for r in rows))
        row=next(r for r in rows if (r['model'],r['tp'],r['batch'],r['system'])==('qwen3_1_7b',8,16,'tilemega'))
        self.assertEqual(row['timing_source'],'docs/results/MX7_tp8_final_c550.csv')

    def test_vector_probe_counts_only_complete_load_groups(self):
        for vectors in (7,96,100,700,1024):
            sample=dict(bytes=vectors*16,grid=3,threads=4)
            actual=sum(8*16 for t in range(12) for start in range(t,vectors,96) if start+7*12<vectors)
            self.assertEqual(self.bw.vector_read_bytes(sample),actual)
            self.assertLessEqual(actual,sample['bytes'])


class SequenceTest(unittest.TestCase):
    def _cohort(self,missing_formal=False):
        collect=module('collect')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);queue=root/'queue';out=root/'out';out.mkdir()
            cell=root/'cell.json';cell.write_text(json.dumps(dict(model='test',batch=1,tpot_ms=1)))
            manifest=[]
            for mode in ('graph','hostlaunch'):
                names=[f'{mode}_check{i}' for i in range(5)];manifest.append(dict(qualifications=names))
                for name in names+[f'{mode}_timing{i}' for i in range(3)]+(['graph_smoke'] if mode=='graph' else []):
                    if missing_formal and name=='graph_check4':continue
                    job=queue/'jobs'/name;result_dir=job/'attempt/result';result_dir.mkdir(parents=True)
                    check='timing' not in name
                    definition=job/'definition.json';definition.write_text(json.dumps(dict(kind='ablation_qualify' if check else 'ablation_timing',command=['--cell',str(cell)])))
                    result=dict(variant=mode,tokens_sha256='smoke-artifact' if name=='graph_smoke' else 'selected-artifact',
                        single_loop_exact_1024=True,frozen_reference_exact=True,reference_tokens=1024,
                        tpot_ms=2,event_tpot_ms=2,ttft_ms=3,e2e_ms=2049,output_tokens_per_second=500)
                    if name.endswith('check0'):result['hf']=dict(pass_=True,**{'pass':True})
                    (result_dir/'RESULT.json').write_text(json.dumps(result))
                    (job/'ACCEPTED.json').write_text(json.dumps(dict(folder=str(result_dir.parent),definition=dict(path=str(definition)))))
            (root/'ABLATION_QUEUE.json').write_text(json.dumps(dict(arms=manifest)))
            tables={}
            with mock.patch.object(collect,'ROOT',root),mock.patch.object(collect.bw,'collect'),mock.patch.object(
                    collect.bw,'write_csv',side_effect=lambda path,rows:tables.update({path.name:rows})):
                collect.consolidate(queue,out)
            return tables['MX8_ab4c_c550.csv']

    def test_smoke_artifact_does_not_join_selected_five_process_cohort(self):
        rows=self._cohort()
        self.assertEqual({r['qualification_processes'] for r in rows},{5})
        self.assertEqual({r['status'] for r in rows},{'complete'})

    def test_smoke_cannot_replace_a_missing_formal_process(self):
        graph=next(r for r in self._cohort(missing_formal=True) if r['variant']=='AB4c_graph')
        self.assertEqual(graph['qualification_processes'],4)
        self.assertEqual(graph['status'],'pending')
        self.assertEqual(graph['tpot_ms'],'')

    def test_overlay_preserves_production_runtime_and_ticket_allocation(self):
        transform=module('build_sequence').transform_runtime
        for runtime in ('DecodeRuntimeV2.cuh','MoeRuntimeV3.cuh'):
            path=ROOT/'include/tilemega/Codegen/tasks/mx'/runtime
            original=path.read_text();changed=transform(original)
            self.assertIn('return Mx8Prepare(s);',changed)
            self.assertIn('Mx8Release(s);\n',changed)
            self.assertIn('return Mx8Launch(s, first, steps, iteration, stream);',changed)
            self.assertEqual(path.read_text(),original)
            self.assertEqual(changed.count('atomic'),original.count('atomic'))

    def test_pending_definitions_are_immutable(self):
        queue=module('queue')
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            definition=dict(kind='test',command=['true'],inputs=[])
            queue.submit(root,'same',definition);queue.submit(root,'same',definition)
            with self.assertRaises(ValueError):queue.submit(root,'same',dict(definition,command=['false']))
            self.assertEqual(json.loads((root/'jobs/same/definition.json').read_text()),definition)


if __name__=='__main__':unittest.main()
