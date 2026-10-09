import json,unittest
from make_completion_repair import definitions
from make_phase0 import HERE

class CompletionRepair(unittest.TestCase):
    def test_isolated_names_and_old_queue_preserved(self):
        steps=definitions();names={r['name'] for r in steps}
        path=HERE/'queue/queue_phase_d_final.json'
        if not path.exists():path=HERE/'retired_queues/queue_phase_d_final.json'
        old=json.loads(path.read_text())
        self.assertFalse(names & {r['name'] for r in old})
        self.assertEqual(len(names),len(steps))

    def test_narrow_real_models_and_both_layouts(self):
        definitions();jobs=json.loads((HERE/'completion_builds.json').read_text())
        narrow=[j for j in jobs if '_tn' in j['label']]
        self.assertEqual(len(narrow),4)
        self.assertEqual({j['cell'] for j in narrow},{'llama_B1','qwen3_B1'})
        self.assertEqual({j['overrides']['nonpaged_weight_layout'] for j in narrow},{'row','tiled'})
        self.assertEqual({g['values']['tile_n'] for j in narrow for g in j['gemm_overrides']},{8,16})
        self.assertTrue(all(g['values']['impl']=='gemv' for j in narrow for g in j['gemm_overrides']))

    def test_lock_and_numerical_gates_precede_timing(self):
        steps=definitions();rows={r['name']:r for r in steps}
        for row in steps:
            if not row['gpu']:
                self.assertEqual(row['command'][:2],['flock','/root/r14_work/gpu.lock'])
                self.assertEqual(row['env']['TILEMEGA_GPU_LOCK_HELD'],'1')
        for r in range(3):
            dependencies=rows[f'R_complete_overhead_r{r}']['after']
            self.assertIn('R_complete_compare',dependencies)
            self.assertEqual(sum(n.startswith('R_complete_c1_') for n in dependencies),4)
        for model in ('llama','qwen3'):
            self.assertEqual(rows[f'D1_{model}_v3']['after'],['D0_v3'])
            self.assertEqual(rows[f'D1_family_{model}_v3']['after'],[f'D1_smoke_{model}_v3'])

if __name__=='__main__':unittest.main()
