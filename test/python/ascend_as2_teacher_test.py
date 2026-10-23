"""Teacher diagnostics must never enter natural-greedy/performance admission."""
import json
from pathlib import Path
import time
import unittest
from tilemega.bench.as2_compare_request import compare
from tilemega.bench.as2_teacher import checkpoints

RUNS=Path('/home/d50064638/junhuip/TileMega/runs/as2')


class TeacherScopeContract(unittest.TestCase):
    def test_teacher_result_is_rejected_by_greedy_comparator(self):
        root=RUNS/'search'/('teacher_scope_contract_'+str(time.time_ns()));root.mkdir()
        for name in ('binding','native','reference'):(root/name).mkdir()
        (root/'binding/request.json').write_text('{}\n')
        for mode,name in [('teacher_forced','native'),('natural_greedy','reference')]:
            (root/name/'result.json').write_text(json.dumps(dict(request_mode=mode))+'\n')
        with self.assertRaisesRegex(ValueError,'cannot admit a teacher-forced'):
            compare(root/'binding',root/'native',root/'reference',root/'comparison.json')
        self.assertFalse((root/'comparison.json').exists())

    def test_teacher_checkpoint_positions_cover_cache_boundaries(self):
        points=checkpoints(512)
        positions={63+n for n in points}
        self.assertTrue({63,64,65,127,128,129,255,256,511,512,575}.issubset(positions))
        self.assertEqual(points,sorted(set(points)))
        self.assertEqual(checkpoints(1),[0,1])


if __name__=='__main__':unittest.main()
