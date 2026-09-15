#!/usr/bin/env python3
"""Regression cases for lost local edges and overlapping CTA waits."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trace_analyze', REPO / 'docs/experiments/TRACE_V2/analyze.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def table(path, columns, rows):
    with path.open('w') as f:
        w = csv.writer(f, delimiter='\t'); w.writerow(columns); w.writerows(rows)


class RebuildTest(unittest.TestCase):
    def test_local_and_lifted_dependencies_survive_without_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            source = d / 'model.cu'
            source.write_text('constexpr StageDependency kDependencies0[] = {\n'
                '  {0u, 1u, StageDependency::Map::kIdentity, 1u, 1, 0, 1u},\n'
                '  {1u, 2u, StageDependency::Map::kIdentity, 1u, 1, 0, 1u},\n};\n')
            table(d / 'meta.tsv', ['key', 'value'], [('kappa',1),('grid',2),('l2_ms',.000065)])
            columns = ('slot worker stage logical_task smid wait_begin ready run_begin run_end publish_end '
                       'wait_begin_idx wait_count dependency_begin dependency_count run_begin_clk run_end_clk').split()
            table(d / 'slots.tsv', columns, [
                (0,0,0,0,0,0,5,10,20,25,0,0,0,0,100,110),
                (1,0,1,0,0,25,25,30,40,45,0,0,0,1,120,130),
                (2,1,2,0,1,0,45,50,60,65,0,0,1,1,200,210)])
            table(d / 'waits.tsv', ['wait_index','producer','group','event_index'], [])
            table(d / 'events.tsv', ['event_index','stage','group','fanin','publish_ns'], [])
            r = trace.analyze(d, source, 1)
            self.assertEqual(r['cp_corrected_ns'],30)
            self.assertEqual(r['cp_corrected_nodes'],3)
            self.assertEqual(r['cp_lb_nosync_ns_legacy'],10)
            self.assertEqual(r['dag_same_worker_edges'],1)
            self.assertEqual(r['dag_cross_worker_edges'],1)
            self.assertEqual(r['cp_reconstructed_ns'],55)
            split = [r['cp_split_'+n+'_ns'] for n in ('task','wait','prerun_barrier','publish','gap')]
            self.assertEqual(sum(split),55)
            self.assertGreaterEqual(min(split),0)
            self.assertLessEqual(r['cp_split_wait_ns'],r['kernel_span_ns'])
            self.assertNotIn('cp_corrected_ns',trace.analyze(d))
            for window in (2,4):
                self.assertEqual(trace.analyze(d,source,window)['cp_corrected_ns'],30)
            for k, v in trace.analyze_legacy(d).items():
                self.assertEqual(r[k+'_legacy'],v)

    def test_cycle_is_rejected_instead_of_zero_priced(self):
        rows = {0:dict(stage=0,logical_task=0),1:dict(stage=1,logical_task=0)}
        with self.assertRaisesRegex(ValueError,'cyclic'):
            trace.graph_longest(rows,{0:{1},1:{0}},lambda s:1)

    def test_zero_tick_tasks_are_retained_on_ties(self):
        rows = {i:dict(stage=i,logical_task=0) for i in range(3)}
        total,path = trace.graph_longest(rows,{0:set(),1:{0},2:{0,1}},lambda s:0 if s==1 else 5)
        self.assertEqual((total,path),(10,[0,1,2]))

    def test_unknown_runtime_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'model.cu'; p.write_text('no table')
            with self.assertRaisesRegex(ValueError,'no dependency table'):
                trace.dependency_graph([],p)


if __name__ == '__main__':
    unittest.main()
