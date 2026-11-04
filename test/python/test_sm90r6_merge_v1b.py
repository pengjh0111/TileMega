import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from sm90r6_merge_analysis import write
from sm90r6_merge_v1b import stage_rows,merge,read_csv


class StageDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def table(self,dense):
        rows=[]
        for model in (('qwen3_1_7b','llama3_1_8b') if dense else ('qwen3_30b_a3b',)):
            for batch in (1,16):
                for stage in ('qkv','o','attention','lm_head',*(('gate_up','down') if dense else ('router','expert_fused'))):
                    row=dict(model=model,batch=batch,stage=stage,processes=3 if dense else 1,
                        additive_stage_us=12,additive_over_measured=1.2,flow_over_measured=.9)
                    row['measured_span_us' if dense else 'measured_boundary_us']=10
                    row['flow_span_us' if dense else 'flow_space_sum_us']=9
                    row['flow_components_per_instance_us' if dense else 'flow_space_medians_us']=json.dumps({'body':7,'reducer':2})
                    rows.append(row)
        path=self.root/('dense.csv' if dense else 'moe.csv');write(path,rows)
        return path,rows

    def test_components_are_reported_separately_without_changing_old_diagnostics(self):
        dense,_=self.table(True);moe,_=self.table(False)
        example=stage_rows(dense,True)[0]
        base=self.root/'base.csv';out=self.root/'out.csv'
        old=[dict(example,check=check,value=123) for check in ('V1(a)','V2-TP-subset','V3-TP','V4-F-warm-subset')]
        write(base,old);merge(base,dense,moe,out)
        result=read_csv(out)
        self.assertEqual(result[:4],read_csv(base))
        components=[r for r in result[4:] if r['metric']=='space.reducer_us']
        self.assertEqual(len(components),36)
        self.assertTrue(all(float(r['ratio'])==.2 for r in components))
        merge(out,dense,moe,out)
        self.assertEqual(result,read_csv(out))

    def test_incomplete_dense_matrix_is_rejected(self):
        path,rows=self.table(True);write(path,[r for r in rows if r['batch']==1])
        with self.assertRaisesRegex(ValueError,'incomplete'):
            stage_rows(path,True)

    def test_changed_semantic_sum_or_ratio_is_rejected(self):
        path,rows=self.table(False);rows[0]['flow_space_sum_us']=8;write(path,rows)
        with self.assertRaisesRegex(ValueError,'sum'):
            stage_rows(path,False)
        rows[0]['flow_space_sum_us']=9;rows[0]['flow_over_measured']=.8;write(path,rows)
        with self.assertRaisesRegex(ValueError,'ratio'):
            stage_rows(path,False)


if __name__=='__main__':unittest.main()
