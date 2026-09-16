#!/usr/bin/env python3
"""Recompute R6 calibration gates from evaluator rows and historical raw timings.
Historical validation is labelled separately from this round's GPU comparisons.
"""
import csv,json,math,statistics,sys,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('r5_rank_math',REPO/'docs/experiments/SIMULATOR/r5/report.py')
rank_math=importlib.util.module_from_spec(spec);spec.loader.exec_module(rank_math)
spearman=rank_math.spearman
def read(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def main():
 rows=read(HERE/'calibrated_replay/evaluations.tsv');measure=read(REPO/'docs/experiments/SIMULATOR/raw/time/l2.tsv')
 modes={'legacy_grid_stride':'0','balanced':'4','rotate':'5'}
 selected=[r for r in rows if r['model']!='real' and r['candidate'] in modes]
 actual=[statistics.median(float(x['l2_ms'])*1e6 for x in measure if (x['model'],x['seq'],x['place'])==(r['model'],r['seq'],modes[r['candidate']])) for r in selected]
 result={'points':len(selected),'scope':'18-point historical raw calibration validation; fresh CPU evaluations, not fresh GPU speedup'}
 for key,gate,threshold in [('full_ns','C-b',.880288958),('coarse_ns','C-c',.85)]:
  value=spearman([float(r[key]) for r in selected],actual);result[gate]={'spearman':value,'threshold':threshold,'pass':len(selected)==18 and value>=threshold}
 for model,limit in [('reference',1000),('real',10000)]:
  group=[r for r in rows if (r['model']=='real')==(model=='real')]
  result[model]={k:max(float(r[k]) for r in group) for k in ['prepare_us','coarse_us','full_us']}
  result[model]['budget_us']=limit;result[model]['pass']=result[model]['full_us']<limit
 (HERE/'calibrated_replay/report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
