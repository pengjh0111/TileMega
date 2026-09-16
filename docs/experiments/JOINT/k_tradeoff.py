#!/usr/bin/env python3
"""Recompute k=1..5 timing/ranking tradeoff on the unchanged 18-point set."""
import statistics
from pathlib import Path
from verify import table,spearman
from search import REPO,HERE,write

def main():
 root=REPO/'docs/experiments/SIMULATOR/r5';ranks=table(root/'ranks.tsv');actual=table(REPO/'docs/experiments/SIMULATOR/raw/time/l2.tsv');modes={'legacy_grid_stride':'0','balanced':'4','rotate':'5'};out=[]
 for k in range(1,6):
  rows=[r for r in ranks if r['k']==str(k) and r['model']!='real' and r['candidate'] in modes]
  measured=[statistics.median(float(x['l2_ms']) for x in actual if (x['model'],x['seq'],x['place'])==(r['model'],r['seq'],modes[r['candidate']])) for r in rows]
  pred=[float(r['predicted_ns']) for r in rows];best=min(range(len(rows)),key=lambda i:measured[i]);rank=sorted(range(len(rows)),key=lambda i:pred[i]).index(best)+1
  batches={(r['model'],r['seq']):float(r['batch_us']) for r in ranks if r['k']==str(k)}
  out.append(dict(k=k,points=len(rows),spearman=spearman(pred,measured),actual_top1_rank=rank,reference_max_batch_us=max(v for (m,s),v in batches.items() if m!='real'),real_max_batch_us=max(v for (m,s),v in batches.items() if m=='real'),simulated_calibration_points=sum(r['simulated']=='1' for r in rows)))
 write(HERE/'k_tradeoff.tsv',out)
 for r in out:print(r)
if __name__=='__main__':main()
