#!/usr/bin/env python3
"""Summarize the specified 65-discard/512-step CTA trace window."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tilemega.build.identity import sha, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--decode-so',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    scope=json.loads((a.capture/'trace_scope.json').read_text());identity=verify(a.decode_so)
    assert scope['decode_binary_sha256']==sha(a.decode_so)and scope['trace']==1
    stages=identity['plan']['stages_per_step'];raw=a.capture/'mx_stage_trace.csv'
    frame=pd.read_csv(raw,dtype=np.uint64)
    assert len(frame)==577*stages*104
    assert np.array_equal(frame['slot'].to_numpy(),np.repeat(np.arange(577),stages*104))
    assert np.array_equal(frame['stage'].to_numpy(),np.tile(np.repeat(np.arange(stages),104),577))
    assert np.array_equal(frame['cta'].to_numpy(),np.tile(np.arange(104),577*stages))
    stamps=frame[['start_ns','work_end_ns','release_ns']].to_numpy().reshape(577,stages,104,3)
    assert np.all(stamps>0)
    release=stamps[:,:,:,2];previous=np.empty_like(release)
    previous[:,1:]=release[:,:-1];previous[1:,0]=release[:-1,-1];previous[0,0]=stamps[0,0,:,0]
    assert np.all(release[65:]>=previous[65:])
    durations=(release[65:]-previous[65:]).astype(np.float64)/1000
    anchor=previous[65:].min(axis=2,keepdims=True)
    wait=(stamps[65:,:,:,1].astype(np.int64)-anchor.astype(np.int64)).astype(np.float64)/1000
    rows=[];names=('qkv','attention','o','router','gate_up','down','head')
    base=dict(checkpoint='CP2prime_diagnostic',model=identity['plan']['model'],batch=scope['batch'],
        variant=scope['variant'],round=1,executor='L1_loop',implementation='mx_moe_v3',contract='bf16',
        trace=1,diag=0,watchdog=0,PID=scope['PID'],discard_steps=65,steps=512,ctas=104,
        source_head=scope['source_head'],source_sha256=scope['source_sha256'],binary_sha256=sha(a.decode_so),
        raw_csv=str(raw),raw_sha256=sha(raw),timing_eligible=False,evidence='verified')
    for stage in range(stages):
        cls=stage%6 if stage+1<stages else 6
        rows.append(dict(base,record_type='layer',layer=stage//6,stage=names[cls],
            duration_us=float(np.median(np.median(durations[:,stage],axis=1))),
            earliest_wait_entry_us=float(np.median(wait[:,stage].min(axis=1))),
            median_wait_entry_us=float(np.median(np.median(wait[:,stage],axis=1))),
            latest_wait_entry_us=float(np.median(wait[:,stage].max(axis=1))),
            wait_entry_spread_us=float(np.median(np.ptp(wait[:,stage],axis=1)))))
    for cls,name in enumerate(names[:6]):
        indexes=np.arange(cls,stages-1,6)
        rows.append(dict(base,record_type='mean_layers_then_median_steps',layer='all',stage=name,
            duration_us=float(np.median(np.median(durations[:,indexes],axis=2).mean(axis=1))),
            earliest_wait_entry_us=float(np.median(wait[:,indexes].min(axis=2).mean(axis=1))),
            median_wait_entry_us=float(np.median(np.median(wait[:,indexes],axis=2).mean(axis=1))),
            latest_wait_entry_us=float(np.median(wait[:,indexes].max(axis=2).mean(axis=1))),
            wait_entry_spread_us=float(np.median(np.ptp(wait[:,indexes],axis=2).mean(axis=1)))))
    with(a.out/'ledger.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(a.out/'step_stage_statistics.npz',duration_median_cta=np.median(durations,axis=2),
        wait_min=wait.min(axis=2),wait_median=np.median(wait,axis=2),wait_max=wait.max(axis=2))
    summary=[r for r in rows if r['record_type']=='mean_layers_then_median_steps'or r['stage']=='head']
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),summary=summary,
        definition='per-CTA release minus previous release; wait offsets from minimum previous-stage release; per-step mean over layers then median of512 steps'),indent=2)+'\n')
    print('TRACE_LEDGER_PASS',scope['variant'],scope['batch'],len(rows),flush=True)


if __name__=='__main__':main()
