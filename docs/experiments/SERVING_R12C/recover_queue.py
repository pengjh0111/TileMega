#!/usr/bin/env python3
"""Append repaired steps without resetting successful or in-progress work."""
import argparse,copy,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--queue-dir',type=Path,required=True);a=p.parse_args()
    state=json.loads(a.snapshot.read_text())['state'];original=json.loads((a.queue_dir/'queue_b.json').read_text())
    suffix='__recovery1';remap={s['name']:s['name']+suffix for s in original if state[s['name']]['status']=='skipped' or s['name'] in ('Bpre','B0b','Bsass','Breport')}
    steps=[]
    for old in original:
        name=old['name']
        if name not in remap:continue
        s=copy.deepcopy(old);s['name']=remap[name]
        for key in ('after','after_any'):s[key]=[remap.get(dep,dep) for dep in s.get(key,[])]
        if name=='B0b':s['command']+=['--resume','--retry-failed']
        if name=='Bpre':
            shell=s['command'].index('bash')+2
            s['command'][shell]="cmake --build build-phase12 --target tilemega tilemega-unit serving_epilogue_test paged_gemm_test serving_attention_cases_test -j 6 && python3 python/tilemega/fingerprint.py --check build-phase12/tools/tilemega && ctest --test-dir build-phase12 -R '^(serving_pruning|serving_lag|serving_task_index|skeleton_search_isolation)$' --output-on-failure && build-phase12/tools/tilemega compile docs/experiments/SERVING_R12B/controls/s1_B1_pages/decode_B1.so.top1.mlir /root/r12c_work/ref_cu/recovery.cu --serving decode --emit serving --batch 1 --past-range 64:1086 --capacity 1088 --sync calibrated --runtime-target /root/r12c_work/target_r12b.json --arch-paths auto --pdl auto --pg pages --page-bytes 16384 --weight-layout tiled --kphase-mask 31 --v3-poll-ns 0 --l2-prefetch-depth 1 --l2-prefetch-stride 128 --event-solo 0 --event-red-publish 0 --barrier-v2 0 && cmp /root/r12c_work/ref_cu/default.cu /root/r12c_work/ref_cu/recovery.cu && echo 'PASS recovery build, host tests and default CUDA identity'"
        steps.append(s)
    # Calibration never started; make its compiler prerequisite the repaired
    # build. Its original ID remains pending and will run exactly once.
    for s in original:
        if s['name'] in ('B0a','Bunit'):
            if state[s['name']]['status']!='pending' or state[s['name']]['attempts']:
                raise ValueError('compiler dependent step already started: '+s['name'])
            s['after']=[remap.get(dep,dep) for dep in s['after']]
    def write(path,value):
        pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)
    write(a.queue_dir/'queue_b.json',original);write(a.queue_dir/'queue_recovery.json',steps)
    write(HERE/'definitions/queue_b.json',original);write(HERE/'definitions/queue_recovery.json',steps)
    print('Appended '+str(len(steps))+' recovery steps; successful A rounds and fixed binaries retained')
if __name__=='__main__':main()
