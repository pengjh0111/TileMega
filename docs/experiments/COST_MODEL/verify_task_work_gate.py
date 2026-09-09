#!/usr/bin/env python3
"""Check work-gate coverage; deliberately not a price or GPU gate."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent
    repo=here.parents[2]
    raw=here/'task_work_full_gate.tsv'
    error=here/'task_work_full_gate.stderr'
    output=here/'task_work_full_gate.json'
    if output.exists():
        raise RuntimeError('refusing to overwrite verified evidence')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    rows=list(csv.DictReader(raw.open(),delimiter='\t'))
    groups={}
    for row in rows:
        key=(row['dtype'],row['model'])
        if key not in {(d,m) for d in ('bf16','f32') for m in ('gqa2','mha4')}:
            raise RuntimeError('unexpected work-gate group')
        if row['status']!='PASS' or int(row['seq_points'])!=5:
            raise RuntimeError('incomplete work-gate row')
        if int(row['gemm_stages'])<=0 or int(row['work_bit_checks'])!=15*int(row['gemm_stages']):
            raise RuntimeError('stage/seq/check coverage differs')
        seen=groups.setdefault(key,set())
        if row['config'] in seen:
            raise RuntimeError('duplicate configuration')
        seen.add(row['config'])
    source_hashes={}
    for dtype in ('bf16','f32'):
        for model in ('gqa2','mha4'):
            oracle=repo/f'docs/experiments/ORACLE/raw/screen_{model}.tsv'
            expected=set()
            for row in csv.DictReader(oracle.open(),delimiter='\t'):
                if row['status']=='PASS':
                    expected.add(f'{row["tile_m"]}x{row["tile_n"]}x{row["tile_k"]}'
                                 f's{row["stages"]}k{row["split_k"]}')
            if len(expected)!=1077 or groups.get((dtype,model))!=expected:
                raise RuntimeError('1077-configuration universe differs')
            marker=f'TASK_WORK_GATE dtype={dtype} model={model} configs=1077 status=PASS'
            if marker not in error.read_text():
                raise RuntimeError('missing completed group marker')
            source_hashes[str(oracle.relative_to(repo))]=sha(oracle)
    if 'ISL_CONTEXT remaining=0' not in error.read_text():
        raise RuntimeError('missing zero-reference evidence')
    checks=sum(int(row['work_bit_checks']) for row in rows)
    report=dict(scope='A3 GEMM work only; not A6 CostBreakdown, ranking or GPU acceptance',
        config_groups=len(rows),work_bit_checks=checks,
        checked_quantities=['task_count','nominal_mainloop_bytes','nominal_mainloop_iterations'],
        configuration_source='same complete archived FP32 1077-shape universe for both dtypes',
        hashes={raw.name:sha(raw),error.name:sha(error),**source_hashes},
        binary_sha256_captured_after_completion=sha(repo/'build-portable/tools/tilemega-task-work-gate'))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'WORK_GATE_VERIFIED config_groups={len(rows)} work_bit_checks={checks}')


if __name__=='__main__':
    main()
