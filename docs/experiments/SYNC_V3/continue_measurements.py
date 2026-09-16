#!/usr/bin/env python3
"""Serialize performance suites after all dependent correctness evidence exists."""
import json
from pathlib import Path
import subprocess
import time
import audit_raw

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def ready():
    witness=HERE/'c2_dependency/witnesses.tsv'
    if not witness.exists():return False
    for c in ('c1','c2','c2_dependency','local2','local4','window2','window4','sharded_red'):
        audit_raw.correctness(HERE/c)
    for c in ('c1','c2'):audit_raw.correctness(HERE/c,subset=True)
    return True


def main():
    start=time.monotonic()
    while not ready():
        if time.monotonic()-start>14400:raise TimeoutError('validation not complete')
        time.sleep(5)
    stages=[['python3',str(REPO/'docs/experiments/CHAIN2/run.py'),'correctness'],
            ['python3',str(REPO/'docs/experiments/CHAIN2/run.py'),'measure'],
            ['python3',str(HERE/'target_positions.py'),'measure'],
            ['python3',str(HERE/'measure_matrix.py'),'--manifest',str(HERE/'matrix_manifest.json'),
             '--out',str(HERE/'ablation')],
            ['python3',str(HERE/'realwidth.py'),'measure']]
    records=[]
    for cmd in stages:
        if str(HERE/'measure_matrix.py') in cmd:
            while not (HERE/'window_probe_fix/build_status.txt').exists():
                time.sleep(5)
        print('SERIAL_MEASURE',cmd,flush=True)
        result=subprocess.run(cmd,cwd=REPO)
        records.append(dict(command=cmd,exit_code=result.returncode))
        (HERE/'measurement_commands.json').write_text(json.dumps(records,indent=2)+'\n')
        # An individual failed suite leaves its evidence intact. Independent
        # suites still run, as directed by the user's clarified stop rule.
    return int(any(r['exit_code'] for r in records))


if __name__=='__main__':raise SystemExit(main())
