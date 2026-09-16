#!/usr/bin/env python3
"""R5 prerequisite audit, rerun with the unchanged R4 trace binaries."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'docs/experiments/TRACE_V2'))
import analyze


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(dump, source):
    r = analyze.analyze(dump, source, 1)
    _, slots, _, _ = analyze.load(dump)
    path = set(tuple(map(int, x.split(':'))) for x in r['cp_corrected_path'].split(','))
    result = {k: r[k] for k in ('cp_corrected_ns', 'cp_corrected_nodes', 'queue_lb_ns',
              'measured_l2_ms', 'cp_corrected_path')}
    for label, values in [('all', [s['run_end']-s['run_begin'] for s in slots]),
                          ('cp', [s['run_end']-s['run_begin'] for s in slots
                                  if (s['stage'], s['logical_task']) in path])]:
        result[label+'_count'] = len(values)
        result[label+'_mean_ns'] = sum(values)/len(values)
        for name, q in [('p50', .5), ('p90', .9), ('max', 1)]:
            result[label+'_'+name+'_ns'] = analyze.percentile(values, q)
    result['measured_over_floor'] = r['measured_l2_ms']*1e6/max(r['cp_corrected_ns'], r['queue_lb_ns'])
    return result


def main():
    raw = HERE/'audit'
    raw.mkdir(exist_ok=True)
    session = str(time.time_ns())
    rows = []
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            for placement, candidate in [(0, 'legacy_grid_stride'), (5, 'rotate')]:
                key = f'{model}_s{seq}_{candidate}'
                origin = REPO/'docs/experiments/SYNC_V3/targets_raw'
                binary = origin/'bin'/key
                build = json.loads((origin/'log'/f'{key}.build.json').read_text())
                assert sha(binary) == build['binary_sha256']
                source = Path(build['source'])
                assert sha(source) == build['source_sha256']
                dump = raw/key
                dump.mkdir(exist_ok=True)
                log = raw/f'{key}.log'
                if log.exists():
                    raise ValueError(f'refusing overwrite: {log}')
                env = {k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
                env.update(TILEMEGA_TRACE_V2='1', TILEMEGA_TRACE_V2_OUT=str(dump), TILEMEGA_MODEL_NAME=model)
                command = [str(binary), str(REPO/f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3')]
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
                log.write_text(result.stdout+result.stderr)
                log.with_suffix('.json').write_text(json.dumps(dict(command=command, session=session,
                    time_ns=time.time_ns(), exit_code=result.returncode, build=build,
                    source_sha256=sha(source), binary_sha256=sha(binary)), indent=2)+'\n')
                assert not result.returncode and 'RESULT status=PASS' in log.read_text()
                row = dict(model=model, seq=seq, placement=placement,
                           dump=str(dump.relative_to(REPO)), source=str(source.relative_to(REPO)),
                           **collect(dump, source))
                rows.append(row)
                print(json.dumps(row), flush=True)
    with (raw/'durations.tsv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    provenance = dict(base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        prompt='/root/Prompt/TileMega_R5_prompt.md', prompt_sha256=sha(Path('/root/Prompt/TileMega_R5_prompt.md')),
        session=session, reused_binaries='R4 default-configuration trace binaries, new processes; immutable digest checked',
        actual_generated_tile='128x128x16 stages=3, split=1; F-128 DP result is a different configuration',
        comparison='S3-b control is rotate plus R3 B protocol; 0.7129ms belongs to placement0 and is historical only')
    (raw/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    main()
