#!/usr/bin/env python3
"""Qualify three independently guarded boundary traces on the reference card."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(os.environ.get('TILEMEGA_DNN4_REPO', '/root/TileMega-dnn3'))
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.identity import sha, verify


def read(path):
    return json.loads(Path(path).read_text())


def qualify(reference_path, traces, library, reference):
    measured = read(reference_path)
    identity, control = verify(library), verify(reference)
    if (measured['protocol'] != 'P-SMOKE' or measured['gpu'] != 0
            or len(measured['measurements']) != 1 or len(traces) != 3
            or identity['defines'].get('TILEMEGA_TRACE_STAGE') != '3'
            or control['trace'] or identity['plan']['native_forward'] != control['plan']['native_forward']):
        raise ValueError('qualification requires three GPU0 traces at the reference coordinates')
    row = measured['measurements'][0]
    timing = read(row['measurement'])
    if (row['guard']['code'] or timing['gpu_index'] != 0
            or timing['artifact_id'] != control['artifact_id']
            or timing['measurement']['warmup'] != 50
            or timing['measurement']['iterations'] != 200 or len(timing['samples_ms']) != 200
            or measured['candidate_ms'] != timing['median_ms']):
        raise ValueError('qualification lacks its complete nontrace reference')
    windows, processes, clocks = [], [], []
    stage_count = len(identity['plan']['native_forward']['stages'])
    reference_hash = sha(reference)
    for output in traces:
        output = Path(output)
        state = read(output/'scheduler/state.json')
        if len(state) != 1:
            raise ValueError('each boundary process needs its own reservation')
        state = next(iter(state.values()))
        folder = Path(state['out'])
        guard = read(folder/'guard_result.json')
        result = read(folder/'trace/result.json')
        if (state['status'] != 'done' or state['exit_code'] or guard['code']
                or result['gpu'] != 0 or result['trace_level'] != 3
                or result['samples'] != 50 or result['warmup'] != 20
                or result['artifact_id'] != identity['artifact_id']
                or result['reference_sha256'] != reference_hash
                or result['input_sha256'] != measured['input_sha256']):
            raise ValueError('boundary process differs from its guarded reference')
        samples = read(folder/'trace/samples.json')
        if len(samples) != 50 or any(len(sample) != stage_count for sample in samples):
            raise ValueError('boundary process lacks complete stage windows')
        if any([stage['stage'] for stage in sample] != list(range(stage_count))
               or any(not math.isfinite(stage['stage_ns']) or stage['stage_ns'] <= 0
                      for stage in sample) for sample in samples):
            raise ValueError('boundary windows are invalid or have duplicate stages')
        windows.append(statistics.median(sum(s['stage_ns'] for s in sample) for sample in samples))
        processes.append(dict(pid=result['pid'], out=str(folder/'trace'),
                              source_state=str(output/'scheduler/state.json')))
        clocks.append({key: result[key] for key in ('start_sm_clock_mhz', 'end_sm_clock_mhz')})
    if len({p['pid'] for p in processes}) != 3 or timing['pid'] in {p['pid'] for p in processes}:
        raise ValueError('qualification reused a process')
    elapsed = measured['candidate_ms']
    window = statistics.median(windows)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid nontrace reference latency')
    difference = window/(elapsed*1e6)-1
    return dict(evidence='verified diagnostic only', model='bert', batch=measured['batch'], gpu=0,
        library=str(library), reference=str(reference), trace_ns=window, nontrace_ms=elapsed,
        relative_difference=difference, passed=abs(difference) <= .05,
        processes=processes, clocks=clocks, nontrace_source=str(reference_path),
        scope='Three independent20/50 boundary processes compare exact output bytes with the accepted '
              'nontrace artifact;fresh same-card50/200 reference;no model correctness or speedup admission')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nontrace', type=Path, required=True)
    parser.add_argument('--traces', type=Path, nargs=3, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.nontrace, args.traces, args.library, args.reference)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('batch', 'relative_difference', 'passed')}))


if __name__ == '__main__':
    main()
