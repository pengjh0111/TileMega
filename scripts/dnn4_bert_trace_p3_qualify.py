"""Reconstruct the trace-only P3 writer's unchanged boundary qualification."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record_windows(folder):
    """Validate every CTA record and reconstruct the original stage windows."""
    frames = read(folder / 'samples.json')
    require(len(frames) == 50, 'boundary needs all fifty frames')
    windows = []
    empty = bytes(320)
    for sample, frame in enumerate(frames):
        raw = gzip.decompress((folder / f'records_{sample:03d}.bin.gz').read_bytes())
        require(len(raw) == 63 * 132 * 352, 'boundary record size changed')
        previous = [0] * 132
        total = 0
        require([row['stage'] for row in frame] == list(range(63)),
                'boundary stage order changed')
        for stage, row in enumerate(frame):
            enter, arrive, leave = [], [], []
            for cta in range(132):
                offset = (stage * 132 + cta) * 352
                a, b, c, first = struct.unpack_from('<4Q', raw, offset)
                require(0 < a <= b <= c and previous[cta] <= a
                        and (first == 0 or a <= first <= b)
                        and raw[offset + 32:offset + 352] == empty,
                        'incomplete, unordered or non-boundary CTA record')
                previous[cta] = c
                enter.append(a)
                arrive.append(b)
                leave.append(c)
            stage_ns = max(leave) - min(enter)
            require(row['stage_ns'] == stage_ns
                    and row['work_window_ns'] == max(arrive) - min(enter)
                    and row['gate_tail_ns'] == max(leave) - max(arrive)
                    and stage_ns > 0, 'stage window differs from original raw records')
            total += stage_ns
        windows.append(total)
    return statistics.median(windows)


def qualify(preparation, runtime, unit_build):
    from tilemega.build.identity import sha, verify
    plan, prepared = read(preparation / 'plan.json'), read(preparation / 'result.json')
    request, result = read(runtime / 'plan.json'), read(runtime / 'result.json')
    require(request['qualification_processes'] == [0, 1, 2]
            and request['total_processes'] == 5
            and request['max_relative_difference'] == plan['max_relative_difference'] == .05
            and request['native_reference_ms'] == plan['native_reference_ms'] ==
                float(plan['reference']['median_ms']) == 1.1459199786186218,
            'predeclared process subset, native reference or threshold changed')
    require(prepared['status'] == 'prepared_CPU_only'
            and all(sha(Path(path)) == digest for path, digest in plan['files'].items())
            and all(sha(Path(path)) == digest for path, digest in
                    read(runtime / 'frozen_source.json').items()), 'prototype source changed')
    tracer = verify(Path(prepared['trace']['library']))
    native = verify(Path(plan['reference']['library']))
    source = Path(prepared['trace']['library'] + '.cu').read_text()
    require(source.startswith('#define TILEMEGA_F3_TRACE_P3_FLUSH 1\n')
            and source.split('\n', 1)[1].replace('#define TILEMEGA_TRACE_STAGE 3',
                '#define TILEMEGA_TRACE_STAGE 0') ==
                    Path(plan['reference']['library'] + '.cu').read_text()
            and tracer['trace'] and not native['trace']
            and tracer['plan']['native_forward'] == native['plan']['native_forward']
            and not any(r['spill'] for r in tracer['resources'].values()),
            'trace changes its original native program or resource contract')
    defaults = read(preparation / 'default_proofs.json')
    require(defaults['passed'] and sorted(r['batch'] for r in defaults['rows']) == [1, 2, 4, 8, 16],
            'five default instruction proofs missing')
    for row in defaults['rows']:
        verify(Path(row['library']))
        require(row['passed'] and sha(Path(row['reference'])) == row['reference_sha256'],
                'default reference changed')
        def words(path):
            return re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/', subprocess.check_output(
                ['cuobjdump', '-sass', path], text=True))
        actual = words(row['library'])
        require(actual and len(actual) == row['encoded_words']
                and actual == words(row['reference']), 'default GPU instruction words changed')
    units, build = read(runtime / 'unit/result.json'), read(unit_build / 'result.json')
    children = units['processes']
    require(units['passed'] and len(children) == len({p['pid'] for p in children}) == 17
            and Counter(p['name'].split('-')[0] for p in children) ==
                Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
            and len({p['output_hash'] for p in children}) == 1
            and sha(Path(build['binary'])) == build['binary_sha256']
            and all(not p['exit_code'] and not read(p['guard'])['code']
                    and p['binary_sha256'] == build['binary_sha256'] for p in children),
            'publication/parity unit needs accepted 5/1/1/5/5')
    cohort = read(runtime / 'traces/result.json')
    require(cohort['status'] == 'passed' and len(cohort['processes']) == len(cohort['traces']) == 5
            and not read(cohort['guard'])['code'], 'five accepted full-model children missing')
    medians, cards, pids, inputs = [], set(), set(), set()
    for process, folder in zip(cohort['processes'], cohort['traces']):
        folder = Path(folder)
        child = read(folder / 'result.json')
        require(not process['exit_code'] and process['pid'] == child['pid']
                and child['pid'] not in pids and folder == Path(process['out']) / 'trace'
                and child['artifact_id'] == tracer['artifact_id']
                and child['reference_sha256'] == native['binary_sha256']
                and child['samples'] == 50 and child['warmup'] == 20 and child['trace_level'] == 3
                and child['dimensions'] == dict(schema=1, stages=63, grid=132, segments=10,
                    record_bytes=352, bytes=63*132*352)
                and all(math.isfinite(child[k]) and child[k] > 0
                    for k in ('start_sm_clock_mhz', 'end_sm_clock_mhz')),
                'full-model child identity or original trace protocol changed')
        medians.append(record_windows(folder))
        cards.add(child['gpu'])
        inputs.add(child['input_sha256'])
        pids.add(child['pid'])
    require(len(cards) == len(inputs) == 1, 'trace mixes physical cards or inputs')
    difference = statistics.median(medians[:3]) / (plan['native_reference_ms'] * 1e6) - 1
    passed = abs(difference) <= .05
    require(result['process_windows_ns'] == medians and result['relative_difference'] == difference
            and result['qualification_processes'] == [0, 1, 2]
            and result['max_relative_difference'] == .05
            and result['native_reference_ms'] == plan['native_reference_ms']
            and result['passed'] == passed and not result['production_admitted'],
            'qualification summary changes original raw evidence')
    return dict(evidence='verified', passed=passed, gpu=cards.pop(), processes='5/5',
        unit_processes='5/1/1/5/5', process_windows_ns=medians,
        qualification_processes=[0, 1, 2], relative_difference=difference,
        native_reference_ms=plan['native_reference_ms'], max_relative_difference=.05,
        artifact_id=tracer['artifact_id'], binary_sha256=tracer['binary_sha256'],
        input_sha256=inputs.pop(), source=str(runtime / 'traces/result.json'),
        scope='Trace-only diagnostic qualification;first3 of5 against the frozen native. '
              'All250 full raw frames checked;no native retime,latency gain,P-MODEL or formal admission.')
