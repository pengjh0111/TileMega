"""Analyze qualified immutable TP traces without keeping a GPU job alive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from tilemega.tp.trace import read_trace, summarize


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def analyze_capture(capture_path, guard_path, out, layers=28, grid=104):
    capture_path, guard_path, out = map(Path, (capture_path, guard_path, out))
    if out.exists():
        raise ValueError('preserve existing analyzed trace evidence')
    data = json.loads(capture_path.read_text())
    guard = json.loads(guard_path.read_text())
    if guard['code'] != 0:
        raise ValueError('GPU capture did not pass its guard')
    if list(map(int, guard['gpu_indices'])) != data['gpu_indices'] or data['tp'] != 4:
        raise ValueError('trace capture and guard GPU groups differ')
    collector = capture_path.parent / 'collector.py'
    if sha(collector) != data['collector']['sha256']:
        raise ValueError('archived GPU trace collector differs')
    if data.get('kind') != 'mx7_tp_stage_capture' or len(data['raw_traces']) != 4:
        raise ValueError('complete four-rank GPU trace capture required')
    out.parent.mkdir(parents=True, exist_ok=True)
    analyzer = out.parent / 'analyzer.py'
    dependency = out.parent / 'analyzer_trace.py'
    if analyzer.exists() or dependency.exists():
        raise ValueError('preserve existing trace analyzer sources')
    shutil.copy2(__file__, analyzer)
    from tilemega.tp import trace
    shutil.copy2(trace.__file__, dependency)
    ranks = []
    minimum_slot = data['trace_slots'] - data['decode_samples']
    for record in data['raw_traces']:
        path = Path(record['path'])
        if sha(path) != record['sha256']:
            raise ValueError('raw compressed trace bytes changed')
        samples = read_trace(path, layers=layers, grid=grid, minimum_slot=minimum_slot)
        if len(samples) != data['decode_samples'] * (5 * layers + 1):
            raise ValueError('trace lacks the full decode scope')
        ranks.append(samples)
    report = dict(data, stages=summarize(ranks, layers=layers), capture=ref(capture_path),
                  capture_guard=ref(guard_path), analyzer=ref(analyzer), analyzer_dependency=ref(dependency))
    report.pop('kind')
    if sha(__file__) != sha(analyzer) or sha(trace.__file__) != sha(dependency):
        raise ValueError('CPU trace analyzer source changed during analysis')
    temporary = out.with_suffix(out.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    os.link(temporary, out)
    temporary.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.capture.read_text())
    if data.get('decode_samples') != 1023 or data.get('trace_slots') != 1087:
        parser.error('final trace requires the complete ISL64/OSL1024 scope')
    analyze_capture(args.capture, args.guard, args.out)
    print('TP4 trace CPU analysis complete', data['batch'], flush=True)


if __name__ == '__main__':
    main()
