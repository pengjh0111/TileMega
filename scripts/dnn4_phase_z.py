#!/usr/bin/env python3
"""Run one resumable DNN4 Phase Z cell, including six in-situ variants."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from dnn4_tune_prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def coordinate_key(coordinates):
    return hashlib.sha256(json.dumps(coordinates, sort_keys=True).encode()).hexdigest()[:20]


class Cell:
    def __init__(self, args):
        self.args = args
        self.key = f'{args.model}_b{args.batch}'
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.source = args.solve.resolve() / self.key
        self.bridge = ROOT / f'runs/dnn3/takeover_20261107/plan_gate/{self.key}.bridge.json'
        self.export = Path('/root/TileMega-dnn1/runs/dnn1/exports') / args.model
        if args.model == 'resnet50':
            self.export = ROOT / 'runs/dnn3/exports/resnet50'
        self.inputs = self.export / f'inputs_b{args.batch}.safetensors'
        self.device = {'resnet18': 3, 'bert': 3, 'resnet50': 0, 'vgg16': 2}[args.model]
        self.cpu = {0: '0-7', 2: '104-111', 3: '112-119'}[self.device]
        self.compile_cpu = f'{136+4*args.slot}-{139+4*args.slot}'
        self.result = dict(evidence='verified', model=args.model, batch=args.batch,
                           gpu=self.device, status='running', stage='start',
                           qualification='provisional_container_pid_namespace')

    def progress(self, stage, **updates):
        self.result.update(stage=stage, **updates)
        save(self.out / 'progress.json', self.result)
        print(json.dumps(dict(cell=self.key, stage=stage, **updates)), flush=True)

    def run(self, name, command, completion=None):
        marker = self.out / (name + '.complete.json')
        if marker.exists():
            if read(marker)['command'] != list(map(str, command)):
                raise ValueError('resume command changed: ' + name)
            return
        self.progress(name)
        with (self.out / (name + '.log')).open('w') as log:
            code = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f'{name} exit {code}; inspect {log.name}')
        if completion is not None and not completion.exists():
            raise RuntimeError('missing completion artifact ' + str(completion))
        save(marker, dict(command=list(map(str, command)), completed_unix=time.time()))

    def common(self):
        return ['--export', self.export, '--bridge', self.bridge, '--inputs', self.inputs,
                '--batch', self.args.batch]

    def build(self, coordinates, trace=0):
        key = coordinate_key(coordinates)
        rank0 = self.prepared['rank0']
        if not trace and coordinates == rank0:
            path = self.args.native.resolve() / self.key / 'model.so'
            for _ in range(720):
                if (path.parent / 'result.json').exists():
                    if read(path.parent / 'result.json')['status'] == 'passed':
                        return path
                time.sleep(5)
            raise RuntimeError('rank0 native build not ready: ' + self.key)
        folder = self.out / 'sources' / key
        folder.mkdir(parents=True, exist_ok=True)
        coords = folder / 'coordinates.json'
        if coords.exists() and read(coords) != coordinates:
            raise ValueError('coordinate digest collision')
        save(coords, coordinates)
        source = folder / 'input.so.cu'
        self.run('codegen-' + key, ['taskset', '-c', '128-135', self.args.binary.resolve(),
            'compile', self.bridge, source, '--emit', 'serving', '--serving', 'forward',
            '--frontend', 'dnn', '--batch', self.args.batch, '--abi', 'f3',
            '--tmexec-config', coords, '--runtime-target', self.args.target.resolve()], source)
        prefix = folder / 'input.so'
        for old, new in [('.cu.plan.json', '.plan.json'), ('.cu.tmexec.mlir', '.tmexec.mlir')]:
            shutil.copyfile(str(prefix) + old, str(prefix) + new)
        dest = self.out / 'artifacts' / (key + f'-trace{trace}')
        self.run('native-' + key + f'-trace{trace}', [sys.executable,
            ROOT / 'scripts/dnn4_compile.py', '--input', prefix, '--out', dest,
            '--trace-level', trace, '--cpu-set', self.compile_cpu,
            '--build-lock', f'/tmp/tilemega_dnn4_native_z{self.args.slot}.lock'], dest / 'result.json')
        return dest / 'model.so'

    def gate(self, library):
        identity = read(Path(str(library) + '.identity.json'))['artifact_id']
        dest = self.out / 'correctness' / identity[:20]
        self.run('corr-' + identity[:20], [sys.executable, ROOT / 'scripts/dnn4_model_gate.py',
            '--library', library, *self.common(), '--device', '0', '--out', dest], dest / 'scheduler/state.json')
        state = read(dest / 'scheduler/state.json')['dnn4-f3-model-gates']
        cohort = Path(state['out']) / 'cohort'
        result = read(cohort / 'result.json')
        if result.get('status') != 'passed' or result.get('passes') != '5/5':
            raise ValueError('five-process correctness gate did not pass')
        return cohort / 'corr-1/check.json'

    def measure(self, name, library, gate, protocol, extra=()):
        dest = self.out / name
        self.run(name, [sys.executable, ROOT / 'scripts/dnn4_measure.py', '--library', library,
            '--gate', gate, *self.common(), '--device', self.device, '--cpu-set', self.cpu,
            '--protocol', protocol, '--out', dest, *extra], dest / 'result.json')
        return read(dest / 'result.json')

    def trace(self, name, library, samples=50, reference=None):
        dest = self.out / name
        extra = [] if reference is None else ['--reference', reference]
        self.run(name, [sys.executable, ROOT / 'scripts/dnn4_marks.py', '--library', library,
            *self.common(), '--device', self.device, '--cpu-set', self.cpu, '--samples', samples,
            '--warmup', 20 if samples == 50 else 50, '--out', dest, *extra], dest / 'scheduler/state.json')
        state = read(dest / 'scheduler/state.json')['dnn4-marks']
        return Path(state['out']) / 'trace'

    def execute(self):
        lock = (self.out / 'cell.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.out / 'result.json').exists():
            return
        receipt = read(self.source / 'winner.cu.search/receipt.json')
        self.prepared = prepare(receipt, (self.source / 'candidate-0.cu').read_text())
        overrides = self.out / 'coordinate_overrides.json'
        if overrides.exists():
            original = self.out / 'tune_inputs.initial.json'
            if not original.exists():
                save(original, self.prepared)
            for override in read(overrides):
                group = next(g for g in self.prepared['groups'] if g['stages'] == override['stages'])
                rank, replacement = override['variant'], override['use_candidate']
                prior = copy.deepcopy(group['candidates'][rank])
                group['candidates'][rank] = copy.deepcopy(group['candidates'][replacement])
                coordinate = group['candidates'][rank]['coordinate']
                for stage in group['stages']:
                    self.prepared['variants'][rank]['stages'][stage] = coordinate
                group['distinct_candidates'] = len({coordinate_key(c['coordinate']) for c in group['candidates']})
                group.setdefault('native_rejections', []).append(dict(override, rejected=prior))
            self.prepared['overrides'] = dict(path=str(overrides), sha256=hashlib.sha256(overrides.read_bytes()).hexdigest())
        save(self.out / 'tune_inputs.json', self.prepared)
        rank0 = self.build(self.prepared['rank0'])
        rank0_gate = self.gate(rank0)
        variants, windows, totals = self.prepared['variants'], [], []
        trace_libraries = {}
        for rank, coordinates in enumerate(variants):
            key = coordinate_key(coordinates)
            if key in trace_libraries:
                earlier = next(i for i in range(rank) if variants[i] == coordinates)
                windows.append(windows[earlier]); totals.append(totals[earlier])
                continue
            library = self.build(coordinates, trace=3)
            trace_libraries[key] = library
            processes = [read(self.trace(f'tune-{rank}-process-{i}', library) / 'samples.json') for i in range(3)]
            group_windows = []
            for group in self.prepared['groups']:
                values = [statistics.median(sum(sample[stage]['stage_ns'] for stage in group['stages'])
                                             for sample in process) for process in processes]
                group_windows.append(statistics.median(values))
            windows.append(group_windows)
            totals.append(statistics.median(statistics.median(sum(s['stage_ns'] for s in sample)
                                                               for sample in process) for process in processes))
        tuned = copy.deepcopy(self.prepared['rank0'])
        selections = []
        for index, group in enumerate(self.prepared['groups']):
            winner = min(range(6), key=lambda rank: (windows[rank][index], rank))
            for stage in group['stages']:
                tuned['stages'][stage] = variants[winner]['stages'][stage]
            selections.append(dict(group=index, stages=group['stages'], candidate=winner,
                                   windows_ns=[windows[rank][index] for rank in range(6)]))
        save(self.out / 'tune_selection.json', dict(groups=selections, variant_total_ns=totals,
                                                    coordinates=tuned, convention='median of three process medians of per-launch group sums'))
        tuned_library = self.build(tuned)
        tuned_gate = self.gate(tuned_library)
        if tuned == self.prepared['rank0']:
            paired = dict(protocol='P-PAIR deduplicated', ratio=1, threshold=.005,
                          candidate_ms=None, reference_ms=None, slower=False)
        else:
            paired = self.measure('tuned_pair', tuned_library, tuned_gate, 'P-PAIR',
                ['--reference-library', rank0, '--reference-gate', rank0_gate,
                 '--reference-bridge', self.bridge])
        save(self.out / 'tuned_pair_decision.json', paired)
        best_trace_rank = min(range(6), key=lambda i: (totals[i], i))
        ordered = [(tuned, 'tuned'), (self.prepared['rank0'], 'rank0'),
                   (variants[best_trace_rank], 'trace_best')]
        ordered += [(r['coordinates'], f'rank{i}') for i, r in enumerate(receipt['ranked'][1:], 1)]
        candidates, seen = [], set()
        tuned_priority = 0 if not paired['slower'] else 1
        for coordinates, name in ordered:
            key = coordinate_key(coordinates)
            if key in seen:
                continue
            seen.add(key)
            library = self.build(coordinates)
            gate = self.gate(library)
            priority = tuned_priority if name == 'tuned' else (1-tuned_priority if name == 'rank0' else len(candidates)+2)
            candidates.append(dict(library=str(library), gate=str(gate), bridge=str(self.bridge),
                                   tune_priority=priority, name=name, coordinate_key=key))
            if len(candidates) == 4:
                break
        if len(candidates) != 4:
            raise ValueError('fewer than four distinct P-SEL programs after model fallback')
        spec = self.out / 'selection_candidates.json'
        save(spec, candidates)
        selected = self.measure('selection', Path(candidates[0]['library']), Path(candidates[0]['gate']),
                                'P-SEL', ['--candidate-spec', spec])
        winner = next(c for c in candidates if c['library'] == selected['winner_library'])
        library = Path(winner['library'])
        gate = self.gate(library)
        measured = self.measure('final', library, gate, 'P-DNN')
        coordinates = read(Path(str(library) + '.plan.json'))['native_forward']
        trace1 = self.build(coordinates, trace=1)
        ledger = self.trace('ledger', trace1, samples=1, reference=library)
        self.progress('complete', status='passed', winner=winner,
                      median_ms=measured['candidate_ms'], p90_ms=measured['p90_ms'],
                      process_range_ms=measured['process_range_ms'], ledger=str(ledger / 'ledger.csv'))
        save(self.out / 'result.json', self.result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('resnet18', 'resnet50', 'vgg16', 'bert'), required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--solve', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--slot', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    Cell(args).execute()


if __name__ == '__main__':
    main()
