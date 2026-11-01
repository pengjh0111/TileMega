"""Collect all TP decode stages, preserving raw traces with lossless compression."""
import argparse
import csv
import ctypes as C
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

import torch

from tilemega.serving.measure_tp import capture_harness, require_group
from tilemega.serving.tp_engine import TpServingEngine

STAGES = ('qkv', 'attention', 'o', 'gate_up', 'down')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def compress_trace(path):
    target = path.with_suffix(path.suffix + '.gz')
    if target.exists():
        raise ValueError('preserve existing compressed trace')
    original = hashlib.sha256()
    with path.open('rb') as source, gzip.open(target, 'wb', compresslevel=1) as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            original.update(chunk)
            output.write(chunk)
    restored = hashlib.sha256()
    with gzip.open(target, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            restored.update(chunk)
    if restored.digest() != original.digest():
        raise ValueError('compressed trace does not reproduce its original bytes')
    record = dict(path=str(target), sha256=digest(target), uncompressed_sha256=original.hexdigest())
    path.unlink()
    return record


def read_trace(path, layers=28, grid=104, minimum_slot=64):
    stages = 5 * layers + 1
    bounds = {}
    current, cell = None, {}

    def finish():
        if current is None:
            return
        if set(cell) != set(range(grid)):
            raise ValueError('trace stage lacks a complete CTA cohort')
        bounds[current] = (min(value[0] for value in cell.values()),
                           max(value[2] for value in cell.values()),
                           min(value[1] for value in cell.values()),
                           statistics.median(value[1] for value in cell.values()),
                           max(value[1] for value in cell.values()))

    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for row in csv.DictReader(stream):
            slot, stage, cta = (int(row[key]) for key in ('slot', 'stage', 'cta'))
            if slot < minimum_slot:
                continue
            if not 0 <= cta < grid or not 0 <= stage < stages:
                raise ValueError('trace CTA/stage outside the compiled grid')
            start, work, release = (int(row[key]) for key in ('start_ns', 'work_end_ns', 'release_ns'))
            if not 0 < start <= work <= release:
                raise ValueError('invalid stage timestamp order')
            key = (slot, stage)
            if key != current:
                if current is not None and key <= current:
                    raise ValueError('trace rows are not in slot/stage order')
                finish()
                current, cell = key, {}
            if cta in cell:
                raise ValueError('duplicate trace CTA')
            cell[cta] = (start, work, release)
    finish()
    slots = sorted({slot for slot, _ in bounds})
    samples = {}
    for slot in slots:
        previous = None
        for stage in range(stages):
            if (slot, stage) not in bounds:
                raise ValueError('trace slot lacks a stage')
            start, release, work_min, work_median, work_max = bounds[slot, stage]
            if previous is None:
                previous = start
            duration = release - previous
            if duration <= 0:
                raise ValueError('stage releases do not follow program order')
            samples[slot, stage] = dict(ns=duration, earliest_start=start, latest_release=release,
                wait_entry_min=work_min, wait_entry_median=work_median, wait_entry_max=work_max)
            previous = release
    if not samples:
        raise ValueError('no decode trace samples')
    return samples


def summarize(ranks, layers=28):
    keys = set(ranks[0])
    if any(set(rank) != keys for rank in ranks):
        raise ValueError('rank trace scopes differ')
    rows = []
    for stage in range(5 * layers + 1):
        name = STAGES[stage % 5] if stage < 5 * layers else 'head'
        samples = [key for key in sorted(keys) if key[1] == stage]
        rank_ns = [statistics.median(rank[key]['ns'] for key in samples) for rank in ranks]
        differences = [max(rank[key]['ns'] for rank in ranks) - min(rank[key]['ns'] for rank in ranks)
                       for key in samples]
        rows.append(dict(layer=stage // 5 if name != 'head' else None, stage=name,
                         samples=len(samples), rank0_median_ns=rank_ns[0], rank_median_ns=rank_ns,
                         max_rank_difference_ns=max(differences),
                         median_rank_difference_ns=statistics.median(differences)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'decode-so', 'prompt-ids', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 16), required=True)
    parser.add_argument('--devices', default='0,1,2,3')
    args = parser.parse_args()
    physical = require_group()
    if len(physical) != 4:
        parser.error('final trace scope is TP4')
    metadata = json.loads(Path(str(args.decode_so) + '.plan.json').read_text())
    if not metadata.get('trace_stage'):
        parser.error('a TILEMEGA_TRACE_STAGE=1 binary is required')
    args.out.mkdir(parents=True, exist_ok=False)
    root, harness = capture_harness(args.out)
    source_ref = dict(path=str(Path(__file__)), sha256=digest(Path(__file__)))
    shutil.copy2(__file__, args.out / 'collector.py')
    prompt = json.loads(args.prompt_ids.read_text())[:args.batch]
    prompt_len = len(prompt[0])
    os.environ['TILEMEGA_TRACE_LAUNCHES'] = str(prompt_len + 1024 - 1)
    records, samples = [], []
    with TpServingEngine(args.model, args.decode_so, args.batch, prompt_len, 1024,
                         tuple(map(int, args.devices.split(',')))) as engine:
        result = engine.generate(torch.tensor(prompt, dtype=torch.int32))
        dump = engine.lib.lib.tm_plan_dump_serving_trace
        dump.argtypes, dump.restype = [C.c_void_p, C.c_char_p], C.c_int
        for rank in engine.ranks:
            folder = args.out / f"rank{rank['rank']}"
            folder.mkdir()
            with torch.cuda.device(rank['device']):
                status = dump(rank['plan'].handle, str(folder).encode())
            if status:
                raise RuntimeError('native trace dump failed on rank ' + str(rank['rank']))
            record = compress_trace(folder / 'mx_stage_trace.csv')
            samples.append(read_trace(Path(record['path']), minimum_slot=prompt_len))
            records.append(record)
        identity = engine.ranks[0]['plan'].execution_identity(1, True)
    report = dict(tp=4, batch=args.batch, gpu_indices=physical, prompt_mode=True,
                  trace_slots=prompt_len + 1024 - 1, decode_samples=1023,
                  formula='max CTA release(stage) minus max CTA release(previous stage); first qkv uses min start',
                  stages=summarize(samples), raw_traces=records, execution_identity=identity,
                  host_source_identity=harness, collector=source_ref,
                  e2e_ms_diagnostic=result.e2e_ms, rank_kernel_ms_diagnostic=result.rank_kernel_ms,
                  timing_eligible=False)
    for name, expected in harness['source_files'].items():
        if digest(root / name) != expected:
            raise RuntimeError('host source changed during trace collection')
    if digest(Path(__file__)) != source_ref['sha256']:
        raise RuntimeError('trace collector changed')
    (args.out / 'trace.json').write_text(json.dumps(report, indent=2) + '\n')
    print('TP4 trace complete', args.batch, '1023 decode steps per rank', flush=True)


if __name__ == '__main__':
    main()
