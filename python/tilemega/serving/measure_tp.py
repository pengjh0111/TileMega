"""Measure a guarded TP cohort with prompt-mode CUDA TTFT and wall E2E."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import time

import torch

from .baseline_common import exclusive_gpu
from .measurement_policy import TimingPolicy
from .tp_engine import TpServingEngine
from .workload import validate_prompt_ids


def require_group():
    text = os.getenv('TILEMEGA_DEVICE_GROUP', '')
    if os.getenv('TILEMEGA_GPU_LOCK_HELD') != '1' or not text:
        raise RuntimeError('TP requires scripts/mx_gpu_group_job.sh')
    physical = list(map(int, text.split(',')))
    if len(physical) not in (2, 4) or len(set(physical)) != len(physical) or not set(physical) <= set(range(8)) or any(
            os.getenv(name) != text for name in ('CUDA_VISIBLE_DEVICES', 'MACA_VISIBLE_DEVICES')):
        raise ValueError('invalid guarded physical GPU group')
    return physical


def clocks():
    from tilemega.bench import mx_smi
    result = {}
    for gpu in os.environ['TILEMEGA_DEVICE_GROUP'].split(','):
        power, xcore, memory, temperature = mx_smi.telemetry(gpu)
        result[gpu] = dict(power_w=power, xcore_mhz=xcore, memory_mhz=memory, temperature_c=temperature)
    return result


def capture_harness(out):
    root = Path(__file__).resolve().parents[3]
    files = list((root / 'python/tilemega/serving').glob('*.py'))
    files += [root / 'python/tilemega' / name for name in (
        'build/identity.py', 'bench/mx_smi.py', 'bench/gpu_guard.py', 'moe/weights.py', 'dnn/weights.py')]
    archive = out / 'host_source'
    archive.mkdir()
    hashes = {}
    for path in files:
        name = path.relative_to(root)
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  source_files=hashes, captured_at_ns=time.time_ns())
    (out / 'host_source.json').write_text(json.dumps(record, indent=2) + '\n')
    return root, record


def measure(engine, prompts, out, warmup=1, repeats=2, policy_path=None):
    out.mkdir(parents=True, exist_ok=False)
    source_root, harness = capture_harness(out)
    policy = TimingPolicy(policy_path)
    guard = out / 'guard.jsonl'
    rows = []
    counts = (engine.max_new_tokens, 1) if engine.max_new_tokens > 1 else (1,)
    for count in counts:
        for run in range(warmup + repeats):
            for attempt in range(3):
                if policy.options.get('guard', True) and not exclusive_gpu(guard, f'N{count}-run{run}-before'):
                    raise SystemExit(75)
                if not policy.observe(guard, f'N{count}-run{run}-attempt{attempt}-before'):
                    continue
                before = clocks()
                try:
                    result = engine.generate(prompts, count)
                except BaseException:
                    (out / 'watchdog.json').write_text(json.dumps(engine.last_failure, indent=2) + '\n')
                    raise
                after = clocks()
                good = policy.observe(guard, f'N{count}-run{run}-attempt{attempt}-after')
                if (not policy.options.get('guard', True) or exclusive_gpu(
                        guard, f'N{count}-run{run}-after', fail_fast=True)) and good:
                    break
            else:
                raise SystemExit(75)
            tokens = result.tokens.tolist()
            token_file = out / f'tokens_N{count}_run{run}.json'
            token_file.write_text(json.dumps(tokens, separators=(',', ':')) + '\n')
            rows.append(dict(N=count, run=run, warmup=run < warmup,
                             e2e_seconds=result.e2e_ms / 1000, ttft_seconds=result.ttft_ms / 1000,
                             gpu_step_ms=[], decode_loop_used=True, step_ns_read=False,
                             execution_identity=engine.ranks[0]['plan'].execution_identity(1, True),
                             rank_kernel_ms=result.rank_kernel_ms, rank_tokens_identical=True,
                             clocks_before=before, clocks_after=after,
                             tokens=tokens, tokens_file=str(token_file)))
    full = [r for r in rows if r['N'] == engine.max_new_tokens and not r['warmup']]
    first = [r for r in rows if r['N'] == 1 and not r['warmup']]
    same = all(r['tokens'] == full[0]['tokens'] for r in full)
    if not same:
        raise AssertionError('C-2: timed TP generations produced different tokens')
    e2e = statistics.median(r['e2e_seconds'] for r in full)
    ttft = statistics.median(r['ttft_seconds'] for r in full)
    for name, expected in harness['source_files'].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('host source changed during TP timing: ' + name)
    summary = dict(host_source_identity=harness, tp=engine.tp, gpu_indices=require_group(), prompt_mode=True,
                   batch=engine.batch, max_tokens=engine.max_new_tokens,
                   timed_tokens_identical=same, rank_tokens_identical=True,
                   execution_identity=engine.ranks[0]['plan'].execution_identity(1, True),
                   decode_loop_used=True, step_ns_read=False, memory=engine.memory_report,
                   measurement_policy=policy.options, e2e_seconds=e2e, ttft_seconds=ttft,
                   ttft_source='rank0 CUDA event, prompt-mode past0..63 in the full request',
                   n1_wall_seconds=statistics.median(r['e2e_seconds'] for r in first),
                   tpot_seconds=(e2e - ttft) / (engine.max_new_tokens - 1) if engine.max_new_tokens > 1 else 0.,
                   output_tokens_per_second=engine.batch * engine.max_new_tokens / e2e,
                   formula='(wall E2E - prompt-mode CUDA TTFT)/(OSL-1)', timing_eligible=True, runs=rows)
    (out / 'measurements.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'decode-so', 'prompt-ids', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    physical = require_group()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError('positive repeats and nonnegative warmup required')
    manifest = Path(str(args.decode_so) + '.plan.json')
    meta = json.loads(manifest.read_text())
    if meta.get('watchdog') or meta.get('trace_stage') or meta.get('diag') or meta.get('tp_litmus'):
        raise ValueError('TP timing requires watchdog0 and no trace/DIAG/TP litmus')
    ids = json.loads(args.prompt_ids.read_text())
    prompt_len = validate_prompt_ids(ids, args.batch)
    devices = tuple(map(int, args.devices.split(',')))
    if len(devices) != len(physical):
        raise ValueError('logical device count differs from guarded physical group')
    prompts = torch.tensor(ids[:args.batch], dtype=torch.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    before = args.out.parent / (args.out.name + '_before_load.jsonl')
    if not exclusive_gpu(before, 'before-weight-load', fail_fast=True):
        raise SystemExit(75)
    with TpServingEngine(args.model, args.decode_so, args.batch, prompt_len=prompt_len,
                         max_new_tokens=args.max_new_tokens, devices=devices) as engine:
        result = measure(engine, prompts, args.out, args.warmup, args.repeats, args.policy)
    print(json.dumps({key: value for key, value in result.items() if key != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
