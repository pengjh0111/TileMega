"""Shared measurement loop and MetaX exclusivity guard for the baseline harnesses.

Standard library only, and loaded by path from the harness directory, so a
baseline environment (vLLM, SGLang, torch.compile) never imports tilemega.
The protocol is R16 M-1: per batch, N=OSL and N=1 runs, `warmup` untimed and
`repeats` timed generations each; TTFT from N=1, E2E from N=OSL,
TPOT=(E2E-TTFT)/(OSL-1).
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import time
from pathlib import Path


def _mx_smi():
    path = Path(__file__).resolve().parent.parent / 'bench' / 'mx_smi.py'
    spec = importlib.util.spec_from_file_location('tilemega_mx_smi', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_root(pid, proc=Path('/proc')):
    current = pid
    while current > 1:
        try:
            base = proc / str(current)
            argv = (base / 'cmdline').read_bytes().split(b'\0')
            if b'tilemega.bench.gpu_guard' in argv:
                return current
            current = int((base / 'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pid


def own_pids() -> set[int]:
    """This process and its descendants (engines may run the GPU in a child)."""
    # Elastic starts each rank in its own session; use the shared guard tree.
    root = job_root(os.getpid()) if os.getenv('TILEMEGA_DEVICE_GROUP') else os.getpid()
    own = {root}
    while True:
        added = set()
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[1]) in own:
                    added.add(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
        if added <= own:
            if os.getenv('TILEMEGA_DEVICE_GROUP'):
                # torchrun ranks are siblings in the guard-created session.
                sid = os.getsid(0)
                for entry in Path('/proc').glob('[0-9]*/stat'):
                    try:
                        if int(entry.read_text().rsplit(')', 1)[1].split()[3]) == sid:
                            own.add(int(entry.parent.name))
                    except (OSError, ValueError, IndexError):
                        continue
            return own
        own.update(added)


def exclusive_gpu(path: Path, label: str, fail_fast: bool = False, hidden_limit_mib: float = 1024) -> bool:
    """mx-smi lists only this container's processes; another tenant shows up
    as memory above the idle baseline that no listed process owns."""
    smi = _mx_smi()
    indices = os.environ.get('TILEMEGA_DEVICE_GROUP',
                             os.environ.get('TILEMEGA_DEVICE_INDEX', '0')).split(',')
    deadline = time.monotonic() if fail_fast else time.monotonic() + 30 * 60
    while True:
        exclusive = True
        own = own_pids()
        for index in indices:
            owners, visible, above_idle = smi.occupancy(index)
            hidden = max(0.0, above_idle - visible)
            accepted = set(owners) <= own and hidden <= hidden_limit_mib
            exclusive &= accepted
            with path.open('a') as stream:
                stream.write(json.dumps({'label': label, 'gpu_index': index,
                                         'pids': sorted(owners), 'visible_mib': visible,
                                         'hidden_mib': hidden, 'exclusive': accepted,
                                         'time': time.time()}) + '\n')
        if exclusive or time.monotonic() >= deadline:
            return exclusive
        time.sleep(10)


def measure(out_root: Path, batches, max_tokens: int, warmup: int, repeats: int, policy,
            generate, meta: dict) -> None:
    """`generate(batch, count)` runs one generation and returns the token rows;
    it must assert prompt order itself. Rows are checked for count here."""
    for batch in batches:
        out = out_root / f'B{batch}'
        out.mkdir(parents=True, exist_ok=True)
        guard_path = out / 'guard.jsonl'
        rows = []
        for count in (max_tokens, 1):
            for run in range(warmup + repeats):
                for attempt in range(3):
                    if policy.options.get('guard', True) and not exclusive_gpu(
                            guard_path, f'N{count}-run{run}-before'):
                        raise SystemExit(75)
                    if not policy.observe(guard_path, f'N{count}-run{run}-attempt{attempt}-before'):
                        continue
                    start = time.perf_counter()
                    tokens = generate(batch, count)
                    elapsed = time.perf_counter() - start
                    exclusive = not policy.options.get('guard', True) or exclusive_gpu(
                        guard_path, f'N{count}-run{run}-after', fail_fast=True)
                    power_ok = policy.observe(guard_path, f'N{count}-run{run}-attempt{attempt}-after')
                    if exclusive and power_ok:
                        break
                else:
                    raise SystemExit(75)
                if len(tokens) != batch or any(len(row) != count for row in tokens):
                    raise AssertionError('baseline did not generate the requested tokens')
                token_file = f'tokens_N{count}_run{run}.json'
                rows.append({'N': count, 'run': run, 'warmup': run < warmup,
                             'wall_seconds': elapsed, 'tokens_file': token_file})
                (out / token_file).write_text(json.dumps(tokens, separators=(',', ':')) + '\n')
        timed = lambda n: [r['wall_seconds'] for r in rows if r['N'] == n and not r['warmup']]
        e2e = statistics.median(timed(max_tokens))
        ttft = statistics.median(timed(1))
        indices = os.environ.get('TILEMEGA_DEVICE_GROUP',
                                 os.environ.get('TILEMEGA_DEVICE_INDEX', '0')).split(',')
        summary = {**meta, 'tp': int(meta.get('tp', len(indices))),
                   'gpu_indices': list(map(int, indices)),
                   'batch': batch, 'max_tokens': max_tokens,
                   'e2e_seconds': e2e, 'ttft_seconds': ttft,
                   'tpot_seconds': (e2e - ttft) / (max_tokens - 1),
                   'output_tokens_per_second': batch * max_tokens / e2e,
                   'generation_runs': rows, 'measurement_policy': policy.options}
        (out / 'measurements.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'generation_runs'}), flush=True)
