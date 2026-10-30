#!/usr/bin/env python3
"""SM90R4 MPK baseline: R16 protocol on the shared random checkpoint.

`run` claims a node-0 card, queues the three fresh processes of one model on
it and drives the guarded scheduler; `collect` writes SM90R4_mpk.csv from the
raw receipts under runs/. One process covers B1-16 (warmup 1, N1 and N1024 x
3), compiling one persistent kernel per (B, N) before its untimed warmup.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

from sm90d1_common import ROOT, PYTHON, read, write

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
PCI = {0: '0f', 1: '10', 2: '41', 3: '44'}
MPK_PYTHON = '/root/.venvs/mpk/bin/python'
MIRAGE_HOME = '/root/mirage-mpk'
CUDA = '/usr/local/cuda-12.8'
PROMPTS = {'qwen3_0_6b': 'qwen', 'qwen3_1_7b': 'qwen', 'qwen3_30b_a3b': 'qwen',
           'llama3_2_1b': 'llama3', 'llama3_2_3b': 'llama3', 'llama3_1_8b': 'llama3',
           'mistral_7b_v0_3': 'mistral', 'olmoe_1b_7b': 'olmo'}
BATCHES = (1, 2, 4, 8, 16)
OUT = ROOT / 'runs/sm90r4/mpk'


def inputs(model):
    for stage in ('sm90r4', 'sm90d1'):
        path = ROOT / 'configs/serving' / stage / 'inputs' / model
        if path.is_dir():
            return path
    raise FileNotFoundError(model)


def job(model, round_index, gpu, max_tokens=1024, batch='all'):
    uuid = read(POLICY)['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
    baseline = [MPK_PYTHON, '-u', str(ROOT / 'python/tilemega/serving/mpk_baseline.py'),
                '--model', '{attempt_dir}/checkpoint',
                '--prompt-ids', str(ROOT / f'configs/prompts/{PROMPTS[model]}_isl64.json'),
                '--batch', batch, '--max-tokens', str(max_tokens), '--warmup', '1', '--repeats', '3',
                '--round-index', str(round_index), '--load-format', 'auto',
                '--policy', str(ROOT / 'configs/bench/measurement_policy.json'),
                '--out', '{attempt_dir}/job/measurements']
    checkpoint = [PYTHON, str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                  '--config', str(inputs(model)), '--out', '{attempt_dir}/checkpoint', '--seed', '0',
                  '--max-gib', '64', '--', *baseline]
    command = ['numactl', '--cpunodebind=0', PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
               '--guard-policy', str(POLICY), '--out', '{attempt_dir}/job', '--expected-uuid', uuid,
               '--expected-pci', f'0000:{PCI[gpu]}:00.0', '--numa-node', '0', '--', *checkpoint]
    env = dict(TILEMEGA_DEVICE_INDEX=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_NUMA_NODE='0',
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_GPU_PID_MODE='namespace_limited',
               CUDA_HOME=CUDA, CUDA_PATH=CUDA, MIRAGE_HOME=MIRAGE_HOME, TORCH_CUDA_ARCH_LIST='9.0',
               PATH=f'/root/.venvs/mpk/bin:{CUDA}/bin:' + os.environ.get('PATH', '/usr/bin:/bin'),
               LD_LIBRARY_PATH=f'{CUDA}/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8',
               TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', TILEMEGA_NUMA_CONCURRENCY_VERIFIED='0')
    return dict(name=f'r4base_{model}_mpk_r{round_index}_g{gpu}', gpu=True, timing=True,
                timeout_s=6 * 3600, needs_free_mib=74000, cwd=str(ROOT), env=env, command=command,
                priority=8.0, after=[],
                protocol=dict(isl=64, osl=max_tokens, warmup=1, repeats=3, fresh_process_round=round_index,
                              precision='BF16 weights/activations/KV', random_weights=True,
                              checkpoint='tilemega-r16-name-seeded-uniform-v1 seed 0, rebuilt per process',
                              concurrency='trial'))


def run(model, rounds, unit):
    sys.path.insert(0, '/root/TileMega-r4a3/scripts')
    from sm90r4_dispatch import claim
    gpu, handle = claim('trial', f'mpk {model}', nodes=[0])
    try:
        if gpu not in PCI:
            raise RuntimeError(f'claim returned a card outside node 0: {gpu}')
        attempt = unit / f'gpu{gpu}'
        queue = attempt / 'queue'
        queue.mkdir(parents=True, exist_ok=True)
        write(queue / 'queue_mpk.json', [job(model, r, gpu) for r in rounds])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                   TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE='0',
                   PYTHONPATH=str(ROOT / 'python'))
        with (attempt / 'scheduler.log').open('a') as log:
            code = subprocess.call([PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(queue),
                                    '--out', str(attempt / 'scheduler'), '--policy', str(POLICY),
                                    '--deadline-hours', '24'], cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    finally:
        handle.close()
    return gpu, code


NOTES = ('timed scope: prompt copy + MPK init_request + one mpk() call (prefill and decode in one '
         'persistent kernel) + sync; T_N uses a kernel compiled for max_seq_length=64+N; '
         'max_num_batched_tokens=B (demo setting) so prefill is chunked at B tokens per iteration '
         'and interleaved with decode; EOS disabled (eos_token_id=-1); greedy argmax in-kernel')


def collect(root, out, unsupported):
    found = {}
    for path in root.glob('*/gpu*/scheduler/raw/*/attempt*/job/measurements/B*/measurements.json'):
        value = json.loads(path.read_text())
        model = path.parts[path.parts.index('raw') + 1].split('_mpk_')[0].removeprefix('r4base_')
        found.setdefault((model, int(path.parent.name.lstrip('B'))), []).append(value)
    rows = []
    for (model, batch) in sorted(found):
        values = found[(model, batch)]
        checkpoints = {v.get('checkpoint', {}).get('checkpoint_sha256') for v in values}
        tpot = [v['tpot_seconds'] for v in values]
        kernel = [statistics_kernel(v, batch) for v in values]
        rows.append(dict(model=model, system='mpk', batch=batch, fresh_processes=len(values),
                         tpot_ms=f"{st.median(tpot) * 1e3:.4f}",
                         ttft_ms=f"{st.median(v['ttft_seconds'] for v in values) * 1e3:.4f}",
                         e2e_ms=f"{st.median(v['e2e_seconds'] for v in values) * 1e3:.3f}",
                         tokens_per_second=f"{st.median(v['output_tokens_per_second'] for v in values):.3f}",
                         process_range_pct=f"{(max(tpot) / min(tpot) - 1) * 100:.2f}",
                         checkpoint_sha256=checkpoints.pop() if len(checkpoints) == 1 else 'MIXED',
                         versions=json.dumps(dict(values[0].get('versions', {}),
                             mirage=values[0]['configuration'].get('mirage_revision')),
                             separators=(',', ':'))[:160],
                         source='new (R16 protocol, SM90R4)', error='',
                         kernel_tpot_ms=f"{st.median(kernel) * 1e3:.4f}",
                         comparable=comparable(model, batch), protocol_notes=NOTES))
    for row in json.loads(unsupported.read_text()) if unsupported.exists() else []:
        for batch in BATCHES:
            rows.append(dict(model=row['model'], system='mpk', batch=batch, fresh_processes=0,
                             tpot_ms='N/A', ttft_ms='N/A', e2e_ms='N/A', tokens_per_second='N/A',
                             checkpoint_sha256=row.get('checkpoint_sha256', ''),
                             versions=row.get('versions', ''),
                             source='new (R16 protocol, SM90R4)', error=row['error'],
                             comparable='no', protocol_notes='not run: ' + row.get('stage', 'smoke')))
    fields = ['model', 'system', 'batch', 'fresh_processes', 'tpot_ms', 'ttft_ms', 'e2e_ms',
              'tokens_per_second', 'process_range_pct', 'checkpoint_sha256', 'versions', 'source', 'error',
              'kernel_tpot_ms', 'comparable', 'protocol_notes']
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def statistics_kernel(value, batch):
    """TPOT from the demo's own scope: CUDA events around mpk() only."""
    path = Path(value['guard_log']).parent / 'mpk_events.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    # Rows follow the run order: warmup first, then the repeats, per N.
    by_n = {}
    for event in events:
        by_n.setdefault(event['N'], []).append(event['kernel_ms'])
    t1, tn = st.median(by_n[1][1:]), st.median(by_n[1024][1:])
    return (tn - t1) / 1023 / 1e3


# Persistent-kernel iterations between T_1 and T_1024 at max_num_batched_tokens=B, from a
# mirror of MODE_OFFLINE prepare_next_batch (inferred): prefill chunks of B tokens
# interleave with decode, so B>1 spans more iterations than the 1023 TPOT divides by.
DECODE_SPAN = {1: 1023, 2: 1055, 4: 1092, 8: 1132, 16: 1174}


def comparable(model, batch):
    if DECODE_SPAN[batch] == 1023:
        return 'yes'
    return f'no: N1024-N1 spans {DECODE_SPAN[batch]} kernel iterations, not 1023 decode steps'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True, choices=sorted(PROMPTS))
    r.add_argument('--rounds', type=int, nargs='+', default=[0, 1, 2])
    c = sub.add_parser('collect')
    c.add_argument('--out', type=Path, default=ROOT / 'docs/results/SM90R4_mpk.csv')
    a = p.parse_args()
    if a.action == 'run':
        print(json.dumps(dict(zip(('gpu', 'code'), run(a.model, a.rounds, OUT / a.model)))))
    else:
        rows = collect(OUT, a.out, OUT / 'unsupported.json')
        print(json.dumps({'rows': len(rows)}))


if __name__ == '__main__':
    main()
