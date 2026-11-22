#!/usr/bin/env python3
"""Freeze and remeasure the accepted A800 matrix with independent requests."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from a100_baselines import BATCHES, MODELS, ROOT, inputs, sha
from a800_baselines import RUN, hardware_class
from a800_native import artifact
from a800_shortlist_abi import freeze_host_preparation
from tilemega.bench.scheduler import write_json


PROTOCOL = 'a800-single-v2'
IDENTITY = ('decode_sha256', 'prefill_sha256', 'target_sha256')
RUNNER = ROOT/'scripts/a800_native.py'


def read(path):
    return json.loads(Path(path).read_text())


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def matching(document, identity):
    return all(document.get(key) == identity[key] for key in IDENTITY)


def process_timeout(model):
    if model == 'qwen3_30b_a3b':
        return 1080
    if model in ('llama3_1_8b', 'mistral_7b_v0_3'):
        return 570
    return 420


def prepare(args):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('prepare the matrix with GPU access disabled on the login node')
    anchors, tokens = read(args.anchors), read(args.tokens)
    cells = {f'{model}/B{batch}' for model in MODELS for batch in BATCHES}
    with args.incumbents.open() as stream:
        incumbents = {f"{r['model']}/B{r['batch']}": r for r in csv.DictReader(stream)}
    if set(anchors) != cells or set(tokens) != cells or set(incumbents) != cells:
        raise ValueError('the frozen native workload must contain all 40 cells')
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in sorted(cells):
        path = args.out/cell/'matrix.plan.json'
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        old, anchor = incumbents[cell], read(anchors[cell])
        decode, info, manifest = artifact(Path(anchor['decode']))
        prefill = anchor['prefill']
        prefix = () if prefill == 'prompt' else (artifact(Path(prefill))[0],)
        target = Path(anchor['artifact_target'])
        identity = dict(decode_sha256=info['binary_sha256'], target_sha256=sha(target),
                        prefill_sha256=sha(prefix[0]) if prefix else 'prompt')
        if (not matching(old, identity) or info['target_sha256'] != identity['target_sha256']
                or manifest['native_origin'] != 'solver'
                or manifest['native_selection']['selection'] != 'measure'
                or old['status'] != 'measured'):
            raise ValueError(f'{cell} differs from the accepted native artifacts')
        model, batch = anchor['model'], anchor['batch']
        if cell != f'{model}/B{batch}':
            raise ValueError('mismatched anchor model or batch')
        dev = read(target)['device_provenance']
        proofs = {}
        for root in args.history:
            for file in sorted(root.glob(f'*/{cell}/*/correctness-*.json')):
                row = read(file)
                if (row['pass'] and row['protocol_version'] == PROTOCOL
                        and row['tokens_sha256'] == tokens[cell] and matching(row, identity)
                        and hardware_class(row['device']) == hardware_class(dev)):
                    proofs.setdefault(row['round'], file.resolve())
        if not any(read(file)['hf_pass'] for file in proofs.values()):
            raise ValueError(f'{cell} has no accepted full HF receipt for the identical artifacts')
        host = path.with_name('host-preparation.json')
        config, prompt = inputs(model)
        if sha(config/'config.json') != old['config_sha256'] or sha(prompt) != old['prompt_sha256']:
            raise ValueError(f'{cell} changed the frozen inputs')
        freeze_host_preparation(read(config/'config.json'), batch, (*prefix, decode), host)
        evidence = list(proofs.values())
        write_json(path, dict(protocol_version=PROTOCOL, head=head(), model=model, batch=batch,
            anchor_sha256=sha(Path(anchors[cell])), **identity, tokens_sha256=tokens[cell],
            checkpoint_sha256=old['checkpoint_sha256'], host_preparation=str(host),
            decode=str(decode), prefill=prefill, target=str(target),
            process_timeout_s=process_timeout(model), correctness=[str(p) for p in evidence],
            assets={str(p.resolve()): sha(p) for p in (RUNNER, host, config/'config.json', prompt, *evidence)}))
        print(path, sha(path), flush=True)


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800':
        raise ValueError('matrix timing requires an ordinary single-A800 allocation')
    plan = read(args.plan)
    if (sha(args.plan) != args.plan_sha256 or plan['protocol_version'] != PROTOCOL
            or plan['head'] != head()
            or any(sha(Path(path)) != expected for path, expected in plan['assets'].items())):
        raise ValueError('the matrix plan or its inputs changed in queue')
    deadline = time.monotonic() + args.budget_s
    model, batch = plan['model'], plan['batch']
    cell = f'{model}/B{batch}'
    _, linked, _ = artifact(Path(plan['decode']))
    prefill = plan['prefill']
    if (linked['binary_sha256'] != plan['decode_sha256'] or sha(Path(plan['target'])) != plan['target_sha256']
            or (sha(Path(prefill)) if prefill != 'prompt' else 'prompt') != plan['prefill_sha256']):
        raise ValueError('the timing artifacts differ from the published pair')
    key = hashlib.sha256(json.dumps([plan[k] for k in ('target_sha256', *IDENTITY[:2])],
                                   separators=(',', ':')).encode()).hexdigest()[:20]
    for original in plan['correctness']:
        receipt = read(original)
        folder = RUN/'native'/receipt['device_uuid']/cell/key
        folder.mkdir(parents=True, exist_ok=True)
        proof = folder/f"correctness-{receipt['round']}.json"
        if not proof.exists():
            proof.symlink_to(Path(original))
        if sha(proof) != plan['assets'][original]:
            raise ValueError('the linked correctness evidence has different contents')
    job = os.environ['SLURM_JOB_ID']
    index = MODELS.index(model) * len(BATCHES) + BATCHES.index(batch)
    round_index = int(job) * len(MODELS) * len(BATCHES) + index
    remaining = int(deadline - time.monotonic())
    command = [sys.executable, str(RUNNER), 'run', '--stage', 'candidate-timing', '--model', model,
               '--batch', str(batch), '--round', str(round_index), '--budget-s', str(remaining),
               '--timeout-s', str(plan['process_timeout_s']), '--decode', plan['decode'],
               '--prefill', prefill, '--target', plan['target'], '--host-preparation', plan['host_preparation'],
               '--expected-checkpoint-sha256', plan['checkpoint_sha256']]
    subprocess.run(command, cwd=ROOT, check=True, timeout=remaining)
    evidence, = (RUN/'native').glob(f'*/{cell}/{key}/candidate-timing-{round_index}.json')
    result = read(evidence)
    if (not result['pass'] or result['job'] != job or not matching(result, plan)
            or result['tokens_sha256'] != plan['tokens_sha256']):
        raise ValueError(f'{cell} did not reproduce the accepted model output')
    write_json(args.plan.with_name(f'matrix-{job}.receipt.json'),
               dict(plan_sha256=args.plan_sha256, timing=str(evidence), timing_sha256=sha(evidence)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    p = commands.add_parser('prepare')
    for name in ('anchors', 'tokens', 'incumbents', 'out'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--history', type=Path, action='append', required=True)
    p.set_defaults(func=prepare)
    p = commands.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--budget-s', type=int, required=True)
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
