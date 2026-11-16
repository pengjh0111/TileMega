#!/usr/bin/env python3
"""Collect a two-arm AB4b/AB4c supplement with the retained R6 toolchain.

Receipts come from sm90r6_search_arms with realize_aggregate and a 600 s
budget. Keep the original eight-arm collectors and their hashed inputs intact.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_dispatch import run_unit
from sm90r6_audit_final_measurements import audit_directory

ARMS = ('AB4b', 'AB4c')


def call(command, log):
    command = list(map(str, command))
    write(log.with_suffix('.command.json'), command)
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)


def validate(arms, model, batch):
    if set(arms) != set(ARMS):
        raise ValueError('the supplement requires exactly AB4b and AB4c')
    identities = {}
    for label, arm in arms.items():
        digest = sha(arm['binary'])
        identity = read(arm['binary'] + '.identity.json')
        identities[label] = identity
        c2 = read(arm['c2'])
        c1 = read(arm['c1'])
        spec = c1['spec']
        if identity['binary_sha256'] != digest:
            raise ValueError('binary identity changed: ' + label)
        if (not c2['passed'] or len(c2['processes']) != 5 or c2['steps'] != 1024 or
                c2['model'] != model or c2['batch'] != batch or
                c2['artifacts']['candidate']['sha256'] != digest):
            raise ValueError('missing selected-binary C2 evidence: ' + label)
        if (not c1['pass'] or not c1['c1_pass'] or spec['correctness']['c1'] != 256 or
                spec['model'] != model or spec['batch'] != batch or
                {v['decode_sha256'] for v in spec['arms'].values()} != {digest}):
            raise ValueError('missing selected-binary C1 evidence: ' + label)
    reference, candidate = (identities[k] for k in ARMS)
    if (candidate['plan'].get('execution') != 'kernel_sequence' or
            any(reference['plan'][key] != candidate['plan'][key] for key in
                ('native_dense', 'export_sha256', 'grid', 'runtime_batch')) or
            reference['shared_memory_bytes'] != candidate['shared_memory_bytes']):
        raise ValueError('AB4c changed the selected AB4b program or launch geometry')
    if read(arms['AB4c']['c2'])['artifacts']['reference']['sha256'] != sha(arms['AB4b']['binary']):
        raise ValueError('AB4c C2 reference is not the selected AB4b')
    return {k: dict(decode=v['binary'], prefill='prompt') for k, v in arms.items()}


def correctness(cell, label, binary, reference, out, gpus):
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'c2/result.json').exists():
        prompt = ('olmo' if cell['model'].startswith('olmo') else
                  'llama3' if cell['model'].startswith('llama') else 'qwen')
        call([PYTHON, ROOT / 'scripts/sm90r6_run_realize.py', '--model', cell['model'],
              '--batch', cell['batch'], '--reference', reference, '--candidate', binary,
              '--prompts', ROOT / f'configs/prompts/{prompt}_isl64.json',
              '--out', out / 'c2', '--gpus', *gpus], out / 'c2.log')
    if not read(out / 'c2/result.json')['passed']:
        raise ValueError('incomplete C2; preserve the attempt before retrying')
    if not (out / 'c1.json').exists():
        spec = dict(name='sm90r6-ab4-supplement-c1-' + label + '-' + cell['id'],
                    kind='corr', model=cell['model'], batch=cell['batch'], gpu='auto',
                    concurrency='trial', arms={'winner': dict(decode=str(binary), prefill='prompt')},
                    correctness=dict(c1=256, reference_processes=1), order=['winner'])
        write(out / 'c1-spec.json', spec)
        write(out / 'c1.json', run_unit(out / 'c1-spec.json', out / 'c1-unit', 'trial'))
    if not read(out / 'c1.json')['pass']:
        raise ValueError('C1 failed: ' + label)
    return dict(binary=str(binary), c2=str(out / 'c2/result.json'), c1=str(out / 'c1.json'))


def timing(cell, arms, out):
    validated = validate(arms, cell['model'], cell['batch'])
    out.mkdir(parents=True, exist_ok=True)
    sequence = [label for r in range(3) for label in ARMS[r % 2:] + ARMS[:r % 2]]
    if not (out / 'inputs.json').exists():
        paths = [Path(__file__), out.parent / 'arms.json']
        for arm in arms.values():
            paths += [Path(arm[k]) for k in ('binary', 'c1', 'c2')]
            paths.append(Path(arm['binary'] + '.identity.json'))
        write(out / 'inputs.json', {str(p.resolve()): sha(p) for p in paths})
        spec = dict(name='sm90r6-ab4-supplement-' + cell['id'], kind='sel',
                    protocol='P-ABL two-arm supplement', model=cell['model'], batch=cell['batch'],
                    gpu='auto', concurrency='trial', arms=validated, order=sequence)
        write(out / 'spec.json', spec)
    if any(sha(p) != digest for p, digest in read(out / 'inputs.json').items()):
        raise ValueError('supplement inputs changed')
    raw = run_unit(out / 'spec.json', out / 'unit', 'trial')
    if any(v['processes'] != 3 for v in raw['arms'].values()):
        raise ValueError('expected three fresh processes per arm')
    audit = audit_directory(out, sequence, cell['model'], cell['batch'])
    # The comparison concerns kernel boundaries, so both arms must also emit
    # identical full outputs in these accepted timing processes.
    tokens = [read(r['tokens']) for r in raw['rows']]
    if any(value != tokens[0] for value in tokens[1:]):
        raise ValueError('cross-arm timing token mismatch')
    reference = raw['arms']['AB4b']
    comparisons = {}
    for label, value in raw['arms'].items():
        change = (value['tpot_ms'] / reference['tpot_ms'] - 1) * 100
        threshold = max(.5, 2 * max(reference['range_pct'], value['range_pct']))
        comparisons[label] = dict(change_pct=change, threshold_pct=threshold,
            decision='indistinguishable' if abs(change) <= threshold else
                     'faster' if change < 0 else 'slower')
    write(out / 'audit.json', audit)
    write(out / 'result.json', dict(protocol='P-ABL two-arm supplement',
        scope='AB4b/AB4c only; no A0-relative or full eight-arm claim',
        model=cell['model'], batch=cell['batch'], arms=raw['arms'], comparisons=comparisons,
        order=sequence, gpu=audit['gpu'], checkpoint=audit['checkpoint'], evidence='verified',
        ownership='provisional_container_pid_namespace', concurrency='trial',
        cross_arm_tokens_equal=True, raw=str(out / 'unit/result.json')))


def collect(cell, out, compiler, gpus):
    directory = out / cell['id']
    try:
        receipt = directory / 'receipt.json'
        deadline = time.monotonic() + 1200
        while not receipt.exists():
            if time.monotonic() > deadline:
                raise ValueError('offline aggregate search receipt did not arrive')
            time.sleep(5)
        request = read(receipt)['input']
        if request != cell or cell['arm'] != 'AB4b' or cell['budget_ms'] != 600000:
            raise ValueError('receipt differs from the requested aggregate search')
        build = directory / 'build'
        if not (build / 'result.json').exists():
            call([PYTHON, ROOT / 'scripts/sm90r6_compile_finalists.py', '--receipt', receipt,
                  '--compiler', compiler, '--out', build], directory / 'finalists.log')
        if read(build / 'result.json')['status'] != 'ready_for_P_SEL':
            raise ValueError('finalists need occupancy rescore')
        selected = directory / 'psel'
        if not (selected / 'result.json').exists():
            paths = [receipt, build / 'candidates.json', ROOT / 'scripts/sm90r3_select.py',
                     ROOT / 'scripts/sm90r3_llm.py']
            for candidate in read(build / 'candidates.json'):
                if candidate['status'] == 'compiled':
                    paths += [Path(candidate['binary']), Path(candidate['binary'] + '.identity.json')]
            write(directory / 'selection-inputs.json', {str(p): sha(p) for p in paths})
            call([PYTHON, ROOT / 'scripts/sm90r3_select.py', '--model', cell['model'],
                  '--batch', cell['batch'], '--prefill', 'prompt', '--gpu', 'auto',
                  '--name', 'sm90r6-ab4-supplement-' + cell['id'], '--concurrency', 'trial',
                  '--candidates', build / 'candidates.json', '--out', selected], directory / 'psel.log')
        write(selected / 'inputs.json', read(directory / 'selection-inputs.json'))
        labels = [f'c{v["rank"]}' for v in read(build / 'candidates.json') if v['status'] == 'compiled']
        if len(labels) != 4:
            raise ValueError('this supplement requires all four finalists for P-SEL')
        write(selected / 'audit.json', audit_directory(selected, labels + labels[::-1],
                                                       cell['model'], cell['batch']))
        winner = read(selected / 'result.json')
        candidate = build / f'candidate-{winner["winner_rank"]}'
        binary = candidate / 'decode.so'
        if sha(binary) != winner['winner_binary_sha256']:
            raise ValueError('P-SEL winner binary changed')
        sequence = directory / 'AB4c'
        sequence.mkdir(exist_ok=True)
        seqbinary = sequence / 'decode.so'
        if not Path(str(seqbinary) + '.compile-result.json').exists():
            call([compiler, 'compile', cell['export'], seqbinary, '--tmexec-config',
                  candidate / 'coordinates.json', '--runtime-target', ROOT / 'configs/targets/sm_90_h800.json',
                  '--batch', cell['batch'], '--serving', 'decode', '--emit', 'kernel-sequence',
                  '--nvcc-split-compile', '4'], sequence / 'compile.log')
        status = read(str(seqbinary) + '.compile-result.json')
        if status['status'] != 'compiled' or not status['inputs_unchanged']:
            raise ValueError('unstable sequence build')
        print(cell['id'] + ' selected; sequence compiled', flush=True)
        arms = {}
        for label, artifact in (('AB4b', binary), ('AB4c', seqbinary)):
            arms[label] = correctness(cell, label, artifact, binary, directory / label, gpus)
            print(cell['id'] + ' ' + label + ' C1/C2 passed', flush=True)
        write(directory / 'arms.json', arms)
        timing(cell, arms, directory / 'measurement')
        write(directory / 'complete.json', dict(status='complete', cell=cell, winner=winner,
              measurement=str(directory / 'measurement/result.json')))
        print(cell['id'] + ' timing audited; complete', flush=True)
        return True
    except Exception as error:
        write(directory / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        print(cell['id'] + ' FAILED ' + str(error), flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--models', nargs='+', help='resume only these model cells')
    args = parser.parse_args()
    cells = read(args.out / 'cells.json')
    if args.models:
        cells = [c for c in cells if c['model'] in args.models]
        if {c['model'] for c in cells} != set(args.models):
            raise ValueError('unknown model in the supplement manifest')
    source = read(args.out / 'source.json')
    if any(sha(path) != digest for path, digest in source['source_hashes'].items()):
        raise ValueError('retained toolchain or target changed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda c: collect(c, args.out.resolve(), args.compiler.resolve(), args.gpus), cells))
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
