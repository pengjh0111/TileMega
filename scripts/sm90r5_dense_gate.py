#!/usr/bin/env python3
"""The dense gate (prompt §1.4): MoE and TP work must not move dense output.

Two R3 cells are regenerated at their recorded coordinates and compared with the
baseline: the emitted CUDA source byte for byte, and the predicted solver ranking
entry for entry. `ref` records the baseline; `check` re-runs and diffs. nvcc is
never invoked and no GPU is touched (CUDA_VISIBLE_DEVICES is cleared), so the
gate costs one host compile per cell.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, read, write
from sm90r4_inputs import extract

CELLS = ('llama3_1_8b_b1', 'qwen3_0_6b_b16')
TARGET = ROOT / 'configs/targets/sm_90_h800.json'
COMPILER = ROOT / 'build/tools/tilemega'
REF = ROOT / 'runs/sm90r5/dense_gate/ref'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_source(graph, source, coordinates, batch):
    command = [str(COMPILER.resolve()), 'compile', str(graph), str(source),
               '--tmexec-config', str(coordinates), '--serving', 'decode', '--emit', 'serving',
               '--batch', str(batch), '--runtime-target', str(TARGET)]
    run(command, source.with_suffix('.emit.log'))


def solve_ranked(graph, source, batch):
    command = [str(COMPILER.resolve()), 'compile', str(graph), str(source),
               '--solve', str(TARGET), '--serving', 'decode', '--emit', 'serving',
               '--batch', str(batch), '--runtime-target', str(TARGET),
               '--selection', 'predicted', '--measure-top', '4']
    command += ['--search', 'additive']
    run(command, source.with_suffix('.solve.log'))
    receipt = Path(str(source) + '.search') / 'receipt.json'
    return read(receipt)['ranked']


def run(command, log):
    with Path(log).open('w') as stream:
        status = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    if status:
        raise RuntimeError(f'{command[2]} failed; see {log}')


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for cell in CELLS:
        config = read(ROOT / 'configs/serving/sm90r3' / f'{cell}.json')
        batch = config['batch']
        coordinates = directory / f'{cell}.tmexec.json'
        write(coordinates, config['coordinates'])
        graph = directory / f'{cell}.export.json'
        extract(config['model'], 'decode', graph)
        source = directory / f'{cell}.cu'
        compile_source(graph, source, coordinates, batch)
        ranked = solve_ranked(graph, directory / f'{cell}-solve.cu', batch)
        result[cell] = {'model': config['model'], 'batch': batch, 'source_sha256': sha(source),
                        'ranked': ranked}
    return result


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def pipeline_gate(out, compiler, reference):
    """R6 P4: identical frozen coordinates through both compiler paths."""
    from concurrent.futures import ThreadPoolExecutor
    cells = [(model,batch) for model in ('qwen3_1_7b','llama3_1_8b',
              'qwen3_30b_a3b','olmoe_1b_7b') for batch in (1,16)]
    out.mkdir(parents=True, exist_ok=False)
    def check(cell):
        model,batch=cell; directory=out/f'{model}-B{batch}';directory.mkdir()
        if model=='olmoe_1b_7b':
            frozen=Path(f'/root/TileMega-r5/runs/sm90r5/a1/olmoe-B{batch}/winner.tmexec.json')
            coordinates=read(frozen)
        else:
            root='sm90r5/ref_r4' if model=='qwen3_30b_a3b' else 'sm90r3'
            frozen=ROOT/f'configs/serving/{root}/{model}_b{batch}.json'
            config=read(frozen);coordinates=config.get('coordinates',config)
        binding=directory/'coordinates.json';write(binding,coordinates)
        graph=extract(model,'decode',directory/'decode.export.json')
        sources=[]
        for name,binary in [('reference',reference),('current',compiler)]:
            source=directory/f'{name}.cu'
            command=[str(binary.resolve()),'compile',str(graph.resolve()),str(source.resolve()),
                     '--tmexec-config',str(binding.resolve()),'--serving','decode','--emit','serving',
                     '--batch',str(batch),'--runtime-target',str(TARGET)]
            run(command,directory/f'{name}.log');sources.append(source)
        identical=sources[0].read_bytes()==sources[1].read_bytes()
        result=dict(model=model,batch=batch,passed=identical,frozen_source=str(frozen),
                    frozen_source_sha256=sha(frozen),coordinates_sha256=sha(binding),
                    reference_sha256=sha(sources[0]),current_sha256=sha(sources[1]))
        write(directory/'result.json',result)
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(check,cells))
    receipt=dict(evidence='verified',head=head(),reference_compiler_sha256=sha(reference),
                 current_compiler_sha256=sha(compiler),cells=results,passed=all(r['passed'] for r in results))
    write(out/'manifest.json',receipt)
    return 0 if receipt['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('ref', 'check', 'pipeline'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compiler', type=Path, default=COMPILER)
    p.add_argument('--reference-compiler', type=Path)
    a = p.parse_args()
    if a.action=='pipeline':
        if not a.reference_compiler:p.error('pipeline requires --reference-compiler')
        return pipeline_gate(a.out,a.compiler,a.reference_compiler)
    cells = build(a.out)
    manifest = {'head': head(), 'cells': cells}
    if a.action == 'ref':
        REF.mkdir(parents=True, exist_ok=True)
        write(REF / 'manifest.json', manifest)
        print(json.dumps({'reference': str(REF / 'manifest.json'), 'head': manifest['head'],
                          'sources': {c: cells[c]['source_sha256'][:16] for c in CELLS}}))
        return 0
    reference = read(REF / 'manifest.json')
    write(a.out / 'manifest.json', manifest)
    failures = []
    for cell in CELLS:
        want, got = reference['cells'][cell], cells[cell]
        if want['source_sha256'] != got['source_sha256']:
            failures.append(f'{cell}: generated source changed')
            diff = subprocess.run(['diff', '-u', str(REF.parent / 'ref' / f'{cell}.cu'),
                                   str(a.out / f'{cell}.cu')], capture_output=True, text=True)
            print('\n'.join(diff.stdout.splitlines()[:40]))
        if json.dumps(want['ranked'], sort_keys=True) != json.dumps(got['ranked'], sort_keys=True):
            failures.append(f'{cell}: predicted ranking changed')
            for i, (x, y) in enumerate(zip(want['ranked'], got['ranked'])):
                if x != y:
                    print(f'{cell}: ranked[{i}] reference {json.dumps(x)[:200]}')
                    print(f'{cell}: ranked[{i}] current   {json.dumps(y)[:200]}')
                    break
    print(json.dumps({'dense_gate': 'fail' if failures else 'pass',
                      'reference_head': reference['head'], 'head': manifest['head'],
                      'failures': failures}))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
