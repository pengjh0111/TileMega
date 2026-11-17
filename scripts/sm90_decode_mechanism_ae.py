#!/usr/bin/env python3
"""Freeze, compile and audit the Qwen3 B1 mechanism-removal experiment."""
import argparse
import copy
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import subprocess

from sm90d1_common import ROOT, read, sha, write


BASE = ROOT / 'configs/serving/sm90_ae/qwen3_1_7b_b1'
CONFIG = ROOT / 'configs/serving/sm90_mechanism_ae/qwen3_1_7b_b1'
MATRIX = {'A': (1, False, 'stage'), 'B': (16, False, 'stage'),
          'C': (1, True, 'la'), 'D': (16, True, 'la'), 'E': (1, False, 'stage')}


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    baseline = read(BASE / 'D.json')
    for arm, (chunks, group, emb) in MATRIX.items():
        config = copy.deepcopy(baseline)
        config['attention']['chunks'] = chunks
        config['group_wait'] = group
        config['emb'] = emb
        write(CONFIG / f'{arm}.json', config)
    (run / 'decode.export.json').write_bytes(lzma.decompress((ROOT /
        'configs/serving/sm90d1/inputs/qwen3_1_7b/decode.export.json.xz').read_bytes()))
    assert read(CONFIG / 'D.json') == baseline
    assert (CONFIG / 'E.json').read_bytes() == (CONFIG / 'A.json').read_bytes()


def build(run, compiler):
    out = run / 'compiled';out.mkdir(parents=True, exist_ok=True)
    for arm in 'ABCE':
        binary = out / f'{arm}.so'
        if binary.exists():
            raise ValueError(f'refusing to overwrite retained artifact {binary}')
        command = [str(compiler), 'compile', str(run / 'decode.export.json'), str(binary),
            '--tmexec-config', str(CONFIG / f'{arm}.json'), '--serving', 'decode',
            '--emit', 'kernel-sequence' if arm == 'E' else 'serving', '--batch', '1',
            '--runtime-target', str(ROOT / 'configs/targets/sm_90_h800.json'),
            '--nvcc-split-compile', '4']
        write(out / f'{arm}.command.json', command)
        with (out / f'{arm}.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'{arm} compiled', flush=True)


def audit(run, prefill):
    baseline = read(BASE / 'D.json')
    original = read(ROOT / 'docs/results/SM90_decode_AE_qwen3_1_7b_b1_provenance.json')
    headers = original['consistency']['task_body_headers_exact_D']
    for path, digest in headers.items():
        assert sha(ROOT / path) == digest, path
    implementation = {}
    for path in ('lib/Solver/Sm90DensePlan.cpp', 'lib/Solver/Sm90DenseSearch.cpp',
            'include/tilemega/Codegen/tasks/Sm90DenseSequence.cuh',
            'include/tilemega/Codegen/tasks/Sm90DenseRuntime.cuh', 'lib/Codegen/Codegen.cpp'):
        inherited = subprocess.check_output(['git', 'show',
            '4a84be39988967cb0a3c8f30c532bd25a6587960:' + path], cwd=ROOT)
        assert hashlib.sha256(inherited).hexdigest() == sha(ROOT / path), path
        implementation[path] = sha(ROOT / path)
    records, sources, arms = {}, {}, {}
    for arm, (chunks, group, emb) in MATRIX.items():
        config = read(CONFIG / f'{arm}.json')
        expected = copy.deepcopy(baseline)
        expected['attention']['chunks'] = chunks
        expected['group_wait'] = group
        expected['emb'] = emb
        assert config == expected, arm
        if arm == 'D':continue
        binary = run / 'compiled' / f'{arm}.so'
        result = read(Path(str(binary) + '.compile-result.json'))
        identity = read(Path(str(binary) + '.identity.json'))
        source = Path(str(binary) + '.cu').read_text()
        assert result['status'] == 'compiled'
        assert sha(CONFIG / f'{arm}.json') == result['coordinates_sha256']
        assert identity['source']['source_files'].items() >= headers.items()
        attention = re.findall(r'using Body_\d+=Attention<([^;]+);', source)
        assert attention and all(int(x.split(',')[5]) == chunks for x in attention)
        qkv = re.findall(r'using Stage_(?:1|6)=Placement<([^;]+);', source)
        # A QKV tile covers 32 columns of a 512-column KV group: 16 tiles/group.
        assert len(qkv) == 2
        assert all(x.endswith(',16>') == group for x in qkv)
        head = re.findall(r'using Body_\d+=Head<([^;]+);', source)
        assert len(head) == 1 and head[0].endswith(('true' if emb == 'la' else 'false') + '>')
        program = re.search(r'using NativeProgram=Program<([^;]+);', source).group(1)
        assert program.split(',')[9] == ('true' if emb == 'la' else 'false')
        sequence = '#define TILEMEGA_NATIVE_KERNEL_SEQUENCE 1' in source
        assert sequence == (arm == 'E')
        if sequence:
            assert 'kPostHead=false' in source
            assert 'TILEMEGA_NATIVE_SEQUENCE_PRESERVE_BODY 1' in source
        sources[arm] = source
        records[arm] = dict(chunks=chunks, attention_tasks_per_layer=8*chunks,
            group_tiles=16 if group else 0, embedding=emb, folded_argmax=True,
            execution='CUDA Graph typed stages' if sequence else 'persistent megakernel',
            source_sha256=sha(Path(str(binary) + '.cu')), binary_sha256=sha(binary),
            identity_sha256=sha(Path(str(binary) + '.identity.json')),
            compile_result_sha256=sha(Path(str(binary) + '.compile-result.json')))
        arms[arm] = dict(decode=str(binary), decode_sha256=sha(binary),
            prefill=str(prefill), prefill_sha256=sha(prefill))
    # Only the sequence launch adapter may differ; body aliases and Program match exactly.
    marker = '#define TILEMEGA_NATIVE_KERNEL_SEQUENCE 1'
    assert sources['E'].split(marker)[0] == sources['A'].split(
        '#include <tilemega/Codegen/tasks/Sm90DenseRuntime.cuh>')[0]
    assert (CONFIG / 'A.json').read_bytes() == (CONFIG / 'E.json').read_bytes()
    receipt = dict(evidence='verified configuration and emitted-source checks; not numerical validation',
        base_head='4a84be39988967cb0a3c8f30c532bd25a6587960',
        task_body_headers_exact_D=headers, arms=records, A_E_coordinates_byte_identical=True,
        implementation_unchanged_from_base=implementation,
        A_E_generated_program_body_prefix_byte_identical=True,
        D_unchanged=True, D_remeasured=False,
        runtime_paths=dict(embedding_stage='Prologue on every step when kEmbed=false',
            folded_argmax='Head finish in all arms; kPostHead=false in sequence',
            attention_S1='direct output; no partial-result merge',
            attention_S16='partial results and last-arriver merge',
            sequence='Capture typed stages, replay graph per decode step; no host stage launches',
            ownership='unchanged cyclic/grid-stride',
            planner='QKV group release and fused Head-to-embedding handoff only'))
    write(run / 'consistency.json', receipt)
    spec = dict(schema=1, model='qwen3_1_7b', batch=1, dtype='BF16', input_tokens=64,
        output_tokens=1024, body_source_head=original['body_source_head'],
        body_version_resolution='Exact historical D task bodies, inherited from 4a84be399; audited per arm',
        guard_policy=str(ROOT / 'configs/serving/sm90_ae/guard_policy.json'), arms=arms,
        consistency_sha256=sha(run / 'consistency.json'))
    write(run / 'spec.json', spec)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'audit'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--prefill', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == 'prepare':prepare(run)
    if args.action == 'build':build(run, args.compiler.resolve())
    if args.action == 'audit':audit(run, args.prefill.resolve())
