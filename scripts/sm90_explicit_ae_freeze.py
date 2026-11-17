#!/usr/bin/env python3
"""Audit the fixed explicit realization and freeze its three timing artifacts."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from sm90d1_common import ROOT, read, sha, write


def b2_first(run):
    binary=run/'build-v1/B2.so';source_path=Path(str(binary)+'.cu');s=source_path.read_text()
    previous=Path('/root/TileMega-mechanism-ae/runs/sm90_mechanism_ae/compiled/B.so.cu')
    def buffers(text):return text.split('kNativeBuffers={{\n')[1].split('}};')[0].splitlines()
    assert buffers(s)[:-2]==buffers(previous.read_text())
    assert read(Path(str(binary)+'.compile-result.json'))['status']=='compiled'
    assert 'std::array<StageInfo,340>' in s
    assert '=Head<' not in s and 'backend::ServingEpilogueOp::kResidual' not in s
    assert 'backend::ServingEpilogueOp::kSwiGLU' not in s
    write(run/'B2-first-audit.json',dict(status='B2 static audit passed before generating A2/E2',
        gemm_geometry_and_pack_recipes='original B recipes byte-identical',native_stages=340,
        per_layer_stages=12,fused_head_absent=True,residual_swiglu_gemm_epilogues_absent=True,
        source_sha256=sha(source_path)))


def freeze(run):
    config = ROOT / 'configs/serving/sm90_explicit_ae/qwen3_1_7b_b1'
    old = ROOT / 'configs/serving/sm90_mechanism_ae/qwen3_1_7b_b1'
    prefill = Path('/root/TileMega/runs/sm90r2/build/sm90r3-final-prefill/qwen3_1_7b-B1/prefill.so')
    sources, arms, records = {}, {}, {}
    for arm, prior in [('A2','A'),('B2','B'),('E2','A'),('C','C'),('D','D')]:
        assert read(config / f'{arm}.json') == read(old / f'{prior}.json')
    assert (config / 'A2.json').read_bytes() == (config / 'E2.json').read_bytes()
    for arm in ('A2','B2','E2'):
        binary = run / 'build-v1' / f'{arm}.so'
        identity = read(Path(str(binary)+'.identity.json'))
        compiled = read(Path(str(binary)+'.compile-result.json'))
        source_path = Path(str(binary)+'.cu');source = source_path.read_text();sources[arm] = source
        assert compiled['status'] == 'compiled'
        assert identity['binary_sha256'] == sha(binary)
        assert compiled['coordinates_sha256'] == sha(config / f'{arm}.json')
        assert identity['plan']['realization'] == 'explicit_stages_qwen_b1_v1'
        assert identity['shared_memory_bytes'] == 217664
        assert identity['plan']['grid'] == 132
        expected = ['RmsQkv','Qkv','PrepareQkv','AttentionCompute']
        chunks = 16 if arm=='B2' else 1
        if chunks>1:expected += ['MergeAttention']
        expected += ['OutputGemm','OutputResidual','RmsGate','GateUp','SwiGLU','Down','DownResidual']
        assert f"using ExplicitLayer=std::tuple<{','.join(expected)}>;" in source
        assert f'std::array<StageInfo,{4+28*len(expected)}>' in source
        assert ('using MergeAttention=' in source) == (chunks>1)
        assert '=Head<' not in source and '=Program<' not in source
        for op in ('kResidual','kSwiGLU'):
            assert 'backend::ServingEpilogueOp::'+op not in source
        assert 'using ExplicitEnd=std::tuple<RmsHead,HeadPartial,ArgmaxFinal>;' in source
        assert 'ExplicitHeadPartial<HeadPartialBody>' in source
        symbols = subprocess.check_output(['nm','-C',str(binary)],text=True)
        (binary.parent/f'{arm}.symbols.txt').write_text(symbols)
        assert ('sm90_dense_native_kernel' in symbols) == (arm!='E2')
        assert ('cudaGraphLaunch' in symbols) == (arm=='E2')
        assert 'Sm90DenseHeadFinish' not in symbols
        # Fixed range formula: history [begin,min(end,past)) plus one new-row owner.
        for past in range(64,1087):
            ranges = [(min((past+1)*c//chunks,past),
                past+1 if c==chunks-1 else min((past+1)*(c+1)//chunks,past)) for c in range(chunks)]
            assert ranges[0][0]==0 and ranges[-1][1]==past+1
            assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
            assert sum(lo<=past<hi for lo,hi in ranges)==1
        records[arm] = dict(chunks=chunks, stages_per_layer=expected,
            end_stages=['RmsHead','HeadPartial','ArgmaxFinal'], native_stage_count=4+28*len(expected),
            graph=arm=='E2', binary_sha256=sha(binary), source_sha256=sha(source_path),
            identity_sha256=sha(Path(str(binary)+'.identity.json')),
            linked_symbols_sha256=sha(binary.parent/f'{arm}.symbols.txt'),
            full_KV_range_static_check=True,
            original_semantic_CTG_stages=143,
            realization='dedicated native emitter; explicit stage list overrides original five-stage lowering')
        arms[arm] = dict(decode=str(binary), decode_sha256=sha(binary),
            prefill=str(prefill), prefill_sha256=sha(prefill))
    marker='#define TILEMEGA_NATIVE_KERNEL_SEQUENCE 1'
    assert sources['E2'].split(marker)[0] == sources['A2'].split(
        '#include <tilemega/Codegen/tasks/Sm90DenseRuntime.cuh>')[0]
    assert read(run/'B2-first-audit.json')['source_sha256']==records['B2']['source_sha256']
    body_paths = ['include/tilemega/Backend/'+p for p in
        ('Sm90DenseGemm.h','Sm90DenseEpilogue.h','Sm90DenseAttentionBody.cuh','Sm90DenseHead.cuh',
         'Sm90DenseAttentionPartition.h','Sm90DenseAttentionMath.cuh')]
    body_paths += ['include/tilemega/Codegen/tasks/'+p for p in
        ('Sm90DenseProgram.cuh','Sm90DenseExplicit.cuh','Sm90DenseExplicitSequence.cuh',
         'Sm90DenseRuntime.cuh','Sm90DenseStage.cuh','Sm90DenseSequence.cuh')]
    body_hashes = {p:sha(ROOT/p) for p in body_paths}
    for arm in arms:
        identity=read(Path(arms[arm]['decode']+'.identity.json'))
        assert identity['source']['source_files'].items() >= body_hashes.items()
    receipt = dict(evidence='static configuration, generated code and binary audit only; no numerical validation',
        base_head='7496665a83f857022f934c6474475be5f6862586',
        arms=records, A2_E2_same_coordinates_and_generated_program=True,
        bodies=body_hashes, B2_first_audit=read(run/'B2-first-audit.json'),
        mechanisms=dict(normalization='explicit FP32 statistics; original output scaling; folded gamma unchanged',
            residual='BF16 GEMM result -> explicit BF16 residual addition',
            swiglu='BF16 interleaved gate/up -> original BF16 SiLU/product rounding',
            attention_prepare='original QK norm/RoPE/cache append once per KV group',
            attention_mainloop='original MMA/TMA/online-softmax; prepared Q and cache row loads only',
            merge='S16 original partial/LSE merge as independent stage; none for S1',
            head='per-CTA local argmax partial only, then independent final reduction',
            embedding='independent stage on every decode step',
            boundaries='aggregate GPU release/acquire at every semantic stage',
            graph='same typed stage list and 132-CTA task mapping as A2'))
    write(run/'consistency.json',receipt)
    write(run/'spec.json',dict(schema=1,model='qwen3_1_7b',batch=1,dtype='BF16',input_tokens=64,
        output_tokens=1024,arms=arms,body_source_head='7496665a83f857022f934c6474475be5f6862586',
        body_version_resolution='Explicit realization fork; unchanged P1 defaults; see static audit',
        guard_policy=str(ROOT/'configs/serving/sm90_ae/guard_policy.json'),
        consistency_sha256=sha(run/'consistency.json')))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--b2-first',action='store_true');a=p.parse_args()
    (b2_first if a.b2_first else freeze)(a.run.resolve())
