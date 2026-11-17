#!/usr/bin/env python3
"""Freeze the actual historical native structure, not a planner claim."""
import collections
import json
import lzma
from pathlib import Path
import re
from sm90_local_global_local import ROOT, CONFIG, sha, write

def main():
    source=Path('/root/TileMega-m/runs/sm90r3/solve/FINAL/qwen3_1_7b-B1/decode.so.cu')
    binary=source.with_suffix('')
    assert sha(source)=='031d3710511f2b72de38ff50bbbae502346d131cf4ff16ba7e0d32c424968dea'
    assert sha(binary)=='0f8989856fc251ab335ffac54e110efa788800c31bd8429a112bc3a2baba299c'
    manifest=ROOT/'configs/serving/sm90_ae/qwen3_1_7b_b1/D.manifest.json'
    original=json.loads(manifest.read_text());coords=original['coordinates']
    text=source.read_text();literal=re.search(r'kNativeCoordinateIdentity\[\]="(.*)";',text)
    if literal:
        embedded=json.loads(json.loads('"'+literal.group(1)+'"'))
        assert embedded==coords
    aliases=[line for line in text.splitlines() if line.startswith('using ')]
    assert any('kResidual' in line for line in aliases)
    assert any('kSwiGLU' in line for line in aliases)
    assert any('Head<HeadGemm_11,0,3,4,2048,1088,true>' in line for line in aliases)
    export=ROOT/'configs/serving/sm90d1/inputs/qwen3_1_7b/decode.export.json.xz'
    graph=json.loads(lzma.open(export).read())
    semantic=collections.Counter(n['target'] for n in graph['nodes'] if n['op']=='call_function')
    out=dict(schema=1,structure='S_TM',coordinates=coords,
        artifact=dict(source=str(source),source_sha256=sha(source),binary=str(binary),binary_sha256=sha(binary),
                      source_commit='4b9d1362e648e2ee2cc7f96da9422f97a6d36c45',
                      manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest)),
        generated_aliases=aliases,export_sha256=sha(export),frontend_operator_counts=dict(semantic),
        fixed_body_contracts=dict(qkv='InputNorm with gamma folded into weights; BF16 packed QKV',
            attention='QK norm/RoPE/cache append inside attention; S16 partial/LSE + LA merge',
            o='BF16-rounded projection + residual in GEMM epilogue',
            gate_up='InputNorm; interleaved gate/up linear + BF16 SwiGLU epilogue',
            down='BF16-rounded projection + residual in GEMM epilogue',
            lm_head='InputNorm; per-tile/per-track argmax; folded final reduction + embedding handoff'),
        historical_plan=dict(ownership='cyclic/grid-stride',order='fixed five typed stages per layer',
            release='QKV group release; other aggregate completion',weight_stream=False,
            selection='historical calibrated additive operator model and policy template; not proof of whole-program owner/order search'),
        D_ref=dict(tpot_ms=original['measured']['tpot_ms'],new_or_reused='historical reference only',
                   cross_session=True,matrix_eligible=False,
                   reason='new actual planner/task-coordinate executor requires D_test'))
    assert out['D_ref']['tpot_ms']==1.6754311115689438
    path=CONFIG/'S_TM.json'
    if path.exists():raise ValueError('preserve frozen structure')
    write(path,out);print(path)

if __name__=='__main__':main()
