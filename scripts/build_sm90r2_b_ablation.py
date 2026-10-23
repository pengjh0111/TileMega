#!/usr/bin/env python3
"""Reconstruct B-only controls, explicitly excluded from final native selection."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from sm90d1_common import ROOT, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--reference-config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--headers', type=Path, required=True)
    a = p.parse_args()
    original, output = a.input.resolve(), a.out.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('retain ablation controls in a new runs/sm90r2 path')
    plan = read(str(original) + '.plan.json')
    c = plan['native_dense']
    if any(g['preissue_smem_bytes'] or g['prefetch_l2_bytes'] for g in [*c['gemms'].values(), c['attention']]):
        raise ValueError('B-only reconstruction starts before cross-stage preissue')
    ref = read(a.reference_config)['coordinates']['decode']
    extent, impl = ref['attention_kv_block'], ref['attention_impl']
    if impl not in ('simt', 'mma16'):
        raise ValueError('historical control expects the frozen R1 attention family')
    source = Path(str(original) + '.cu').read_text()
    body_pattern = re.compile(r'using Body_(\d+)=Attention<([^;]+)>;')
    matches = list(body_pattern.finditer(source))
    if not matches:
        raise ValueError('native source has no attention stages')
    buffers_start = source.index(' kNativeBuffers={{')
    buffers_end = source.index('\n}};', buffers_start)
    prefix, buffer_text, suffix = source[:buffers_start], source[buffers_start:buffers_end], source[buffers_end:]
    lines = buffer_text.splitlines()
    aliases, metadata = {}, {}
    for m in matches:
        index = int(m[1]);v = [0xffffffff if x=='kAbsent' else int(x) for x in m[2].split(',')]
        if len(v) != 20:
            raise ValueError('native attention signature changed; review this reconstruction')
        batch,d,q,norm,_,_,_,_,capacity,*ids = v
        qkv,key,value,out,cos,sin,qn,kn,partial,lse,arrivals = ids
        blocks = (capacity+extent-1)//extent
        for bid, size in [(partial,8*blocks*q*d),(lse,8*blocks*q)]:
            line = lines[bid+1]
            line, count = re.subn(r',(\d+)ull,(\d+)ull,',f',0ull,{size}ull,',line,count=1)
            if count != 1:
                raise ValueError('ambiguous partial buffer recipe')
            lines[bid+1] = line
        args = [batch,d,q,norm,extent,capacity,qkv,key,value,out,cos,sin,qn,kn,partial,lse]
        attention = f'using Body_{index}=HistoricalAttention<'+','.join(map(str,args))+'>;'
        merge_args = [batch,d,q,extent,capacity,partial,lse,out]
        attention += f'\nusing MergeBody_{index}=HistoricalMerge<'+','.join(map(str,merge_args))+'>;'
        attention += f'\nusing MergeStage_{index}=Placement<MergeBody_{index}>;'
        aliases[index] = attention
        metadata[index] = (batch*8*blocks,d,q,blocks)
    source = prefix+'\n'.join(lines)+suffix
    source = body_pattern.sub(lambda m: aliases[int(m[1])],source)
    for index in aliases:
        # Historical attention must overwrite the same shared arena, invalidating
        # activation-lifetime reuse just as the original attention did.
        source = re.sub(rf'using Stage_{index}=Placement<Body_{index},[^;]+>;',
                        f'using Stage_{index}=Placement<Body_{index}>;',source)
    program = re.search(r'using NativeProgram=Program<[^;]+>;',source)
    text = program[0]
    for index in aliases:
        text = re.sub(rf'\bStage_{index}(?=[,>])',f'Stage_{index},MergeStage_{index}',text)
    source = source[:program.start()]+text+source[program.end():]
    old_count = re.search(r'constexpr std::array<StageInfo,(\d+)> kNativeStages=\{\{',source)
    start, end = old_count.end(), source.index('\n}};',old_count.end())
    stage_lines = source[start:end].splitlines();new_lines=[];stage_index=0
    for line in stage_lines:
        if not line.strip():continue
        if stage_index in aliases:
            tasks,d,q,blocks=metadata[stage_index]
            line=f'  {{"kFusedAttention","",0ull,{tasks},8,{q},{d},{blocks}}},'
        new_lines.append(line)
        if stage_index in aliases:
            new_lines.append(f'  {{"kAttentionMerge","",0ull,{plan["runtime_batch"]*8},8,{q},{d},{blocks}}},')
        stage_index+=1
    source=source[:start]+'\n'+'\n'.join(new_lines)+source[end:]
    source=source.replace(old_count[0],old_count[0].replace(old_count[1],str(int(old_count[1])+len(aliases)),1),1)
    header=a.headers.resolve()/'tilemega/Diagnostics/Sm90R2HistoricalAttention.cuh'
    if not header.exists():raise ValueError('a frozen historical control header is required')
    source=source.replace('#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>',
        '#define TILEMEGA_SM90R2_HISTORICAL_ABLATION 1\n'+
        f'#define TILEMEGA_ATTENTION_IMPL {3 if impl=="simt" else 0}\n'+
        '#define TILEMEGA_ATTENTION_NOINLINE 1\n'+
        '#include <tilemega/Diagnostics/Sm90R2HistoricalAttention.cuh>')
    output.parent.mkdir(parents=True,exist_ok=True)
    Path(str(output)+'.cu').write_text(source)
    plan['historical_ablation']=dict(mechanism='A+B; old capacity attention and separate merge',
        production_eligible=False,reference_config_sha256=sha(a.reference_config),
        attention_kv_block=extent,attention_impl=3 if impl=='simt' else 0,
        input_sha256=sha(original),source_commit=read(str(original)+'.identity.json')['source']['head'],
        reconstructed=True,original_B_only_commit_exists=False)
    plan['native_origin']='historical_ablation'
    write(str(output)+'.plan.json',plan)
    command=shlex.split(Path(str(original)+'.build_command.txt').read_text())
    command=[str(output)if x==str(original)else str(output)+'.cu'if x==str(original)+'.cu'
             else '-I'+str(a.headers.resolve())if x.startswith('-I')and x.endswith('/include')and'cutlass'not in x else x for x in command]
    Path(str(output)+'.build_command.txt').write_text(shlex.join(command)+'\n')
    sys.path.insert(0,str(ROOT/'python'))
    from tilemega.build.identity import snapshot, generate
    snapshot(ROOT,str(output)+'.source.json')
    with Path(str(output)+'.ptxas.log').open('w')as stream:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
    generate(output,ROOT,str(output)+'.source.json')
    write(str(output)+'.ablation.json',dict(binary_sha256=sha(output),
        generated_source_sha256=sha(str(output)+'.cu'),historical_header_sha256=sha(header),
        control=plan['historical_ablation'],final_eligible=False))


if __name__=='__main__':main()
