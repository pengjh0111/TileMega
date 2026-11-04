#!/usr/bin/env python3
"""Describe the operations a generated native forward artifact actually issues.

Template arguments and generated bindings/geometries are authoritative. Traffic
is an inferred compulsory-read/write byte model, not measured DRAM traffic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_fields(text):
    result=[];begin=0;depth={'{':0,'(':0,'[':0,'<':0};close={'}':'{',')':'(',']':'[','>':'<'}
    quoted=False;escaped=False
    for i,char in enumerate(text):
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
            continue
        if char=='"':quoted=True
        elif char in depth:depth[char]+=1
        elif char in close:
            depth[close[char]]-=1
            if depth[close[char]]<0:raise ValueError('unbalanced generated initializer/template')
        elif char==',' and not any(depth.values()):
            if text[begin:i].strip():result.append(text[begin:i].strip())
            begin=i+1
    if quoted or any(depth.values()):raise ValueError('unterminated generated initializer/template')
    if text[begin:].strip():result.append(text[begin:].strip())
    return result


def aggregate(text):
    if not text.startswith('{') or not text.endswith('}'):
        raise ValueError('expected generated aggregate')
    return split_fields(text[1:-1])


def integer(text):
    text=text.strip()
    if text=='kAbsent':return None
    if not re.fullmatch(r'-?\d+(?:u|ull)?',text):raise ValueError('nonliteral generated integer: '+text)
    return int(re.sub(r'(?:ull|u)$','',text))


def constants(source,kind,index):
    match=re.search(rf'struct {kind}_{index} \{{(.*?)\n\}};',source,re.S)
    if not match:raise ValueError(f'missing {kind}_{index}')
    return {key:integer(value) for key,value in re.findall(r'\b(k\w+)=(kAbsent|-?\d+(?:ull|u)?)(?=[,;])',match[1])}


def generated_tables(source):
    bm=re.search(r'kNativeForwardBuffers=\{\{(.*?)\}\};',source,re.S)
    mm=re.search(r'kNativeForwardMaps=\{\{(.*?)\}\};',source,re.S)
    if not bm or not mm:raise ValueError('missing generated native tables')
    buffers=[]
    for row in split_fields(bm[1]):
        f=aggregate(row)
        if len(f)!=6:raise ValueError('unknown native buffer initializer')
        widths={'TM_DTYPE_BF16':2,'TM_DTYPE_F32':4,'TM_DTYPE_I32':4,'TM_DTYPE_I64':8}
        if f[2] not in widths:raise ValueError('unknown buffer dtype')
        buffers.append(dict(name=json.loads(f[0]),width=widths[f[2]],constant=integer(f[3]),per_batch=integer(f[4])))
    maps=[]
    for row in split_fields(mm[1]):
        f=aggregate(row)
        if len(f)!=10:raise ValueError('unknown native map initializer')
        maps.append(dict(buffer=integer(f[0]),kind=integer(f[1]),rank=integer(f[2]),
                         dims=[integer(x) for x in aggregate(f[5])],
                         strides=[integer(x) for x in aggregate(f[6])],
                         box=[integer(x) for x in aggregate(f[7])],
                         conv=[integer(x) for x in aggregate(f[8])],
                         layout=aggregate(f[9])))
    return buffers,maps


def describe(source_path,manifest_path,identity_path=None,logical_plan_path=None,inputs_path=None):
    source_path,manifest_path=Path(source_path),Path(manifest_path)
    source=source_path.read_text();manifest=json.loads(manifest_path.read_text())
    if identity_path:
        identity=json.loads(Path(identity_path).read_text())
        if identity['generated_source_sha256']!=sha(source_path) or identity['manifest_sha256']!=sha(manifest_path):
            raise ValueError('work input differs from artifact identity')
    from tilemega.build.feature_contract import build_defines
    from tilemega.build.forward_contract import dispatch
    defines=build_defines(source,[]);dispatch(manifest,defines,source)
    from dnn1_logical_work import logical_work
    logical=logical_work(logical_plan_path,manifest,inputs_path)
    buffers,maps=generated_tables(source);batch=int(manifest['runtime_batch']);grid=int(manifest['grid'])
    def allocated(index):
        if index is None:return 0
        b=buffers[index];return b['width']*(b['constant']+batch*b['per_batch'])
    work=[]
    def covered_axis(size,out,kernel,stride,pad,dilation):
        return len({o*stride-pad+r*dilation for o in range(out) for r in range(kernel)
                    if 0<=o*stride-pad+r*dilation<size})
    for index,stage in enumerate(manifest['stages']):
        bind=constants(source,'Bind',index);shape=constants(source,'Shape',index)
        match=re.search(rf'using Body_{index}=(\w+)(?:<([^;]*)>)?;',source)
        if not match:raise ValueError('missing typed body')
        name,raw=match.group(1),match.group(2) or '';args=split_fields(raw)
        kind=stage['body'];m=tm=n=k=pixels=channels=0;k_iter=None;slots=0;tasks=0;weight_task=0
        weight=activation=output=flops=0;coverage=batch;notes=[];issued_flops=0;mma_issued_flops=0
        vectors={bind.get('kVector'+str(i)) for i in range(4)}-{None}
        vectors|={bind.get('kDwVector'+str(i)) for i in range(4)}-{None}
        vector_bytes=sum(allocated(i) for i in vectors)
        residual=bind.get('kResidual');stats=bind.get('kStats');row_stats=bind.get('kRowStats')
        if kind in ('gemm','conv'):
            tm,tn,tk,slots,split=map(integer,args[:5]);m,n,k,store=map(integer,args[6:10])
            tasks=((m+tm-1)//tm)*(n//tn)*split;k_iter=k//(split*tk)
            weight=2*n*k;weight_task=2*tn*(k//split);issued_flops=2*m*n*k
            amap=maps[bind['kMap']]
            if amap['kind']==1:
                c=amap['conv'];physical=[integer(x) for x in aggregate(amap['layout'][3])]
                unique_h=covered_axis(c[1],c[13],c[5],c[7],c[9],c[11])
                unique_w=covered_axis(c[2],c[14],c[6],c[8],c[10],c[12])
                activation=2*c[0]*unique_h*unique_w*physical[3]
                logical_k=c[3]*c[5]*c[6]
            else:
                activation=2*m*min(k,amap['dims'][0]);logical_k=min(k,amap['dims'][0])
            output=2*m*store;flops=issued_flops
            mma_issued_flops=2*((m+tm-1)//tm)*tm*n*k
            if 2*m*store>allocated(bind['kOutput']):
                notes.append('typed output writes exceed generated destination allocation; do not treat this as a passed correctness artifact')
            if residual is not None:activation+=2*m*store
            if stats is not None:activation+=8*m
            if row_stats is not None:output+=8*m
            if 'kPixelShuffle' in raw or 'Sm90ForwardEpilogueChain<'+str(tm)+','+str(tn)+',2,' in raw:
                notes.append('PixelShuffle: typed M can differ from pre-shuffle logical M; no solver M substituted')
            logical_n=2*store if 'Sm90ForwardOp::kGatePair' in raw else store
            logical_flops=2*m*logical_n*logical_k
        elif kind in ('depthwise','dwpw','inverted_residual'):
            tw,th,cb=map(integer,args[:3]);slots=integer(args[4] if kind=='depthwise' else args[5]);tm=tw*th
            channels=shape['kChannels'];out_channels=shape['kOutChannels'];p=shape['kOutputHeight'];q=shape['kOutputWidth']
            padded=shape['kPaddedChannels'];pack=shape['kDepthwisePack']
            batch_arg=7 if kind=='depthwise' else 9
            coverage=integer(args[batch_arg]) if len(args)>batch_arg else 1
            m=coverage*p*q;pixels=m
            tasks=coverage*((q+tw-1)//tw)*((p+th-1)//th)
            weight=2*channels*9*pack;activation=coverage*2*shape['kWidth']*shape['kHeight']*padded
            if kind=='depthwise':
                tasks*=(channels+cb-1)//cb;stored=channels//2 if 'Sm90ForwardOp::kGatePair' in raw else channels
                output=2*m*stored;flops=2*m*channels*9;weight_task=2*cb*9*pack
            else:
                tn=integer(args[3]);npad=((out_channels+tn-1)//tn)*tn;tasks*=(out_channels+tn-1)//tn
                weight+=2*npad*padded;output=2*m*out_channels;flops=2*m*channels*9+2*m*npad*padded
                weight_task=2*tn*padded+2*channels*9*pack
            if residual is not None:activation+=output
            issued_flops=logical_flops=flops
            notes.append('explicit template Batch covers image tasks and TMA offsets' if len(args)>batch_arg else 'original spatial body has no batch task/offset; TMA batch coordinate=0')
        elif kind=='encoder_attention':
            if len(args)==5:seq,d,heads,rows,slots=map(integer,args);coverage=1
            elif len(args)>=6:coverage,seq,d,heads,rows,slots=map(integer,args[:6])
            else:raise ValueError('unknown attention template')
            m=seq;n=seq;k=d;pixels=coverage*seq;channels=heads*d
            tm=rows;tasks=coverage*heads*(seq//rows);k_iter=d//16
            activation=coverage*2*seq*heads*d*3;output=coverage*2*seq*heads*d
            flops=issued_flops=logical_flops=coverage*4*heads*seq*seq*d
            if coverage!=batch:notes.append('known F11: template/maps/tasks/output cover one sequence, not runtime batch')
        elif kind=='layout':
            image_pixels=shape['kWidth']*shape['kHeight'];channels=shape['kOutChannels']
            coverage=integer(args[2]) if len(args)>=3 else 1;pixels=coverage*image_pixels
            block=integer(args[1]) if len(args)>=3 else 1
            tasks=coverage*((image_pixels+block-1)//block)
            activation=2*pixels*channels;output=2*pixels*channels;logical_flops=0
            notes.append('explicit template Batch and pixel Block; padded channels are not copied' if len(args)>=3 else 'original layout width/height and offsets cover one image; padded channels are not copied')
        elif kind=='pool':
            window,stride=map(integer,args[:2]);width=shape['kWidth'];height=shape['kHeight'];channels=shape['kChannels'];coverage=integer(args[2]) if len(args)>=3 else 1
            pw=(width+1)//stride;ph=(height+1)//stride;tasks=coverage*pw*ph;pixels=coverage*pw*ph
            if len(args)>=4:
                elements=integer(args[3])
                if elements!=128:raise ValueError("unknown max-pool consumer assignment")
                tasks=(pixels*channels+elements-1)//elements
                notes.append("flattened pixel/channel tasks, explicit128 elements per task")
            activation=coverage*2*width*height*channels;output=coverage*2*pw*ph*channels
            validw=sum(sum(0<=x*stride-1+s<width for s in range(window)) for x in range(pw))
            validh=sum(sum(0<=y*stride-1+r<height for r in range(window)) for y in range(ph))
            flops=coverage*validw*validh*channels;issued_flops=logical_flops=flops
            notes.append('explicit template Batch covers pool image offsets' if len(args)>=3 else 'original pool loops/addresses cover one image')
        elif kind=='global_pool':
            channels=shape['kChannels'];parts=shape['kWidth'];pixels=shape['kHeight']
            coverage=integer(args[0]) if args else 1
            tasks=coverage*((channels+127)//128)
            activation=coverage*4*parts*channels;output=coverage*2*channels;flops=coverage*(parts+1)*channels
            issued_flops=logical_flops=flops
            notes.append('explicit template Batch covers GAP images; input partial extent from generated Shape width' if args else 'original GAP writes output[c] only; input partial extent from generated Shape width')
        elif kind=='gemv':
            block=integer(args[0]);m=shape['kWidth'] if shape['kWidth']>0 else 1
            k=shape['kChannels'];n=shape['kOutChannels'];channels=k;pixels=m
            tasks=m*((n+block-1)//block)
            weight=2*((n+block-1)//block)*block*k;activation=2*m*k;output=2*m*n
            weight_task=2*block*k;flops=2*m*n*k;issued_flops=logical_flops=flops
        elif kind=='embedding':
            channels=integer(args[0]);rows=shape['kWidth'];tasks=(rows+3)//4;pixels=rows
            activation=2*rows*channels*3;output=2*rows*channels+(8*rows if row_stats is not None else 0)
            flops=5*rows*channels;issued_flops=logical_flops=flops
            notes.append('embedding bytes count addressed table elements; repeated table/cache reuse is not measured DRAM')
        elif kind=='layer_norm':
            channels=integer(args[0]);rows=shape['kWidth']*shape['kHeight'];tasks=(rows+3)//4;pixels=rows
            activation=2*rows*channels;output=2*rows*channels;weight=4*channels
            flops=8*rows*channels+320*rows;issued_flops=logical_flops=flops
            notes.append('scalar arithmetic count is an inferred operation count; no tensor-core throughput claim')
        else:raise ValueError('unsupported actual body: '+kind)
        # Affine vectors are mandatory weights. Split-K and row-partial traffic
        # are implementation overhead and are reported separately from this bound.
        weight+=vector_bytes
        if mma_issued_flops:issued_flops=mma_issued_flops
        if not tasks:raise ValueError('zero actual task count')
        work.append(dict(index=index,body=kind,representative=stage.get('representative',''),
            task_body=name,template_arguments=args,tasks=tasks,grid=grid,slots=slots,k_iter=k_iter,
            m=m,n=n,k=k,pixels=pixels,channels=channels,tm=tm,
            weight_bytes_per_task=weight_task,weight_bytes=weight,activation_bytes=activation,
            output_bytes=output,dram_bytes=weight+activation+output,flops=flops,issued_flops=issued_flops,
            mma_flops_including_m_tail=mma_issued_flops,
            source_visible_unpadded_flops_diagnostic=logical_flops,
            source_visible_unpadded_status="storage-visible dimensions may still include channel padding; not a model logical bound",covered_batches=coverage,runtime_batch=batch,
            geometry_evidence='verified generated template/bind/Shape literals; static task/coverage formulas inferred',
            traffic_evidence='inferred compulsory addressed bytes; not measured DRAM traffic/cache misses',
            geometry_source=str(source_path.resolve())+f':Body_{index}/Bind_{index}/Shape_{index}',
            notes=notes,**logical[index]))
    return dict(schema=1,source=str(source_path.resolve()),source_sha256=sha(source_path),
        logical_plan=str(Path(logical_plan_path).resolve()) if logical_plan_path else None,
        logical_plan_sha256=sha(logical_plan_path) if logical_plan_path else None,
        logical_inputs_sha256=sha(inputs_path) if inputs_path else None,
        manifest=str(manifest_path.resolve()),manifest_sha256=sha(manifest_path),runtime_batch=batch,grid=grid,
        body_header_status='body semantics decoded from generated template arity; runtime correctness requires separate proof',
        lower_bound_method='weight + required activation + output addressed bytes; compiled M/N/K mathematical FLOPs, M-tail issuance separately recorded; scalar counts inferred; split-K/row partials excluded as overhead',
        trace=defines.get('TILEMEGA_TRACE_STAGE','0') not in ('0','false'),stages=work)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--identity',type=Path);p.add_argument('--logical-plan',type=Path)
    p.add_argument('--inputs',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=describe(a.source,a.manifest,a.identity,a.logical_plan,a.inputs)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise FileExistsError('retain prior work ledger')
    a.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'stages':len(value['stages']),'out':str(a.out),'trace':value['trace']}))


if __name__=='__main__':main()
