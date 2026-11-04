"""Logical work from an explicitly bound CPU DnnModelPlan dump; never solver M."""
from pathlib import Path
import hashlib
import json


def logical_work(plan_path, manifest, inputs_path=None):
    unknown=lambda reason:dict(logical_flops=None,logical_weight_bytes=None,
        logical_m=None,logical_n=None,logical_k=None,logical_status='unknown: '+reason,
        logical_weight_scope='unknown',logical_work_complete=False)
    count=len(manifest['stages'])
    if plan_path is None:return [unknown('no logical plan provided') for _ in range(count)]
    plan=json.loads(Path(plan_path).read_text())
    if plan['batch']!=manifest['runtime_batch'] or len(plan['stages'])!=count:
        return [unknown('logical plan batch/stage count mismatch') for _ in range(count)]
    bridge=Path(plan['bridge']);expected=manifest.get('export_sha256')
    if not expected or not bridge.exists() or hashlib.sha256(bridge.read_bytes()).hexdigest()!=expected:
        return [unknown('logical plan is not bound to the artifact bridge hash') for _ in range(count)]
    if any(a.get('representative')!=b.get('representative') for a,b in zip(plan['stages'],manifest['stages'])):
        return [unknown('logical plan representatives differ from artifact') for _ in range(count)]
    buffers={b['index']:b for b in plan['buffers']};batch=plan['batch'];inputs=None
    if inputs_path:
        from safetensors.torch import load_file
        inputs=load_file(str(inputs_path),device='cpu')
    rows=[]
    for t in plan['stages']:
        body=t['body'];g=t.get('gemm_descriptor');c=t.get('conv_descriptor')
        result=unknown('no exact logical operation/weight count for this body')
        source=str(Path(plan_path).resolve())+':stage '+str(t['index'])
        if body in ('gemm','conv','gemv'):
            m=c['batch']*c['output_height']*c['output_width'] if c else batch*g['rows_per_batch']
            n=g['n'];k=c['channels']*c['kernel_r']*c['kernel_s'] if c else g['k']
            result.update(logical_m=m,logical_n=n,logical_k=k,logical_flops=2*m*n*k,
                logical_weight_bytes=2*n*k,logical_status='verified plan extents; inferred core MAC work without tile/channel padding',
                logical_weight_scope='BF16 main matmul/conv weights only; affine/epilogue extras excluded',
                logical_work_complete=False)
        elif body in ('depthwise','dwpw','inverted_residual'):
            pixels=c['batch']*c['output_height']*c['output_width'];channels=c['channels'];taps=c['kernel_r']*c['kernel_s']
            flops=2*pixels*channels*taps;weight=2*channels*taps
            if g:flops+=2*pixels*g['n']*channels;weight+=2*g['n']*channels
            result.update(logical_m=pixels,logical_n=g['n'] if g else channels,logical_k=channels if g else taps,
                logical_flops=flops,logical_weight_bytes=weight,
                logical_status='verified plan convolution geometry; one semantic DW/PW per image without repeated N-tile DW',
                logical_weight_scope='BF16 main filters/projection only; affine/epilogue extras excluded',logical_work_complete=False)
        elif body=='encoder_attention':
            seq,heads,dim=t['stage_width'],t['stage_extent'],t['stage_group']
            result.update(logical_m=seq,logical_n=seq,logical_k=dim,logical_flops=4*batch*heads*seq*seq*dim,
                logical_weight_bytes=0,logical_status='verified full-batch plan attention geometry; softmax scalar FLOPs excluded',
                logical_weight_scope='attention has no separate trainable matrix weights',logical_work_complete=False)
        elif body=='layout':
            result.update(logical_flops=0,logical_weight_bytes=0,logical_status='implementation-only layout conversion; not a model operator',
                logical_weight_scope='no weights',logical_work_complete=True)
        elif body=='global_pool':
            result.update(logical_flops=batch*t['rows_per_batch']*t['stage_extent'],logical_weight_bytes=0,
                logical_status='verified plan pixels/channels; inferred sum+scale operation count for full batch',
                logical_weight_scope='no weights',logical_work_complete=True)
        elif body=='pool':
            count_h=sum(sum(0<=y*c['stride_h']-c['pad_h']+r<c['height'] for r in range(c['kernel_r'])) for y in range(c['output_height']))
            count_w=sum(sum(0<=x*c['stride_w']-c['pad_w']+s<c['width'] for s in range(c['kernel_s'])) for x in range(c['output_width']))
            result.update(logical_flops=batch*count_h*count_w*c['channels'],logical_weight_bytes=0,
                logical_status='verified plan valid pooling positions; comparison count inferred',
                logical_weight_scope='no weights',logical_work_complete=True)
        elif body=='embedding':
            width=t['stage_width'];seq=t['rows_per_batch']
            result.update(logical_flops=2*batch*seq*width,
                logical_status='verified full-batch plan sums; touched-table weight count needs frozen IDs',
                logical_weight_scope='unknown without frozen input IDs',logical_work_complete=False)
            if inputs is not None:
                word_name=buffers[t['operands'][0]]['name'];type_name=buffers[t['operands'][1]]['name']
                if word_name in inputs and type_name in inputs:
                    word=inputs[word_name];types=inputs[type_name]
                    if tuple(word.shape)==(batch,seq) and tuple(types.shape)==(batch,seq):
                        words=int(word.unique().numel());segments=int(types.unique().numel())
                        result.update(logical_weight_bytes=2*width*(words+segments+seq),
                            logical_status='verified plan extents and frozen unique token/type/position IDs; inferred touched BF16 table bytes',
                            logical_weight_scope='unique touched embedding table rows; full unused vocabulary not counted',logical_work_complete=True)
        elif body=='layer_norm':
            ids=t['operands'][1:3];weight=0;valid=True
            for bid in ids:
                b=buffers[bid]
                if b['dtype'] not in ('bf16','f32'):valid=False;break
                weight+=(2 if b['dtype']=='bf16' else 4)*b['constant']
            result.update(logical_weight_bytes=weight if valid else None,
                logical_status='verified plan affine bytes; exact normalization scalar operation count unknown',
                logical_weight_scope='explicit LayerNorm affine parameters',logical_work_complete=False)
        result['logical_geometry_source']=source
        rows.append(result)
    return rows
