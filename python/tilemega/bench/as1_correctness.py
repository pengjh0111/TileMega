"""Prepare one fixed random AS-1 graph and compare a native output file."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tilemega.export_bridge import serialize

ROOT = Path('/home/d50064638/junhuip/TileMega/runs/as1')
SEED = 20261016


def controlled(path: Path) -> Path:
    path = path.absolute()
    if not path.is_relative_to(ROOT) or path == ROOT or path.resolve() != path:
        raise ValueError('Artifact path must be canonical and below runs/as1')
    return path


def write_tensor(path, tensor):
    tensor = tensor.detach().cpu().contiguous()
    data = tensor.view(torch.uint8).numpy().tobytes()
    path.write_bytes(data)
    return dict(file=path.name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read_tensor(path, shape, dtype):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).clone().reshape(shape)


def prepare_e0(out: Path, shape_index: int):
    out = controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    m, n, k = ((64, 128, 256), (33, 96, 129))[shape_index]
    torch.manual_seed(SEED)
    model = torch.nn.Linear(k, n, bias=False, dtype=torch.float32).eval()
    with torch.no_grad():
        model.weight.copy_(torch.randn(n, k) * 0.125)
    model = model.to(torch.bfloat16)
    x = (torch.randn(m, k) * 0.25).to(torch.bfloat16)
    with torch.no_grad():
        fp32 = torch.nn.functional.linear(x.float(), model.weight.float())
        bf16 = model(x)
    # Export a dynamic row domain; runtime binding chooses the fixed workload M.
    program = torch.export.export(model, (x[:4],),
        dynamic_shapes=({0: torch.export.Dim('rows', min=1, max=128)},))
    bridge = serialize(program)
    out.mkdir(parents=True)
    (out/'export.json').write_text(json.dumps(bridge, indent=2)+'\n')
    tensors = {
        'input': write_tensor(out/'input.bin', x),
        'weight_nk': write_tensor(out/'weight_nk.bin', model.weight),
        'weight_kn': write_tensor(out/'weight_kn.bin', model.weight.T.contiguous()),
        'reference_fp32': write_tensor(out/'reference_fp32.bin', fp32),
        'reference_bf16': write_tensor(out/'reference_bf16.bin', bf16),
    }
    config = dict(test_id='E-0', seed=SEED, model='gemm', shape_mnk=[m,n,k],
                  dtype='bf16', accumulator='fp32', tensors=tensors,
                  bridge='export.json', pipeline_status='prepared_not_executed',
                  weight_layout='logical N,K; native implementation packs K,N',
                  tolerance=dict(atol=0.001, rtol=0.0078125,
                      reason='one BF16 relative spacing, with 1e-3 near-zero allowance; FP32 accumulation'))
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(status='prepared', out=str(out), shape_mnk=[m,n,k], seed=SEED)))


def prepare_e1(out: Path, batch: int):
    out = controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.RMSNorm(128, eps=1e-5)
            self.up = torch.nn.Linear(128,256,bias=False)
            self.down = torch.nn.Linear(256,128,bias=False)
        def forward(self,x):
            return self.down(torch.nn.functional.silu(self.up(self.norm(x))))+x
    torch.manual_seed(SEED)
    model=Mlp().eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.copy_((1+torch.randn_like(p)*0.1) if name=='norm.weight' else torch.randn_like(p)*0.0625)
    model=model.bfloat16()
    x=(torch.randn(batch,128)*0.25).bfloat16()
    with torch.no_grad():
        bf16=model(x)
        fp32=model.float()(x.float())
        # A full FP32 graph removes five BF16 rounding points. Propagate
        # round-to-nearest error through each operator before device output
        # exists; relative error alone is unsuitable after cancellation.
        u=2**-8
        gamma=lambda k:(k*2**-24)/(1-k*2**-24)
        norm=model.norm(x.float())
        norm_error=(u+gamma(128)+2**-20)*norm.abs()
        def dot(value,error,weight):
            exact=torch.nn.functional.linear(value,weight)
            propagated=torch.nn.functional.linear(error,weight.abs())
            accumulation=gamma(weight.shape[1])*torch.nn.functional.linear(value.abs()+error,weight.abs())
            return exact,propagated+accumulation+u*(exact.abs()+propagated+accumulation)
        up,up_error=dot(norm,norm_error,model.up.weight)
        activation=torch.nn.functional.silu(up)
        # |SiLU'| <= 1+1/e < 1.5; this bound also covers negative inputs.
        activation_error=1.5*up_error+u*(activation.abs()+1.5*up_error)+2**-20
        down,down_error=dot(activation,activation_error,model.down.weight)
        fp32_rounding_bound=down_error+u*((down+x.float()).abs()+down_error)
        if not torch.isfinite(fp32_rounding_bound).all() or bool(((bf16.float()-fp32).abs()>fp32_rounding_bound).any()):
            raise RuntimeError('CPU BF16 graph exceeds the a priori FP32 rounding envelope')
    model=model.bfloat16()
    example=x.expand(4,-1).contiguous() if batch==1 else x
    program=torch.export.export(model,(example,),
        dynamic_shapes=({0:torch.export.Dim('batch',min=1,max=16)},))
    out.mkdir(parents=True)
    (out/'export.json').write_text(json.dumps(serialize(program),indent=2)+'\n')
    tensors={name:write_tensor(out/(name+'.bin'),tensor) for name,tensor in
             [('input',x),('reference_fp32',fp32),('reference_bf16',bf16),
              ('fp32_rounding_bound',fp32_rounding_bound)]}
    for name,p in model.named_parameters():
        tensors[name]=write_tensor(out/(name+'.bin'),p)
    config=dict(test_id='E-1',model='random_mlp',seed=SEED,batch=batch,
                shape=[batch,128],dtype='bf16',epsilon=1e-5,bias=False,
                graph=['RMSNorm','GEMM(128,256)','SiLU','GEMM(256,128)','ResidualAdd'],
                tensors=tensors,tolerance=dict(atol=0.001,rtol=0.0078125,
                    reason='one BF16 spacing for same-graph comparison; FP32 additionally uses precomputed operator rounding bound'),
                fp32_bound=dict(unit_roundoff=2**-8,accumulator_unit_roundoff=2**-24,
                    silu_lipschitz_upper_bound=1.5,
                    method='RMS scale/reduction rounding; |W| times input error; dot gamma_K; SiLU Lipschitz; BF16 rounding at each stage'),
                pipeline_status='prepared_not_executed')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',out=str(out),test_id='E-1',batch=batch)))


def bert_block(attention_fp32=False):
    class BertEncoderBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query=torch.nn.Linear(256,256,bias=False)
            self.key=torch.nn.Linear(256,256,bias=False)
            self.value=torch.nn.Linear(256,256,bias=False)
            self.output=torch.nn.Linear(256,256,bias=False)
            self.ln1=torch.nn.LayerNorm(256,eps=1e-12)
            self.up=torch.nn.Linear(256,1024,bias=False)
            self.down=torch.nn.Linear(1024,256,bias=False)
            self.ln2=torch.nn.LayerNorm(256,eps=1e-12)
        def forward(self,x,mask):
            batch=x.shape[0]
            q=self.query(x).view(batch,16,4,64).transpose(1,2)
            k=self.key(x).view(batch,16,4,64).transpose(1,2)
            v=self.value(x).view(batch,16,4,64).transpose(1,2)
            valid=mask.to(torch.bool).view(batch,1,1,16).expand(batch,1,16,16)
            if attention_fp32:
                scores=(q.float()@k.float().transpose(-2,-1))*0.125
                probabilities=torch.softmax(scores.masked_fill(~valid,float('-inf')),dim=-1)
                context=(probabilities@v.float()).to(x.dtype)
            else:
                context=torch.nn.functional.scaled_dot_product_attention(q,k,v,valid,dropout_p=0.0,is_causal=False)
            context=context.transpose(1,2).contiguous().view(batch,16,256)
            hidden=self.ln1(x+self.output(context))
            return self.ln2(hidden+self.down(torch.nn.functional.gelu(self.up(hidden),approximate='none')))
    return BertEncoderBlock()


def bert_reference_stages(model,x,mask):
    batch=x.shape[0]
    q,k,v=[projection(x).view(batch,16,4,64) for projection in (model.query,model.key,model.value)]
    packed=torch.stack([q,k,v],dim=3).reshape(batch,16,768)
    valid=mask.bool().view(batch,1,1,16).expand(batch,1,16,16)
    context=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),valid)
    context=context.transpose(1,2).contiguous().reshape(batch,16,256)
    projected=model.output(context);residual1=projected+x;norm1=model.ln1(residual1)
    up=model.up(norm1);gelu=torch.nn.functional.gelu(up,approximate='none')
    down=model.down(gelu);residual2=down+norm1;norm2=model.ln2(residual2)
    return [packed,context,projected,residual1,norm1,up,gelu,down,residual2,norm2]


def prepare_e2(out: Path):
    out=controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    torch.manual_seed(SEED)
    model=bert_block().eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.copy_((1+torch.randn_like(p)*0.1) if name.startswith('ln') and name.endswith('weight') else
                    torch.randn_like(p)*(0.1 if name.startswith('ln') else 0.0625))
    model=model.bfloat16()
    x=(torch.randn(1,16,256)*0.25).bfloat16()
    mask=torch.ones(1,16,dtype=torch.int64);mask[:,-3:]=0
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        bf16=model(x,mask)
        stage_values=bert_reference_stages(model,x,mask)
        if not torch.equal(stage_values[-1],bf16):
            raise RuntimeError('Independent staged CPU replay disagrees with exported model')
        fp32=model.float()(x.float(),mask)
    model=model.bfloat16()
    batch=torch.export.Dim('batch',min=1,max=4)
    program=torch.export.export(model,(x.expand(2,-1,-1).contiguous(),mask.expand(2,-1).contiguous()),
        dynamic_shapes=({0:batch},{0:batch}))
    out.mkdir(parents=True)
    (out/'export.json').write_text(json.dumps(serialize(program),indent=2)+'\n')
    tensors={name:write_tensor(out/(name+'.bin'),tensor) for name,tensor in
             [('input',x),('mask',mask),('reference_fp32',fp32),('reference_bf16',bf16)]}
    for name,p in model.named_parameters():
        tensors[name]=write_tensor(out/(name+'.bin'),p)
    stage_references=[write_tensor(out/('stage_'+str(index)+'.bin'),value)
                      for index,value in enumerate(stage_values)]
    config=dict(test_id='E-2',model='bert_encoder_block',seed=SEED,batch=1,sequence=16,hidden=256,
        heads=4,intermediate=1024,bias=False,shape=[1,16,256],dtype='bf16',dropout=0.0,
        epsilon=1e-12,activation='gelu_erf',mask='int64 explicit key padding, last three keys invalid',
        tensors=tensors,stage_references=stage_references,correctness_reference='reference_bf16',fp32_reference_policy='diagnostic_full_precision_graph',
        reference_attention_backend='CPU SDPBackend.MATH, FP32 opmath',
        tolerance=dict(atol=0.00390625,rtol=0.0078125,
            reason='one absolute BF16 spacing near unity plus one relative spacing for different FP32 reduction order'),
        pipeline_status='prepared_not_executed')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',out=str(out),test_id='E-2',model=config['model'],bias=False)))


def error_stats(actual, reference, atol, rtol):
    a, r = actual.float(), reference.float()
    diff = (a-r).abs()
    finite = torch.isfinite(a) & torch.isfinite(r)
    bad = ~finite | (diff > atol + rtol*r.abs())
    return dict(max_abs_error=float(diff.max()),
                max_rel_error=float((diff/r.abs().clamp_min(1e-30)).max()),
                rms_error=float(diff.square().mean().sqrt()),
                nonfinite_elements=int((~finite).sum()), failed_elements=int(bad.sum()),
                elements=a.numel(), numerical_pass=not bool(bad.any()))


def prepare_aiv(out: Path, kind: int, rows: int, width: int):
    out = controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    if kind not in (1, 4, 10) or rows not in (1, 4) or width not in (128, 129):
        raise ValueError('Use the bounded individual body cases')
    torch.manual_seed(SEED)
    x = (torch.randn(rows, width)*0.25).bfloat16()
    rhs = ((1+torch.randn(width)*0.1) if kind == 1 else
           torch.randn(rows, width)*0.25).bfloat16()
    if kind == 1:
        fp32 = x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-5)*rhs.float()
        bf16 = torch.nn.functional.rms_norm(x, (width,), rhs, 1e-5)
    elif kind == 4:
        fp32 = torch.nn.functional.silu(x.float())
        bf16 = torch.nn.functional.silu(x)
    else:
        fp32 = x.float()+rhs.float()
        bf16 = x+rhs
    out.mkdir(parents=True)
    tensors = {name:write_tensor(out/(name+'.bin'), tensor) for name,tensor in
               [('input',x),('rhs',rhs),('reference_fp32',fp32),('reference_bf16',bf16)]}
    config = dict(test_id='AIV-body', kind=kind, seed=SEED, shape=[rows,width],
                  tensors=tensors, epsilon=1e-5 if kind == 1 else None,
                  tolerance=dict(atol=0.001, rtol=0.0078125,
                    reason='one BF16 relative spacing and 1e-3 near-zero allowance'))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',out=str(out),kind=kind,shape=[rows,width])))


def prepare_encoder_vector(out: Path, operation: str, rows: int, width: int):
    out=controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    torch.manual_seed(SEED)
    x=(torch.randn(rows,width)*(2.0 if operation=='gelu' else 0.25)).bfloat16()
    gamma=(1+torch.randn(width)*0.1).bfloat16()
    beta=(torch.randn(width)*0.1).bfloat16()
    if operation=='gelu':
        fp32=torch.nn.functional.gelu(x.float(),approximate='none')
        bf16=torch.nn.functional.gelu(x,approximate='none')
    elif operation=='add':
        fp32=x.float()+gamma.float()
        bf16=x+gamma
    else:
        fp32=torch.nn.functional.layer_norm(x.float(),(width,),gamma.float(),beta.float(),1e-12)
        bf16=torch.nn.functional.layer_norm(x,(width,),gamma,beta,1e-12)
    out.mkdir(parents=True)
    tensors={name:write_tensor(out/(name+'.bin'),tensor) for name,tensor in
        [('input',x),('rhs',gamma),('beta',beta),('reference_fp32',fp32),('reference_bf16',bf16)]}
    cfg=dict(test_id='encoder-vector-body',operation=operation,kind=19 if operation=='layernorm' else 10 if operation=='add' else 4,
        seed=SEED,shape=[rows,width],epsilon=1e-12 if operation=='layernorm' else None,tensors=tensors,
        tolerance=dict(atol=0.001,rtol=0.0078125,reason='one BF16 relative spacing and 1e-3 near-zero allowance'))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',out=str(out),operation=operation,shape=[rows,width])))


def prepare_encoder_attention(out: Path, sequence: int, all_masked: bool, heads: int):
    out=controlled(out)
    if out.exists():
        raise FileExistsError('Use a new artifact directory')
    torch.manual_seed(SEED)
    packed=(torch.randn(sequence,heads,3,64)*0.5).bfloat16()
    mask=torch.ones(sequence,dtype=torch.int64)
    if sequence>1:
        mask[-min(3,sequence-1):]=0
    if all_masked:
        mask.zero_()
    q,k,v=[packed[:,:,index,:].transpose(0,1).unsqueeze(0) for index in range(3)]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        fp32=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float(),mask.bool()).squeeze(0).transpose(0,1).reshape(sequence,heads*64)
        bf16=torch.nn.functional.scaled_dot_product_attention(q,k,v,mask.bool()).squeeze(0).transpose(0,1).reshape(sequence,heads*64)
    out.mkdir(parents=True)
    tensors={name:write_tensor(out/(name+'.bin'),tensor) for name,tensor in
        [('input',packed),('rhs',mask),('beta',torch.zeros(heads*64,dtype=torch.bfloat16)),
         ('reference_fp32',fp32),('reference_bf16',bf16)]}
    cfg=dict(test_id='encoder-attention-body',kind=20,seed=SEED,shape=[sequence,heads*64],epsilon=None,
        all_masked=all_masked,heads=heads,head_dim=64,probability_dtype='fp32',tensors=tensors,
        tolerance=dict(atol=0.001,rtol=0.0078125,reason='one BF16 relative spacing and 1e-3 near-zero allowance'))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',out=str(out),sequence=sequence,all_masked=all_masked)))


def compare_aiv(directory: Path, output: Path):
    directory,output = controlled(directory),controlled(output)
    config = json.loads((directory/'config.json').read_text())
    actual = read_tensor(output,config['shape'],torch.bfloat16)
    tol = config['tolerance']
    stats = {name:error_stats(actual,read_tensor(directory/config['tensors'][name]['file'],
                config['shape'],dtype),tol['atol'],tol['rtol']) for name,dtype in
                [('reference_fp32',torch.float32),('reference_bf16',torch.bfloat16)]}
    job = json.loads((output.parent/'job.json').read_text())
    result = dict(status='passed' if job['status']=='passed' and all(s['numerical_pass'] for s in stats.values())
                  else 'failed',test_id=config['test_id'],kind=config['kind'],shape=config['shape'],
                  references=stats,tolerance=tol,epsilon=config['epsilon'],seed=SEED,
                  binary_sha256=job['executable_sha256'],physical_device_id=job['physical_device_id'],
                  tilemega_pipeline=False,device_locked=False)
    (output.parent/'numerical_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['status']=='passed' else 1


def compare(directory: Path, output: Path, pipeline: Path | None = None):
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)
    directory, output = controlled(directory), controlled(output)
    cfg = json.loads((directory/'config.json').read_text())
    for tensor in cfg['tensors'].values():
        require(hashlib.sha256((directory/tensor['file']).read_bytes()).hexdigest()==tensor['sha256'],
                'Prepared input, weight or reference changed')
    if cfg['test_id']=='E-0':
        m,n,_ = cfg['shape_mnk'];shape=[m,n]
    else:
        shape=cfg['shape']
    actual = read_tensor(output, shape, torch.bfloat16)
    refs = {name: read_tensor(directory/cfg['tensors'][name]['file'], shape, dtype)
            for name,dtype in [('reference_fp32',torch.float32),('reference_bf16',torch.bfloat16)]}
    tol=cfg['tolerance']
    stats={name:error_stats(actual,ref,tol['atol'],tol['rtol']) for name,ref in refs.items()}
    if 'fp32_rounding_bound' in cfg['tensors']:
        bound=read_tensor(directory/cfg['tensors']['fp32_rounding_bound']['file'],shape,torch.float32)
        diff=(actual.float()-refs['reference_fp32']).abs()
        allowed=bound+tol['atol']+tol['rtol']*refs['reference_bf16'].float().abs()
        failed=(~torch.isfinite(actual)) | (diff>allowed)
        stats['reference_fp32'].update(fixed_tolerance_failed_elements=stats['reference_fp32']['failed_elements'],
            failed_elements=int(failed.sum()),numerical_pass=not bool(failed.any()),
            precomputed_rounding_bound_file=cfg['tensors']['fp32_rounding_bound']['file'],
            rounding_bound_max=float(bound.max()),rounding_bound_min=float(bound.min()),
            rounding_bound_method=cfg['fp32_bound'])
    passed=stats[cfg['correctness_reference']]['numerical_pass'] if 'correctness_reference' in cfg else all(s['numerical_pass'] for s in stats.values())
    result=dict(test_id=cfg['test_id'], seed=cfg['seed'], shape=shape, dtype='bf16',
                tolerance=tol, references=stats, numerical_pass=passed,
                correctness_reference=cfg.get('correctness_reference','both_references'),
                fp32_reference_policy=cfg.get('fp32_reference_policy','fixed_tolerance_or_explicit_rounding_bound'),
                output_file=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                c0_status='not_evaluated_body_probe', c2_status='not_met_single_executor',
                c3_status='pending_exact_executor_repeats')
    if pipeline is not None:
        pipeline=controlled(pipeline)
        generated=json.loads((pipeline/'pipeline.json').read_text())
        plan=json.loads((pipeline/'plan.json').read_text())
        job=json.loads((output.parent/'job.json').read_text())
        require(Path(job['command'][1]).resolve()==directory,'Runtime input directory differs from numerical reference')
        records=[json.loads(line) for line in (output.parent/'stdout.log').read_text().splitlines()
                 if line.startswith('{')]
        launch=next(r for r in records if r.get('tilemega_pipeline'))
        trace=json.loads(Path(str(output)+'.trace.json').read_text())
        forwards=launch.get('forwards',1)
        require(job['status']=='passed' and launch['host_kernel_launches']==forwards and forwards in (1,4),'Launch receipt failed')
        require(launch['plan_trace_pass'] and launch['abi_roundtrip_pass'],'Plan/ABI receipt failed')
        require(launch.get('executor','task_event')==plan.get('executor','task_event'),'Executor receipt mismatch')
        require(generated['tasks']==launch['plan_tasks']==len(plan['tasks'])==len(trace),'Task count mismatch')
        for task,observed in zip(plan['tasks'],trace):
            require(task['logical_task']==observed['tile'] and task['owner']==observed['worker'],'Task ownership mismatch')
            if 'stage' in observed:
                require(task['stage']==observed['stage'],'Task stage mismatch')
            require(observed['executions']==observed['event']==plan['epochs']*forwards,'Task event/epoch mismatch')
        require('tmcg.tile_space' in (pipeline/'ctg.mlir').read_text(),'Missing CTG evidence')
        require('tmexec.skeleton' in (pipeline/'solved.mlir').read_text(),'Missing Solver evidence')
        executable=Path(job['command'][0])
        require(hashlib.sha256(executable.read_bytes()).hexdigest()==job['executable_sha256'],'Selected binary changed')
        result.update(c0_status='passed', single_persistent_launch_verified=True,
            pipeline_directory=str(pipeline), physical_device_id=job['physical_device_id'],
            logical_device_id=launch['logical_device_id'], binary_sha256=job['executable_sha256'],
            codegen_sha256=hashlib.sha256((pipeline/'kernel.cpp').read_bytes()).hexdigest(),
            device_locked=False, plan_tasks=len(plan['tasks']), groups=plan['launch_groups'],
            epochs=plan['epochs'], other_processes_present=job['other_processes_present_before'])
        result['executor']=plan.get('executor','task_event')
        result['forwards']=forwards
        result['single_persistent_launch_per_forward_verified']=True
        result['single_persistent_launch_verified']=forwards==1
        if cfg.get('stage_references'):
            stage_buffers=json.loads((pipeline/'stage_buffers.json').read_text())
            stage_errors=[]
            for item,reference in zip(stage_buffers,cfg['stage_references']):
                expected_file=directory/reference['file']
                require(hashlib.sha256(expected_file.read_bytes()).hexdigest()==reference['sha256'],'Stage reference changed')
                observed_file=Path(str(output)+'.buffer'+str(item['output_buffer'])+'.bin')
                observed=read_tensor(observed_file,reference['shape'],torch.bfloat16)
                expected=read_tensor(expected_file,reference['shape'],torch.bfloat16)
                stage_errors.append(dict(stage=item['stage'],**error_stats(observed,expected,tol['atol'],tol['rtol'])))
            require(len(stage_errors)==len(plan['stage_implementations']),'Incomplete stage numerical coverage')
            result['stage_errors']=stage_errors
            result['numerical_pass']=result['numerical_pass'] and all(s['numerical_pass'] for s in stage_errors)
    (output.parent/'numerical_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['numerical_pass'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare-e0');prepare.add_argument('--out',type=Path,required=True)
    prepare.add_argument('--shape-index',type=int,choices=[0,1],required=True)
    e1=sub.add_parser('prepare-e1');e1.add_argument('--out',type=Path,required=True)
    e1.add_argument('--batch',type=int,choices=[1,4],required=True)
    e2=sub.add_parser('prepare-e2');e2.add_argument('--out',type=Path,required=True)
    check=sub.add_parser('compare');check.add_argument('--directory',type=Path,required=True)
    check.add_argument('--output',type=Path,required=True)
    check.add_argument('--pipeline-directory',type=Path)
    body=sub.add_parser('prepare-aiv');body.add_argument('--out',type=Path,required=True)
    body.add_argument('--kind',type=int,choices=[1,4,10],required=True)
    body.add_argument('--rows',type=int,choices=[1,4],required=True)
    body.add_argument('--width',type=int,choices=[128,129],required=True)
    body_check=sub.add_parser('compare-aiv');body_check.add_argument('--directory',type=Path,required=True)
    body_check.add_argument('--output',type=Path,required=True)
    encoder=sub.add_parser('prepare-encoder-vector');encoder.add_argument('--out',type=Path,required=True)
    encoder.add_argument('--operation',choices=['layernorm','gelu','add'],required=True)
    encoder.add_argument('--rows',type=int,choices=[1,4],required=True)
    encoder.add_argument('--width',type=int,choices=[256,257],required=True)
    attention=sub.add_parser('prepare-encoder-attention');attention.add_argument('--out',type=Path,required=True)
    attention.add_argument('--sequence',type=int,choices=[1,13,16],required=True)
    attention.add_argument('--heads',type=int,choices=[1,4],default=4)
    attention.add_argument('--all-masked',action='store_true')
    args=p.parse_args()
    if args.command=='prepare-e0':
        prepare_e0(args.out,args.shape_index);return 0
    if args.command=='prepare-e1':
        prepare_e1(args.out,args.batch);return 0
    if args.command=='prepare-e2':
        prepare_e2(args.out);return 0
    if args.command=='prepare-aiv':
        prepare_aiv(args.out,args.kind,args.rows,args.width);return 0
    if args.command=='compare-aiv':
        return compare_aiv(args.directory,args.output)
    if args.command=='prepare-encoder-vector':
        prepare_encoder_vector(args.out,args.operation,args.rows,args.width);return 0
    if args.command=='prepare-encoder-attention':
        prepare_encoder_attention(args.out,args.sequence,args.all_masked,args.heads);return 0
    return compare(args.directory,args.output,args.pipeline_directory)


if __name__=='__main__':
    raise SystemExit(main())
