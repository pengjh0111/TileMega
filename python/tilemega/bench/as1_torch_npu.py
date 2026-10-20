"""Measure four forwards of the already verified fixed random encoder."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn.attention import SDPBackend, sdpa_kernel
from tilemega.bench.as1_correctness import bert_block, controlled, error_stats, read_tensor, write_tensor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--attention-opmath',choices=['native','fp32'],default='fp32')
    args=parser.parse_args()
    directory,output=controlled(args.directory),controlled(args.output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new output file inside the wrapper job directory')
    cfg=json.loads((directory/'config.json').read_text())
    if cfg['test_id']!='E-2' or cfg['shape']!=[1,16,256] or cfg['bias'] or cfg['seed']!=20261016:
        raise ValueError('Only the verified bias-free encoder is admitted')
    for tensor in cfg['tensors'].values():
        if hashlib.sha256((directory/tensor['file']).read_bytes()).hexdigest()!=tensor['sha256']:
            raise ValueError('Prepared input, parameter or reference changed')
    if os.environ.get('AS1_PHYSICAL_DEVICE_ID')!='1' or torch.npu.device_count()!=1:
        raise RuntimeError('Use the no-lock wrapper with physical device 1 only')
    physical=ctypes.c_uint32(0xffffffff)
    runtime=ctypes.CDLL('libruntime.so')
    mapping=runtime.rtGetDevicePhyIdByIndex
    mapping.argtypes=[ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
    mapping.restype=ctypes.c_int
    if mapping(0,ctypes.byref(physical)) or physical.value!=1:
        raise RuntimeError('Runtime physical/logical mapping differs from the wrapper')
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32=False
    model=bert_block(attention_fp32=args.attention_opmath=='fp32').eval().bfloat16()
    state={}
    for name,p in model.named_parameters():
        tensor=cfg['tensors'][name]
        state[name]=read_tensor(directory/tensor['file'],list(p.shape),torch.bfloat16)
    model.load_state_dict(state,strict=True)
    model=model.to('npu:0')
    x=read_tensor(directory/'input.bin',cfg['shape'],torch.bfloat16).to('npu:0')
    mask=read_tensor(directory/'mask.bin',[1,16],torch.int64).to('npu:0')
    captured={}
    capture=[False]
    def remember(name):
        def hook(module,args,result):
            if capture[0]:
                captured[name]=result
        return hook
    def remember_input(name):
        def hook(module,args):
            if capture[0]:
                captured[name]=args[0]
        return hook
    for name in ('query','key','value','output','ln1','up','down','ln2'):
        getattr(model,name).register_forward_hook(remember(name))
    for name in ('output','ln1','down','ln2'):
        getattr(model,name).register_forward_pre_hook(remember_input(name+'_input'))
    samples=[]
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for forward in range(4):
            capture[0]=forward==3
            torch.npu.synchronize()
            start=time.perf_counter()
            result=model(x,mask)
            torch.npu.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            if forward:
                samples.append(elapsed)
    actual=result.cpu()
    output_tensor=write_tensor(output,actual)
    tol=cfg['tolerance']
    references={}
    for name,dtype in [('reference_bf16',torch.bfloat16),('reference_fp32',torch.float32)]:
        reference=read_tensor(directory/cfg['tensors'][name]['file'],cfg['shape'],dtype)
        references[name]=error_stats(actual,reference,tol['atol'],tol['rtol'])
    passed=references['reference_bf16']['numerical_pass']
    qkv=[captured[n].cpu().view(1,16,4,64) for n in ('query','key','value')]
    stages=[torch.stack(qkv,dim=3).reshape(1,16,768)]
    stages.extend(captured[n].cpu() for n in ('output_input','output','ln1_input','ln1','up','down_input','down','ln2_input','ln2'))
    stage_errors=[]
    for index,(observed,tensor) in enumerate(zip(stages,cfg['stage_references'])):
        if hashlib.sha256((directory/tensor['file']).read_bytes()).hexdigest()!=tensor['sha256']:
            raise ValueError('Prepared stage reference changed')
        expected=read_tensor(directory/tensor['file'],tensor['shape'],torch.bfloat16)
        stage_errors.append(dict(stage=index,**error_stats(observed,expected,tol['atol'],tol['rtol'])))
    receipt=dict(test_id='E-2',backend='torch_npu_eager_'+('explicit_fp32_attention' if args.attention_opmath=='fp32' else 'math_sdpa'),status='passed' if passed else 'numerical_failure',
        seed=cfg['seed'],shape=cfg['shape'],dtype='bf16',physical_device_id=physical.value,logical_device_id=0,
        device_locked=False,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        device_name=torch.npu.get_device_name(0),warmup_forwards=1,timed_forwards=3,samples_ms=samples,
        median_ms=statistics.median(samples),endpoint='host forward through device synchronization; resident data',
        references=references,stage_errors=stage_errors,output=output_tensor,tolerance=tol,
        attention_opmath=args.attention_opmath,matmul_allow_hf32=False,
        timing_status='consult_wrapper_telemetry; no exclusive access')
    output.with_suffix('.torch_npu.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
