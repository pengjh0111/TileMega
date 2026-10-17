#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Apply the frozen BF16 criteria to a full DNN's existing fixed input."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--candidate-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if sha(args.candidate)!=args.candidate_sha256:
        raise ValueError('selected DNN candidate changed')
    config=json.loads(args.candidate.read_text())
    if (os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or
            os.environ.get('CUDA_VISIBLE_DEVICES')!=str(config['gpu']) or
            os.environ.get('TILEMEGA_DEVICE_INDEX')!=str(config['gpu'])):
        raise ValueError('matching guarded GPU lease required')
    import torch
    from safetensors.torch import load_file
    from tilemega.dnn.check import NativeForward,cosine,psnr
    from tilemega.dnn.export import model
    from tilemega.dnn.gates import evaluate
    from tilemega.dnn import gates
    torch.set_num_threads(4)
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    export=Path(config['export'])
    manifest=json.loads((export/'manifest.json').read_text())
    for name,digest in config['export_files'].items():
        if sha(export/name)!=digest:raise ValueError('fixed DNN export changed: '+name)
    checkpoint=export/'checkpoint/model.safetensors'
    if sha(checkpoint)!=manifest['checkpoint']['sha256']:
        raise ValueError('fixed DNN checkpoint changed')
    name=manifest['model']
    fp32,_=model(name,random_weights=True,dtype=torch.float32)
    state=load_file(str(checkpoint),device='cpu')
    fp32.load_state_dict(state,strict=True)
    bf16=copy.deepcopy(fp32).to(torch.bfloat16)
    # These references represent the exact stored BF16 checkpoint. No missing
    # original FP32 weights or dataset accuracy is inferred from the gate.
    fp32,bf16=fp32.cuda().eval(),bf16.cuda().eval()
    if name in ('resnet18','mbv1','mbv2'):
        fp32=fp32.to(memory_format=torch.channels_last)
        bf16=bf16.to(memory_format=torch.channels_last)
    inputs=load_file(str(export/'inputs_b1.safetensors'),device='cuda')
    library=Path(config['directory'])/'plan.so'
    with torch.inference_mode(),NativeForward(library,export,config['bridge'],1) as native:
        arguments=[inputs[key] for key in native.inputs]
        actual=native(arguments)
        if name=='bert':
            kwargs=dict(zip(native.inputs,arguments))
            truth,baseline=fp32(**kwargs),bf16(**kwargs)
            hidden,pooler=actual
            metrics=dict(hidden_cosine=float(cosine(hidden,truth.last_hidden_state).mean()),
                bf16_hidden_cosine=float(cosine(baseline.last_hidden_state,truth.last_hidden_state).mean()),
                pooler_cosine=float(cosine(pooler,truth.pooler_output).mean()),
                bf16_pooler_cosine=float(cosine(baseline.pooler_output,truth.pooler_output).mean()))
        else:
            image=arguments[0]
            truth,baseline=fp32(image.float()),bf16(image)
            value,=actual
            if name=='nafnet':
                # Predeclared identity restoration is a synthetic target,
                # distinct from the unavailable SIDD validation ground truth.
                target=image.float()
                metrics=dict(tm_fp32_mse=float((value.float()-truth).square().mean()),
                    bf16_fp32_mse=float((baseline.float()-truth).square().mean()),
                    tm_gt_psnr=float(psnr(value,target).mean()),
                    bf16_gt_psnr=float(psnr(baseline,target).mean()))
            else:
                oracle=truth.argmax(-1)
                metrics=dict(cosine=float(cosine(value,truth).mean()),
                    bf16_cosine=float(cosine(baseline,truth).mean()),
                    tm_top1_match=float((value.argmax(-1)==oracle).float().mean()),
                    bf16_top1_match=float((baseline.argmax(-1)==oracle).float().mean()))
        gate=evaluate(name,metrics)
        result=dict(evidence='verified',scope='fixed-input random-checkpoint BF16 criteria; not dataset G-DNN',
            passed=gate['passed'],model=name,batch=1,metrics=metrics,gate=gate,
            artifact_id=native.identity['artifact_id'],binary_sha256=sha(library),
            candidate_sha256=args.candidate_sha256,checkpoint_sha256=sha(checkpoint),
            input_sha256=sha(export/'inputs_b1.safetensors'),gate_source_sha256=sha(gates.__file__),
            checker_sha256=sha(__file__),pid=os.getpid(),
            reference='stored BF16 checkpoint promoted exactly to FP32; same checkpoint BF16 PyTorch on GPU',
            target='synthetic identity restoration' if name=='nafnet' else 'FP32 output agreement')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    if not gate['passed']:raise AssertionError('fixed BF16 criteria failed: '+str(gate))


if __name__=='__main__':
    main()
