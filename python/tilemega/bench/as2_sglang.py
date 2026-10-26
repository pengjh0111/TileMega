"""Independent single-rank SGLang/Ascend requests; no TileMega computation."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('qwen3_0_6b','qwen3_1_7b','llama3_2_1b','llama3_2_3b','llama3_1_8b'),required=True)
    parser.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True)
    parser.add_argument('--outputs',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path('/home/d50064638/junhuip');runs=root/'TileMega/runs/as2'
    out=args.out.absolute()
    if out.resolve()!=out or not out.is_relative_to(runs) or out==runs or out.exists():raise ValueError('Require a new canonical AS2 output')
    if not 1<=args.outputs<=1024 or not 0<=args.warmup<=1 or not 1<=args.repetitions<=3:raise ValueError('Bounded request count/length required')
    if os.environ.get('AS2_PHYSICAL_DEVICE_ID') not in ('0','1'):raise ValueError('Use the no-lock single-device wrapper')
    import torch
    import torch_npu
    if torch.version.cuda is not None or torch.npu.device_count()!=1:raise ValueError('Require CPU-build torch and one visible NPU')
    torch.npu.set_device(0);torch.set_num_threads(1)
    directory=root/'tilemega-as2/weights'/args.model;manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text());cfg=root/'TileMega/configs/ascend/as2_models'/f'{args.model}.json'
    if manifest['status']!='complete' or manifest['model_id']!=args.model or digest(cfg)!=manifest['config_sha256']:raise ValueError('Canonical identity mismatch')
    token_file=directory/manifest['input_ids']['file']
    if digest(token_file)!=manifest['input_ids']['sha256']:raise ValueError('Input identity mismatch')
    tokens=torch.from_file(str(token_file),shared=False,size=16*64,dtype=torch.int32).view(16,64)[:args.batch].tolist()
    out.mkdir();model_path=out/'model';model_path.mkdir();(model_path/'config.json').write_bytes(cfg.read_bytes())
    (model_path/'as2_weights.json').write_text(json.dumps(dict(model_id=args.model,manifest_sha256=digest(manifest_path)))+'\n')
    from sglang import Engine
    from tilemega.bench.as2_sglang_loader import CanonicalModelLoader
    started=time.monotonic()
    engine=Engine(model_path=str(model_path),load_format=CanonicalModelLoader,
        device='npu',base_gpu_id=0,tp_size=1,pp_size=1,dp_size=1,
        # Pinned scheduler caps generation at max_req_len - prompt_len - 1.
        skip_tokenizer_init=True,dtype='bfloat16',context_length=1089,
        attention_backend='ascend',disable_cuda_graph=True,disable_radix_cache=True,
        mem_fraction_static=min(.5,max(.1,(manifest['unique_weight_bytes']+(6<<30))/(128<<30))),
        max_running_requests=args.batch,max_total_tokens=args.batch*1280,
        chunked_prefill_size=max(128,args.batch*64),random_seed=20261016)
    identity=dict(schema='tilemega.as2.sglang_pilot.v1',system='sglang_npu',model_id=args.model,batch=args.batch,
        isl=64,osl=args.outputs,dtype='bf16',logical_capacity=1088,engine_context_length=1089,
        scheduler_guard_positions=1,initialization_seconds=time.monotonic()-started,
        weight_manifest_sha256=digest(manifest_path),config_sha256=manifest['config_sha256'],
        input_sha256=manifest['input_ids']['sha256'],physical_device_id=int(os.environ['AS2_PHYSICAL_DEVICE_ID']),
        logical_device_id=0,private_environment=os.sys.prefix,
        versions={n:metadata.version(n) for n in ('sglang','torch','torch-npu','transformers','sgl-kernel-npu')},
        baseline_config=dict(disable_graph=True,disable_radix_cache=True,tp=1,pp=1,dp=1,
                             ascend_use_fia=os.environ.get('ASCEND_USE_FIA')),
        formal_performance_admitted=False)
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    try:
        for repetition in range(-args.warmup,args.repetitions):
            if not engine.flush_cache():raise ValueError('Previous engine work did not drain')
            torch.npu.synchronize();t0=time.perf_counter();first=None;latest={}
            stream=engine.generate(input_ids=tokens,sampling_params=dict(temperature=0,max_new_tokens=args.outputs,ignore_eos=True),stream=True)
            for chunk in stream:
                for part in (chunk if isinstance(chunk,list) else [chunk]):
                    row=part.get('index',0)
                    if not 0<=row<args.batch or 'output_ids' not in part:raise ValueError('Unknown engine stream format')
                    latest[row]=part['output_ids']
                if first is None and len(latest)==args.batch and all(len(x)>=1 for x in latest.values()):
                    torch.npu.synchronize();first=time.perf_counter()
            torch.npu.synchronize();last=time.perf_counter()
            if first is None or set(latest)!=set(range(args.batch)) or any(len(x)!=args.outputs for x in latest.values()):
                raise ValueError(f'Incomplete generation: expected {args.batch} rows x {args.outputs}, got '+str({i:len(x) for i,x in latest.items()}))
            output=torch.tensor([latest[i] for i in range(args.batch)],dtype=torch.int32)
            dest=out/f'request_{repetition}';dest.mkdir();(dest/'tokens.bin').write_bytes(output.numpy().tobytes())
            sample=dict(repetition=repetition,ttft_ms=1000*(first-t0),e2e_ms=1000*(last-t0),
                tpot_ms=1000*(last-first)/(args.outputs-1) if args.outputs>1 else None,
                output_tokens=args.batch*args.outputs,throughput_tokens_per_s=args.batch*args.outputs/(last-t0),
                output_sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),timing_domain='host_request_wall',formal_performance_admitted=False)
            (dest/'sample.json').write_text(json.dumps(sample,indent=2)+'\n');print(json.dumps(sample),flush=True)
    finally:engine.shutdown()


if __name__=='__main__':main()
