"""Independent single-rank SGLang/Ascend requests; no TileMega computation."""
import argparse
import faulthandler
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time
import traceback


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('qwen3_0_6b','qwen3_1_7b','llama3_2_1b','llama3_2_3b','llama3_1_8b'),required=True)
    parser.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True)
    parser.add_argument('--outputs',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--graph',choices=('decode','off'),default='decode')
    parser.add_argument('--stream-output',choices=('delta','cumulative'),default='delta')
    parser.add_argument('--host-debug',action='store_true',help='Bounded setup pilot only; periodic Python stack, no timing admission')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path('/home/d50064638/junhuip');runs=root/'TileMega/runs/as2'
    out=args.out.absolute()
    if out.resolve()!=out or not out.is_relative_to(runs) or out==runs or out.exists():raise ValueError('Require a new canonical AS2 output')
    if not 1<=args.outputs<=1024 or not 0<=args.warmup<=1 or not 1<=args.repetitions<=3:raise ValueError('Bounded request count/length required')
    if os.environ.get('AS2_PHYSICAL_DEVICE_ID') not in ('0','1'):raise ValueError('Use the no-lock single-device wrapper')
    if args.host_debug:
        if args.outputs>4 or args.warmup or args.repetitions!=1:raise ValueError('Host debug is separate from measurements')
        faulthandler.dump_traceback_later(60,repeat=True)
    import torch
    import torch_npu
    if torch.version.cuda is not None or torch.npu.device_count()!=1:raise ValueError('Require CPU-build torch and one visible NPU')
    # The scheduler child owns compute. A parent NPU context neither fences
    # that process's stream nor helps loading the CPU token/weight manifest.
    torch.set_num_threads(1)
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
        # Worker reserves one context position; scheduler reserves one more.
        skip_tokenizer_init=True,dtype='bfloat16',context_length=1090,
        attention_backend='ascend',disable_cuda_graph=args.graph=='off',disable_prefill_cuda_graph=True,
        cuda_graph_bs_decode=[args.batch],cuda_graph_max_bs_decode=args.batch,
        incremental_streaming_output=args.stream_output=='delta',stream_interval=1,disable_radix_cache=True,
        mem_fraction_static=min(.5,max(.1,(manifest['unique_weight_bytes']+(6<<30))/(128<<30))),
        # ignore_eos admission reserves input rounding + another KV page as
        # well as the entire output. 1280 slots can stall before first token.
        page_size=128,max_running_requests=args.batch,max_total_tokens=args.batch*2048,
        chunked_prefill_size=max(128,args.batch*64),random_seed=20261016)
    identity=dict(schema='tilemega.as2.sglang_pilot.v1',system='sglang_npu',model_id=args.model,batch=args.batch,
        isl=64,osl=args.outputs,dtype='bf16',logical_capacity=1088,engine_context_length=1090,
        scheduler_guard_positions=2,physical_kv_pool_tokens=args.batch*2048,kv_page_size=128,
        initialization_seconds=time.monotonic()-started,
        weight_manifest_sha256=digest(manifest_path),config_sha256=manifest['config_sha256'],
        input_sha256=manifest['input_ids']['sha256'],physical_device_id=int(os.environ['AS2_PHYSICAL_DEVICE_ID']),
        logical_device_id=0,private_environment=os.sys.prefix,
        versions={n:metadata.version(n) for n in ('sglang','torch','torch-npu','transformers','sgl-kernel-npu')},
        collector_source_sha256=digest(Path(__file__)),collector_version=2,
        collector_parent_set_device=False,collector_parent_synchronize=False,
        timing_boundary='CPU token delivery by scheduler after sampling; exhausted stream completes request',
        baseline_config=dict(graph=args.graph,stream_output=args.stream_output,decode_capture_batch_sizes=[args.batch],disable_radix_cache=True,tp=1,pp=1,dp=1,
                             ascend_use_fia=os.environ.get('ASCEND_USE_FIA')),
        formal_performance_admitted=False)
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    try:
        for repetition in range(-args.warmup,args.repetitions):
            # Exhausting the preceding stream completes its request. With
            # radix caching disabled the next request owns fresh logical KV.
            # Resetting logical KV does not require flushing allocator pools.
            print(json.dumps(dict(event='request_start',repetition=repetition)),flush=True)
            t0=time.perf_counter();first=None;latest={}
            stream=engine.generate(input_ids=tokens,sampling_params=dict(temperature=0,max_new_tokens=args.outputs,ignore_eos=True),stream=True)
            for chunk in stream:
                for part in (chunk if isinstance(chunk,list) else [chunk]):
                    row=part.get('index',0)
                    if not 0<=row<args.batch or 'output_ids' not in part:raise ValueError('Unknown engine stream format')
                    if args.stream_output=='delta':latest.setdefault(row,[]).extend(part['output_ids'])
                    else:latest[row]=part['output_ids']
                    count=part.get('meta_info',{}).get('completion_tokens')
                    if count is not None and count!=len(latest[row]):raise ValueError('Engine stream token offsets disagree')
                    if len(latest[row])>args.outputs:raise ValueError('Engine stream repeated output tokens')
                if first is None and len(latest)==args.batch and all(len(x)>=1 for x in latest.values()):
                    first=time.perf_counter()
            last=time.perf_counter()
            if first is None or set(latest)!=set(range(args.batch)) or any(len(x)!=args.outputs for x in latest.values()):
                raise ValueError(f'Incomplete generation: expected {args.batch} rows x {args.outputs}, got '+str({i:len(x) for i,x in latest.items()}))
            output=torch.tensor([latest[i] for i in range(args.batch)],dtype=torch.int32,device='cpu')
            dest=out/f'request_{repetition}';dest.mkdir();(dest/'tokens.bin').write_bytes(output.numpy().tobytes())
            sample=dict(repetition=repetition,ttft_ms=1000*(first-t0),e2e_ms=1000*(last-t0),
                tpot_ms=1000*(last-first)/(args.outputs-1) if args.outputs>1 else None,
                output_tokens=args.batch*args.outputs,throughput_tokens_per_s=args.batch*args.outputs/(last-t0),
                output_sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),timing_domain='host_request_wall',formal_performance_admitted=False)
            (dest/'sample.json').write_text(json.dumps(sample,indent=2)+'\n');print(json.dumps(sample),flush=True)
    except BaseException:
        traceback.print_exc();raise
    finally:engine.shutdown()


if __name__=='__main__':main()
