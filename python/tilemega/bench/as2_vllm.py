"""Independent vLLM/Ascend requests with streamed canonical random weights."""
from __future__ import annotations
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time

ROOT=Path('/home/d50064638/junhuip')
RUNS=ROOT/'TileMega/runs/as2'
WEIGHTS=ROOT/'tilemega-as2/weights'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=('qwen3_0_6b','qwen3_1_7b','llama3_2_1b','llama3_2_3b','llama3_1_8b'))
    parser.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True)
    parser.add_argument('--outputs',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--graph',choices=('decode','off'),default='decode')
    parser.add_argument('--logprobs',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.absolute()
    if out.resolve()!=out or not out.is_relative_to(RUNS) or out==RUNS or out.exists():
        raise ValueError('Require a new canonical AS2 output directory')
    if not 1<=args.outputs<=1024 or not 0<=args.warmup<=1 or not 1<=args.repetitions<=3:
        raise ValueError('Bounded request count/length required')
    if os.environ.get('AS2_PHYSICAL_DEVICE_ID') not in ('0','1'):
        raise ValueError('Use the no-lock single-device wrapper')
    if os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')!='0':
        raise ValueError('This first canonical-loader pilot requires the in-process single-rank executor')
    import torch
    import torch_npu
    if torch.version.cuda is not None or torch.npu.device_count()!=1:
        raise ValueError('Require CPU-build torch and exactly one visible NPU')
    torch.npu.set_device(0);torch.set_num_threads(1)
    directory=WEIGHTS/args.model;manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text());config_path=ROOT/'TileMega/configs/ascend/as2_models'/f'{args.model}.json'
    if manifest['status']!='complete' or manifest['model_id']!=args.model or digest(config_path)!=manifest['config_sha256']:
        raise ValueError('Canonical model/config identity mismatch')
    if not 0<=args.logprobs<=manifest['config']['vocab_size']:
        raise ValueError('Logprob extent outside the true vocabulary')
    if args.logprobs and (args.outputs>4 or args.repetitions!=1 or args.warmup):
        raise ValueError('Full-output diagnostics are separate from long/timed requests')
    out.mkdir();model_path=out/'model';model_path.mkdir()
    (model_path/'config.json').write_bytes(config_path.read_bytes())
    input_file=directory/manifest['input_ids']['file']
    if digest(input_file)!=manifest['input_ids']['sha256']:raise ValueError('Input identity mismatch')
    tokens=torch.from_file(str(input_file),shared=False,size=16*64,dtype=torch.int32).view(16,64)[:args.batch].tolist()
    from vllm import LLM,SamplingParams
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    @register_model_loader('as2-canonical')
    class CanonicalLoader(BaseModelLoader):
        def download_model(self,model_config):
            if Path(model_config.model).resolve()!=model_path:
                raise ValueError('Only the local paired random model is admitted')

        def load_weights(self,model,model_config):
            self.download_model(model_config)
            consumed=[]
            def tensors():
                for record in manifest['tensors']:
                    file=directory/record['file']
                    if file.resolve().parent!=directory or file.stat().st_size!=record['bytes'] or digest(file)!=record['sha256']:
                        raise ValueError('Canonical weight mismatch: '+record['name'])
                    cpu=torch.from_file(str(file),shared=False,size=record['bytes']//2,dtype=torch.bfloat16).reshape(record['shape'])
                    consumed.append(record['name']);yield record['name'],cpu
            loaded=model.load_weights(tensors())
            if len(consumed)!=len(manifest['tensors']):raise ValueError('Engine did not consume every canonical weight')
            (out/'weight_load.json').write_text(json.dumps(dict(
                manifest_sha256=digest(manifest_path),config_sha256=manifest['config_sha256'],
                consumed=consumed,loaded_parameters=sorted(loaded),aliases=manifest['aliases'],
                loader='native model.load_weights; BF16 canonical mmap, no TileMega Body/reference'),indent=2)+'\n')

    config=manifest['config'];block_capacity=((1088+127)//128)*128
    kv_bytes=2*config['num_hidden_layers']*args.batch*config['num_key_value_heads']*block_capacity*config['head_dim']*2
    cache_budget=((kv_bytes+(128<<20)+(255<<20))//(256<<20))*(256<<20)
    started=time.monotonic()
    llm=LLM(model=str(model_path),tokenizer=None,skip_tokenizer_init=True,
        load_format='as2-canonical',dtype='bfloat16',max_model_len=1088,
        tensor_parallel_size=1,pipeline_parallel_size=1,distributed_executor_backend='uni',
        cpu_offload_gb=0,enforce_eager=args.graph=='off',disable_custom_all_reduce=True,
        compilation_config=dict(mode=3,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[args.batch]) if args.graph=='decode' else None,
        gpu_memory_utilization=min(.5,max(.1,(manifest['unique_weight_bytes']+cache_budget+(4<<30))/(128<<30))),
        kv_cache_memory_bytes=cache_budget,max_num_seqs=args.batch,max_num_batched_tokens=max(1088,args.batch*64),
        enable_prefix_caching=False,enable_chunked_prefill=True,disable_log_stats=True,
        max_logprobs=max(20,args.logprobs),seed=20261016)
    initialization=time.monotonic()-started;engine=llm.llm_engine
    params=SamplingParams(temperature=0,max_tokens=args.outputs,ignore_eos=True,detokenize=False,
                          logprobs=args.logprobs or None)
    identity=dict(schema='tilemega.as2.vllm_pilot.v1',system='vllm_ascend',model_id=args.model,batch=args.batch,
        isl=64,osl=args.outputs,dtype='bf16',weight_manifest_sha256=digest(manifest_path),
        config_sha256=manifest['config_sha256'],input_sha256=manifest['input_ids']['sha256'],
        physical_device_id=int(os.environ['AS2_PHYSICAL_DEVICE_ID']),logical_device_id=0,
        initialization_seconds=initialization,kv_cache_budget_bytes=cache_budget,
        versions={name:metadata.version(name) for name in ('vllm','vllm-ascend','torch','torch-npu','transformers','triton-ascend')},
        private_environment=os.sys.prefix,
        baseline_config=dict(enforce_eager=args.graph=='off',graph=args.graph,compilation_mode=3 if args.graph=='decode' else 0,
                             decode_capture_batch_sizes=[args.batch],executor='uni',multiprocessing=False,prefix_caching=False,
                             chunked_prefill=True,speculative_decoding=False),
        diagnostic_logprobs=args.logprobs,formal_performance_admitted=False)
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for repetition in range(-args.warmup,args.repetitions):
        if engine.has_unfinished_requests():raise ValueError('Previous request is not finished')
        torch.npu.synchronize();t0=time.perf_counter();first=None;latest={};ids={}
        for row,prompt in enumerate(tokens):
            request_id=f'as2-{repetition}-{row}'
            engine.add_request(request_id,{'prompt_token_ids':prompt},params)
            # RequestOutput restores the caller's ID, not the internal ID.
            ids[request_id]=row
        while engine.has_unfinished_requests():
            for result in engine.step():
                row=ids[result.request_id]
                if len(result.outputs)!=1:raise ValueError('Greedy request returned multiple sequences')
                latest[row]=result.outputs[0]
            if first is None and len(latest)==args.batch and all(len(x.token_ids)>=1 for x in latest.values()):
                torch.npu.synchronize();first=time.perf_counter()
        torch.npu.synchronize();last=time.perf_counter()
        if first is None or set(latest)!=set(range(args.batch)) or any(len(x.token_ids)!=args.outputs for x in latest.values()):
            raise ValueError('Incomplete generation or early EOS')
        output=torch.tensor([latest[i].token_ids for i in range(args.batch)],dtype=torch.int32)
        sample=dict(repetition=repetition,ttft_ms=1000*(first-t0),e2e_ms=1000*(last-t0),
                    tpot_ms=1000*(last-first)/(args.outputs-1) if args.outputs>1 else None,
                    output_tokens=args.batch*args.outputs,throughput_tokens_per_s=args.batch*args.outputs/(last-t0),
                    output_sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),timing_domain='host_request_wall',
                    full_logprob_diagnostic=bool(args.logprobs),formal_performance_admitted=False)
        dest=out/f'request_{repetition}';dest.mkdir();(dest/'tokens.bin').write_bytes(output.numpy().tobytes())
        if args.logprobs==config['vocab_size']:
            probabilities=torch.full((args.batch,config['vocab_size']),float('nan'))
            for row in range(args.batch):
                for token,value in latest[row].logprobs[-1].items():probabilities[row,token]=value.logprob
            if not torch.isfinite(probabilities).all():raise ValueError('Engine did not materialize the complete vocabulary')
            (dest/'final_logprobs.bin').write_bytes(probabilities.numpy().tobytes())
        (dest/'sample.json').write_text(json.dumps(sample,indent=2)+'\n');print(json.dumps(sample),flush=True)


if __name__=='__main__':main()
