"""Closed-cohort, pretrained BF16 benchmark with host-observed token events."""
import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.serving.baseline_common import require_guard
from moe_profile_metrics import summarize_repeats, summarize_round, token_hash


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, ensure_ascii=False) + '\n')


def describe_model(model):
    """Run outside timing, in the engine worker; metadata only, no tensor copies."""
    types = {}
    for name, module in model.named_modules():
        cls = type(module).__module__ + '.' + type(module).__qualname__
        if any(word in (name + cls).lower() for word in ('moe', 'expert', 'attn', 'attention')):
            if cls not in types:
                row = dict(example_name=name, cls=cls)
                for attr in ('quant_method', 'impl', 'moe_kernel', 'runner', 'attn_backend'):
                    obj = getattr(module, attr, None)
                    if obj is not None:
                        row[attr] = type(obj).__module__ + '.' + type(obj).__qualname__
                types[cls] = row
    parameters = {}
    for parameter in model.parameters():
        key = f'{parameter.dtype}/{parameter.device.type}'
        parameters[key] = parameters.get(key, 0) + parameter.numel()
    return dict(module_types=list(types.values()), parameter_elements=parameters)


class Vllm:
    def __init__(self, args, plan):
        from vllm import LLM, SamplingParams
        self.sampling = SamplingParams(temperature=0, seed=plan['sampling']['seed'],
                                       max_tokens=args.output_tokens, min_tokens=args.output_tokens,
                                       ignore_eos=True, detokenize=False)
        options = dict(model=str(args.model), dtype='bfloat16', kv_cache_dtype='auto',
                       tensor_parallel_size=args.tp, skip_tokenizer_init=True,
                       max_model_len=8448, max_num_seqs=64, max_num_batched_tokens=8192,
                       gpu_memory_utilization=.93, enable_prefix_caching=False,
                       enable_chunked_prefill=True, swap_space=0,
                       enforce_eager=args.eager, seed=plan['sampling']['seed'])
        self.llm = LLM(**options)
        self.engine = self.llm.llm_engine
        self.metadata = dict(options=options, sampling=str(self.sampling),
                             worker_model=self.engine.apply_model(describe_model),
                             engine_config=str(self.engine.get_vllm_config()))
        self.round_id = 0

    def run(self, rows, count, observe):
        self.round_id += 1
        start = time.perf_counter_ns()
        for i, ids in enumerate(rows):
            self.engine.add_request(str(i), {'prompt_token_ids': ids}, self.sampling)
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            now = time.perf_counter_ns()
            for output in outputs:
                observe(int(output.request_id), now, list(output.outputs[0].token_ids),
                        getattr(output, 'metrics', None))
        return start, time.perf_counter_ns()

    def close(self):
        self.engine.engine_core.shutdown()


class Sglang:
    def __init__(self, args, plan):
        import sglang as sgl
        options = dict(model_path=str(args.model), dtype='bfloat16', kv_cache_dtype='bf16',
                       tp_size=args.tp, load_format='auto', disable_radix_cache=True,
                       skip_tokenizer_init=True, max_running_requests=64,
                       cuda_graph_max_bs=64, cuda_graph_bs=[1, 4, 16, 64],
                       context_length=8448, mem_fraction_static=.93,
                       random_seed=plan['sampling']['seed'], chunked_prefill_size=8192,
                       stream_interval=1, disable_cuda_graph=args.eager)
        self.engine = sgl.Engine(**options)
        self.sampling = dict(temperature=0, max_new_tokens=args.output_tokens,
                             min_new_tokens=args.output_tokens, ignore_eos=True)
        self.metadata = dict(options=options, sampling=self.sampling,
                             server_info=self.engine.get_server_info())

    def run(self, rows, count, observe):
        cumulative = [[] for _ in rows]
        start = time.perf_counter_ns()
        for chunk in self.engine.generate(input_ids=rows, sampling_params=self.sampling, stream=True):
            now = time.perf_counter_ns()
            for item in chunk if isinstance(chunk, list) else [chunk]:
                index = int(item.get('index', 0))
                ids = list(item['output_ids'])
                completed = item['meta_info']['completion_tokens']
                if len(ids) == completed:
                    cumulative[index] = ids
                elif len(cumulative[index]) + len(ids) == completed:
                    cumulative[index] = cumulative[index] + ids
                else:
                    raise ValueError('SGLang stream skipped token IDs;cannot invent cumulative output')
                observe(index, now, cumulative[index], item.get('meta_info'))
        return start, time.perf_counter_ns()

    def close(self):
        self.engine.shutdown()


class Trtllm:
    def __init__(self, args, plan):
        # This standard engine option avoids MPI spawning for the TP1 study.
        if args.tp != 1:
            raise ValueError('TP>1 requires the separately recorded static MPI launcher')
        os.environ['TLLM_WORKER_USE_SINGLE_PROCESS'] = '1'
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
        options = dict(model=str(args.model), backend='pytorch', dtype='bfloat16',
                       load_format='auto', tensor_parallel_size=args.tp,
                       skip_tokenizer_init=True, max_batch_size=64, max_seq_len=8448,
                       max_num_tokens=8192, enable_chunked_prefill=True,
                       use_cuda_graph=not args.eager,
                       cuda_graph_batch_sizes=[1, 4, 16, 64])
        self.engine = LLM(**options, kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=.85))
        eos = json.loads((args.model / 'config.json').read_text())['eos_token_id']
        eos = eos[0] if isinstance(eos, list) else eos
        self.sampling = SamplingParams(max_tokens=args.output_tokens,
                                       min_tokens=args.output_tokens, ignore_eos=True,
                                       top_k=1, temperature=0, seed=plan['sampling']['seed'],
                                       end_id=eos, pad_id=eos, detokenize=False,
                                       return_perf_metrics=True)
        self.metadata = dict(options=options, sampling=str(self.sampling),
                             kv_cache=dict(enable_block_reuse=False,
                                           free_gpu_memory_fraction=.85,
                                           dtype='unquantized BF16 model dtype'),
                             actual_engine_args=str(self.engine.args),
                             worker_use_single_process=True)

    async def generate(self, rows, observe):
        async def consume(index, future):
            async for item in future:
                observe(index, time.perf_counter_ns(), list(item.outputs[0].token_ids),
                        getattr(item, 'metrics', None))
        start = time.perf_counter_ns()
        futures = [self.engine.generate_async({'prompt_token_ids': ids}, self.sampling,
                                              streaming=True) for ids in rows]
        await asyncio.gather(*(consume(i, future) for i, future in enumerate(futures)))
        return start, time.perf_counter_ns()

    def run(self, rows, count, observe):
        return asyncio.run(self.generate(rows, observe))

    def close(self):
        self.engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['vllm', 'sglang', 'trtllm'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--input-tokens', type=int, required=True)
    parser.add_argument('--output-tokens', type=int, required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_guard(args.tp)
    args.out.mkdir(parents=True, exist_ok=False)
    engine = None
    result = dict(status='started', arguments=vars(args), pid=os.getpid(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  python=sys.executable, started_unix=time.time())
    try:
        if args.rounds < 3 or args.warmup < 3:
            raise ValueError('formal protocol requires >=3 warmup and measured rounds')
        plan = json.loads((ROOT / 'configs/moe_profile/workload.json').read_text())
        verified = json.loads((ROOT / 'runs/moe_profile/model_metadata/download_result.json').read_text())
        if not verified['passed'] or Path(verified['model_path']).resolve() != args.model.resolve():
            raise ValueError('real checkpoint must pass all pinned shard SHA256 checks')
        with gzip.open(args.inputs, 'rt') as stream:
            corpus = json.load(stream)
        rows = corpus['inputs'][str(args.input_tokens)][:args.batch]
        if len(rows) != args.batch or any(len(row) != args.input_tokens for row in rows):
            raise ValueError('frozen input token count differs from requested matrix cell')
        result.update(model_revision=verified['revision'],
                      input_file_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                      input_hashes=[token_hash(row) for row in rows],
                      scope='A: closed cohort;no new arrivals;no profiler;host-observed stream metrics')
        write(args.out / 'result.json', result)
        engine = {'vllm': Vllm, 'sglang': Sglang, 'trtllm': Trtllm}[args.framework](args, plan)
        write(args.out / 'engine.json', engine.metadata)
        measured = []
        for iteration in range(args.warmup + args.rounds):
            events, final_meta = [], {}

            def observe(index, now, ids, meta):
                events.append(dict(index=index, observed_ns=now, ids=ids))
                final_meta[index] = meta

            start, end = engine.run(rows, args.output_tokens, observe)
            metrics = summarize_round(args.batch, args.input_tokens, args.output_tokens,
                                      start, end, events)
            warmup = iteration < args.warmup
            name = ('warmup' if warmup else 'round') + str(iteration if warmup else iteration-args.warmup)
            write(args.out / (name + '.json'), dict(metrics=metrics, engine_metrics=final_meta))
            with gzip.open(args.out / (name + '_events.json.gz'), 'wt') as stream:
                json.dump(dict(start_ns=start, end_ns=end, events=events), stream)
            if not warmup:
                measured.append(metrics)
            print(json.dumps(dict(iteration=iteration, warmup=warmup, e2e_ms=metrics['e2e_ms'],
                                  tpot_ms=metrics['tpot_request_median_ms'])), flush=True)
        result.update(status='ok', summary=summarize_repeats(measured))
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        result['finished_unix'] = time.time()
        write(args.out / 'result.json', result)
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    main()
