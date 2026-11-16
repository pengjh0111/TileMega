"""Closed-cohort, pretrained BF16 benchmark with host-observed token events."""
import argparse
import asyncio
import ctypes
import gzip
import functools
import hashlib
import json
import os
import resource
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
                if 'experts' in name:
                    row['parameter_shapes'] = {key: list(value.shape)
                                               for key, value in module.named_parameters(recurse=False)}
                if 'FusedMoEMethod' in cls:
                    for attr in ('flashinfer_cutlass_moe_enabled', 'rocm_aiter_moe_enabled', 'fused_experts'):
                        value = getattr(module, attr, None)
                        row[attr] = (value if value is None or isinstance(value, bool)
                                     else str(type(value)))
                cache = getattr(module, 'kv_cache', None)
                if isinstance(cache, (list, tuple)):
                    row['kv_cache_tensors'] = [dict(shape=list(value.shape), dtype=str(value.dtype),
                                                    device=str(value.device)) for value in cache
                                               if hasattr(value, 'dtype')]
                types[cls] = row
    parameters = {}
    for parameter in model.parameters():
        key = f'{parameter.dtype}/{parameter.device.type}'
        parameters[key] = parameters.get(key, 0) + parameter.numel()
    return dict(module_types=list(types.values()), parameter_elements=parameters)


def reset_allocator_peak(model=None):
    import torch
    torch.cuda.reset_peak_memory_stats()


def allocator_peak(model=None):
    import torch
    return dict(max_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_reserved_bytes=torch.cuda.max_memory_reserved(),
                scope='Torch CUDA allocator since post-warmup reset;excludes external CUDA allocations')


def worker_profile_start(model=None):
    import torch
    from moe_profile_instrumentation import set_nvtx_active
    set_nvtx_active(True)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()


def worker_profile_stop(model=None):
    import torch
    from moe_profile_instrumentation import set_nvtx_active
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    set_nvtx_active(False)


def worker_profile_boundary(worker, begin):
    import torch
    torch.cuda.synchronize()
    snapshot = dict(event='begin' if begin else 'end', observed_ns=time.perf_counter_ns(),
                    scope='CPU mirror of the last prepared GPU forward at synchronized worker boundary;'
                          'first following steady Decode uses the next sequence position.')
    try:
        runner = worker.model_runner
        count = runner.input_batch.num_reqs
        snapshot.update(request_ids=list(runner.input_batch.req_ids[:count]),
                        last_prepared_seq_lens=runner.seq_lens.np[:count].tolist(),
                        sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    except AttributeError as error:
        snapshot['unavailable'] = str(error)
    if begin:
        worker_profile_start()
    else:
        worker_profile_stop()
    return snapshot


def telemetry():
    index = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
    uuid = json.loads((ROOT / 'configs/moe_profile/devices.json').read_text())['devices'][index]['uuid']
    library = ctypes.CDLL('libnvidia-ml.so.1')
    handle = ctypes.c_void_p()
    calls = [library.nvmlInit_v2(), library.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle))]
    result = dict(time_unix=time.time(), gpu_uuid=uuid, nvml_errors=calls)
    for name, function, arguments in [('sm_mhz', 'nvmlDeviceGetClockInfo', [1]),
                                       ('memory_mhz', 'nvmlDeviceGetClockInfo', [2]),
                                       ('temperature_c', 'nvmlDeviceGetTemperature', [0]),
                                       ('power_mw', 'nvmlDeviceGetPowerUsage', [])]:
        value = ctypes.c_uint()
        error = getattr(library, function)(handle, *arguments, ctypes.byref(value))
        result[name] = value.value if error == 0 else None
        if error:
            result['nvml_errors'].append(error)
    library.nvmlShutdown()
    return result


class Vllm:
    def __init__(self, args, plan):
        # Only this local engine's trusted metadata/diagnostic callables cross RPC.
        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
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
        if args.mode == 'routing':
            options.update(enforce_eager=True, compilation_config=0)
        if args.attribution:
            options.update(enforce_eager=True, compilation_config=0)
        if args.no_cuda_graph:
            options['compilation_config'] = dict(level=3, cudagraph_mode=0)
        self.llm = LLM(**options)
        self.engine = self.llm.llm_engine
        self.metadata = dict(options=options, sampling=str(self.sampling),
                             trusted_local_callable_rpc=True,
                             worker_model=self.engine.apply_model(describe_model),
                             engine_config=str(self.engine.get_vllm_config()))
        self.round_id = 0
        if args.mode == 'routing':
            from moe_profile_instrumentation import install_vllm
            self.metadata['instrumentation'] = self.engine.apply_model(install_vllm)
        if args.attribution:
            from moe_profile_instrumentation import install_nvtx_vllm, set_nvtx_active
            self.metadata['nvtx_attribution'] = self.engine.apply_model(install_nvtx_vllm)
            self.engine.apply_model(functools.partial(set_nvtx_active, False))

    def arm_routes(self):
        from moe_profile_instrumentation import arm
        self.engine.apply_model(arm)

    def export_routes(self, path):
        from moe_profile_instrumentation import export
        return self.engine.apply_model(functools.partial(export, str(path)))

    def reset_peak(self):
        self.engine.apply_model(reset_allocator_peak)

    def peak(self):
        return self.engine.apply_model(allocator_peak)

    def profile_start(self):
        if not hasattr(self, 'worker_capture_boundaries'):
            self.worker_capture_boundaries = []
        self.worker_capture_boundaries.extend(self.engine.collective_rpc(worker_profile_boundary, args=(True,)))

    def profile_stop(self):
        self.worker_capture_boundaries.extend(self.engine.collective_rpc(worker_profile_boundary, args=(False,)))

    def run(self, rows, count, observe):
        self.round_id += 1
        prompts = {}
        start = time.perf_counter_ns()
        for i, ids in enumerate(rows):
            self.engine.add_request(str(i), {'prompt_token_ids': ids}, self.sampling)
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            now = time.perf_counter_ns()
            for output in outputs:
                prompts.setdefault(int(output.request_id), output.prompt_token_ids)
                observe(int(output.request_id), now, list(output.outputs[0].token_ids),
                        getattr(output, 'metrics', None))
        end = time.perf_counter_ns()
        if any(prompts.get(index) != row for index, row in enumerate(rows)):
            raise ValueError('vLLM actual prompt IDs differ from the frozen input')
        return start, end

    def close(self):
        self.engine.engine_core.shutdown()


class Sglang:
    def __init__(self, args, plan):
        import sglang as sgl
        options = dict(model_path=str(args.model), dtype='bfloat16', kv_cache_dtype='bf16',
                       tp_size=args.tp, load_format='auto', disable_radix_cache=True,
                       skip_tokenizer_init=True, max_running_requests=64,
                       cuda_graph_max_bs=64, cuda_graph_bs=[1, 4, 16, 64],
                       context_length=8450 if args.input_tokens == 8192 else 8448, mem_fraction_static=.93,
                       random_seed=plan['sampling']['seed'], chunked_prefill_size=8192,
                       stream_interval=1, disable_cuda_graph=args.eager or args.no_cuda_graph)
        if args.mode == 'routing':
            os.environ['SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR'] = str(args.out / 'routing_stats')
            options.update(disable_cuda_graph=True, expert_distribution_recorder_mode='stat',
                           expert_distribution_recorder_buffer_size=512)
        if args.attribution:
            options['disable_cuda_graph'] = True
        self.engine = sgl.Engine(**options)
        self.profile_output = args.out / 'native_profile'
        # get_server_info() awaits scheduler IPC;its response loop starts lazily.
        self.engine.tokenizer_manager.auto_create_handle_loop()
        self.sampling = dict(temperature=0, max_new_tokens=args.output_tokens,
                             ignore_eos=True)
        self.metadata = dict(options=options, sampling=self.sampling,
                             context_semantics='For8192+256,SGLang0.5.5 reserves one position in tp_worker '
                                               'and another in scheduler.init_req_max_new_tokens;'
                                               'use8450 so actual workload remains8192+256,never255.'
                                               if args.input_tokens == 8192 else 'inclusive workload below8448',
                             forced_length='ignore_eos=True;no stop strings/token IDs;actual O checked. '
                                           'SGLang0.5.5 min_new_tokens with skip_tokenizer_init '
                                           'dereferences a None tokenizer;failed diagnostic retained.',
                             server_info=self.engine.get_server_info())
        if args.attribution:
            proof_files = sorted((args.out / 'nvtx_sources').glob('*.json'))
            self.metadata['nvtx_attribution'] = [json.loads(path.read_text()) for path in proof_files]
            if not proof_files:
                raise ValueError('SGLang attribution startup hook produced no source evidence')

    def arm_routes(self):
        if os.environ.get('MOE_PROFILE_ROUTE_META_OUT'):
            self.engine.collective_rpc('moe_profile_arm_route_metadata')
        self.engine.start_expert_distribution_record()

    def export_routes(self, path):
        self.engine.stop_expert_distribution_record()
        self.engine.dump_expert_distribution_record()
        if os.environ.get('MOE_PROFILE_ROUTE_META_OUT'):
            self.engine.collective_rpc('moe_profile_export_route_metadata', path=str(path))
        return dict(path=str(path), format='SGLang native stat recorder;GPU-resident per-step counts',
                    position_scope='Whole diagnostic cohort;select stable full-cohort Decode samples '
                                   'using per-step assignment counts and retain phase uncertainty.')

    def profile_start(self):
        self.engine.start_profile(output_dir=str(self.profile_output), activities=['CUDA_PROFILER'],
                                  with_stack=False, record_shapes=False)

    def profile_stop(self):
        self.engine.stop_profile()

    def run(self, rows, count, observe):
        cumulative = [[] for _ in rows]
        prompts = {}
        start = time.perf_counter_ns()
        for chunk in self.engine.generate(input_ids=rows, sampling_params=self.sampling, stream=True):
            now = time.perf_counter_ns()
            for item in chunk if isinstance(chunk, list) else [chunk]:
                index = int(item.get('index', 0))
                ids = list(item['output_ids'])
                completed = item['meta_info']['completion_tokens']
                prompts[index] = item['meta_info']['prompt_tokens']
                if len(ids) == completed:
                    cumulative[index] = ids
                elif len(cumulative[index]) + len(ids) == completed:
                    cumulative[index] = cumulative[index] + ids
                else:
                    raise ValueError('SGLang stream skipped token IDs;cannot invent cumulative output')
                observe(index, now, cumulative[index], item.get('meta_info'))
        end = time.perf_counter_ns()
        if any(prompts.get(index) != len(row) for index, row in enumerate(rows)):
            raise ValueError('SGLang actual prompt token count differs from frozen input')
        return start, end

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
        if args.mode == 'routing':
            from moe_profile_instrumentation import install_trtllm
            instrumentation = install_trtllm()
        options = dict(model=str(args.model), backend='pytorch', dtype='bfloat16',
                       load_format='auto', tensor_parallel_size=args.tp,
                       skip_tokenizer_init=True, max_batch_size=64, max_seq_len=8448,
                       max_num_tokens=8192, enable_chunked_prefill=True,
                       use_cuda_graph=not (args.eager or args.no_cuda_graph),
                       cuda_graph_batch_sizes=[1, 4, 16, 64])
        if args.mode == 'routing':
            options['use_cuda_graph'] = False
        if args.attribution:
            options['use_cuda_graph'] = False
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
        try:
            self.model = self.engine._executor.engine.model_engine.model
            self.metadata['worker_model'] = describe_model(self.model)
        except AttributeError as error:
            self.metadata['worker_model_unavailable'] = str(error)
        if args.mode == 'routing':
            self.metadata['instrumentation'] = instrumentation
        if args.attribution:
            from moe_profile_instrumentation import install_nvtx_vllm, set_nvtx_active
            self.metadata['nvtx_attribution'] = [install_nvtx_vllm(self.model)]
            set_nvtx_active(False)

    def arm_routes(self):
        from moe_profile_instrumentation import arm
        arm()

    def export_routes(self, path):
        from moe_profile_instrumentation import export
        return export(path)

    def profile_start(self):
        worker_profile_start()

    def profile_stop(self):
        worker_profile_stop()

    async def generate(self, rows, observe):
        async def consume(index, future):
            async for item in future:
                observe(index, time.perf_counter_ns(), list(item.outputs[0].token_ids),
                        getattr(item, 'metrics', None))
        start = time.perf_counter_ns()
        futures = [self.engine.generate_async({'prompt_token_ids': ids}, self.sampling,
                                              streaming=True) for ids in rows]
        await asyncio.gather(*(consume(i, future) for i, future in enumerate(futures)))
        end = time.perf_counter_ns()
        if any(future.prompt_token_ids != row for future, row in zip(futures, rows)):
            raise ValueError('TensorRT-LLM actual prompt IDs differ from frozen input')
        return start, end

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
    parser.add_argument('--mode', choices=['benchmark', 'profile', 'routing'], default='benchmark')
    parser.add_argument('--phase', choices=['prefill', 'decode'], default='decode')
    parser.add_argument('--attribution', action='store_true')
    parser.add_argument('--no-cuda-graph', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_guard(args.tp)
    if args.framework == 'trtllm':
        _, maximum = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (1 if maximum == resource.RLIM_INFINITY else min(1, maximum), maximum))
    args.out.mkdir(parents=True, exist_ok=False)
    engine = None
    result = dict(status='started', arguments=vars(args), pid=os.getpid(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  python=sys.executable, started_unix=time.time(),
                  script_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in Path(__file__).parent.glob('moe_profile_*.py')})
    try:
        if (args.mode == 'benchmark' and args.rounds < 3) or args.warmup < 3:
            raise ValueError('formal protocol requires >=3 warmup and measured rounds')
        if args.attribution and (args.mode != 'profile' or args.no_cuda_graph):
            raise ValueError('source NVTX attribution requires a separate profile run')
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
        if args.mode != 'benchmark':
            result.update(diagnostic_only=True, scope='Separate diagnostic run;never formal timing evidence')
        if args.framework == 'trtllm' and args.mode in ('profile', 'routing'):
            result['teardown_policy'] = ('After successful inference,flushed artifacts and LLM.shutdown, '
                'use process-local os._exit to avoid the observed diagnostic CublasMMWrapper global-destructor '
                'SIGSEGV. Inference/shutdown failures still propagate;no shared system setting changed.')
        write(args.out / 'result.json', result)
        engine = {'vllm': Vllm, 'sglang': Sglang, 'trtllm': Trtllm}[args.framework](args, plan)
        write(args.out / 'engine.json', engine.metadata)
        measured = []
        result['capture_boundaries'] = []
        for iteration in range(args.warmup + args.rounds):
            events, final_meta = [], {}
            capturing = False
            profile_done = False
            lengths = [0] * args.batch
            diagnostic = iteration >= args.warmup and args.mode != 'benchmark'
            if iteration == args.warmup:
                if args.framework == 'vllm':
                    engine.reset_peak()
                elif args.framework == 'trtllm':
                    reset_allocator_peak()
            if diagnostic and args.mode == 'routing':
                engine.arm_routes()
            if diagnostic and args.mode == 'profile':
                import torch
                cudart = torch.cuda.cudart()
                if args.phase == 'prefill':
                    engine.profile_start()
                    capturing = True
                    boundary = dict(event='begin', phase=args.phase, observed_ns=time.perf_counter_ns(),
                                    host_output_counts=list(lengths), input_tokens=args.input_tokens,
                                    scope='Host-observed boundary;actual GPU KV lengths sampled in separate routing diagnostics')
                    result['capture_boundaries'].append(boundary)
                    torch.cuda.nvtx.mark('moe_profile/capture|' + json.dumps(boundary, separators=(',', ':')))

            def observe(index, now, ids, meta):
                nonlocal capturing, profile_done
                events.append(dict(index=index, observed_ns=now, ids=ids))
                final_meta[index] = meta
                lengths[index] = len(ids)
                if diagnostic and args.mode == 'profile' and not profile_done:
                    if args.phase == 'decode' and not capturing and min(lengths) >= 32:
                        engine.profile_start()
                        capturing = True
                        boundary = dict(event='begin', phase=args.phase, observed_ns=time.perf_counter_ns(),
                                        host_output_counts=list(lengths), input_tokens=args.input_tokens,
                                        scope='Host-observed boundary;actual GPU KV lengths sampled in separate routing diagnostics')
                        result['capture_boundaries'].append(boundary)
                        torch.cuda.nvtx.mark('moe_profile/capture|' + json.dumps(boundary, separators=(',', ':')))
                    if capturing and min(lengths) >= (1 if args.phase == 'prefill' else 48):
                        boundary = dict(event='end', phase=args.phase, observed_ns=time.perf_counter_ns(),
                                        host_output_counts=list(lengths), input_tokens=args.input_tokens)
                        result['capture_boundaries'].append(boundary)
                        torch.cuda.nvtx.mark('moe_profile/capture|' + json.dumps(boundary, separators=(',', ':')))
                        engine.profile_stop()
                        capturing, profile_done = False, True

            before_telemetry = telemetry()
            start, end = engine.run(rows, args.output_tokens, observe)
            after_telemetry = telemetry()
            if capturing:
                engine.profile_stop()
            if diagnostic and args.mode == 'profile' and not profile_done:
                raise ValueError('requested profiler ROI was not fully observed')
            if diagnostic and args.mode == 'routing':
                result['routing_export'] = engine.export_routes(args.out / 'routing_stats')
            try:
                metrics = summarize_round(args.batch, args.input_tokens, args.output_tokens,
                                          start, end, events)
            except ValueError:
                result['observed_output_counts'] = {index: len(next((event['ids'] for event in reversed(events)
                    if event['index'] == index), [])) for index in range(args.batch)}
                result['failure_engine_metrics'] = final_meta
                with gzip.open(args.out / 'failed_round_events.json.gz', 'wt') as stream:
                    json.dump(dict(start_ns=start, end_ns=end, events=events), stream)
                raise
            warmup = iteration < args.warmup
            name = ('warmup' if warmup else 'round') + str(iteration if warmup else iteration-args.warmup)
            write(args.out / (name + '.json'), dict(metrics=metrics, engine_metrics=final_meta,
                                                   before_telemetry=before_telemetry,
                                                   after_telemetry=after_telemetry))
            with gzip.open(args.out / (name + '_events.json.gz'), 'wt') as stream:
                json.dump(dict(start_ns=start, end_ns=end, events=events), stream)
            if not warmup:
                measured.append(metrics)
            print(json.dumps(dict(iteration=iteration, warmup=warmup, e2e_ms=metrics['e2e_ms'],
                                  tpot_ms=metrics['tpot_request_median_ms'])), flush=True)
        result.update(status='ok', summary=summarize_repeats(measured)
                      if args.mode == 'benchmark' else dict(diagnostic_rounds=len(measured)))
        result['worker_capture_boundaries'] = getattr(engine, 'worker_capture_boundaries', None)
        if args.framework == 'vllm':
            result['allocator_peak'] = engine.peak()
        elif args.framework == 'trtllm':
            result['allocator_peak'] = allocator_peak()
            try:
                result['engine_iteration_stats'] = engine.engine.get_stats(timeout=.1)
            except Exception as error:
                result['engine_iteration_stats_unavailable'] = repr(error)
        elif args.framework == 'sglang':
            result['final_server_info'] = engine.engine.get_server_info()
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        result['finished_unix'] = time.time()
        write(args.out / 'result.json', result)
        if engine is not None:
            engine.close()
    if args.framework == 'trtllm' and args.mode in ('profile', 'routing'):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
