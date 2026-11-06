"""Independent DNN baselines consuming identified torch.export/ONNX artifacts.

Run this file directly. It imports no TileMega package or model implementation.
GPU execution requires the external guard, lock and NUMA scheduler.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

_run_context = {}
CLASSIFICATION_MODELS = ('resnet18', 'resnet34', 'vgg16', 'alexnet', 'mbv1', 'mbv2')


class BaselineFailure(RuntimeError):
    def __init__(self, message, kind, metrics=None):
        self.failure_kind = kind
        self.metrics = metrics
        super().__init__(message)

_helper_directory = str(Path(__file__).resolve().parents[1] / 'serving')
sys.path.insert(0, _helper_directory)
try:
    from baseline_common import guarded_main, observe, require_guard, versions
finally:
    # Serving contains trace.py; retaining its path would shadow stdlib trace
    # during Dynamo/TensorRT imports after this standalone helper is loaded.
    sys.path.remove(_helper_directory)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def engine_cache_key(identity):
    """Content-address official build inputs; no framework objects are cached."""
    required = {'schema_version', 'onnx_sha256', 'exported_program_sha256',
                'source_checkpoint_sha256', 'precision', 'tensorrt_version',
                'gpu_uuid', 'compute_capability', 'builder_flags', 'static_batch'}
    if set(identity) != required or identity['schema_version'] != 1:
        raise ValueError('incomplete TensorRT engine cache identity')
    if identity['precision'] not in ('bf16', 'fp16') or not identity['gpu_uuid']:
        raise ValueError('invalid TensorRT engine cache precision or GPU identity')
    for name in ('onnx_sha256', 'exported_program_sha256', 'source_checkpoint_sha256'):
        value = identity[name]
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid TensorRT engine cache artifact digest')
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def serialized_engine_cache(directory, identity, build):
    """Build once under flock; validate every byte before each deserialization."""
    import fcntl
    import time
    key = engine_cache_key(identity)
    directory = Path(directory).resolve()
    entry = directory / key
    directory.mkdir(parents=True, exist_ok=True)
    lock_dir = directory / '.locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / (key + '.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        engine_path, meta_path = entry / 'engine.plan', entry / 'identity.json'
        hit = entry.exists()
        if hit:
            if (entry.is_symlink() or engine_path.is_symlink() or meta_path.is_symlink()
                    or not engine_path.is_file() or not meta_path.is_file()):
                raise ValueError('partial TensorRT engine cache entry; refusing reuse')
            meta = json.loads(meta_path.read_text())
            if meta.get('identity') != identity or meta.get('key') != key:
                raise ValueError('TensorRT engine cache identity conflict')
            serialized = engine_path.read_bytes()
            if hashlib.sha256(serialized).hexdigest() != meta.get('serialized_engine_sha256'):
                raise ValueError('TensorRT serialized engine cache SHA256 differs')
        else:
            serialized = bytes(build())
            if not serialized:
                raise RuntimeError('TensorRT engine build returned empty serialization')
            entry.mkdir()
            # A crash during publication leaves an explicit partial entry which
            # must be diagnosed, rather than silently trusted or overwritten.
            temp = entry / 'engine.plan.tmp'
            temp.write_bytes(serialized)
            temp.replace(engine_path)
            meta = dict(schema_version=1, key=key, identity=identity,
                        serialized_engine_sha256=sha(engine_path),
                        builder_pid=os.getpid(), built_at=time.time())
            temp = entry / 'identity.json.tmp'
            temp.write_text(json.dumps(meta, indent=2) + '\n')
            temp.replace(meta_path)
        return serialized, engine_path, dict(key=key, hit=hit,
            identity=identity, identity_file=str(meta_path), identity_sha256=sha(meta_path),
            engine_file=str(engine_path), builder_pid=meta['builder_pid'],
            deserialize_pid=os.getpid())


def load_export(directory):
    import torch
    # Exported BERT outputs use HF's registered pytree nodes.
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['model'].startswith('bert'):
        import transformers.modeling_outputs
    archive = directory / 'exported_program.pt2'
    expected = manifest['artifacts']['exported_program.pt2']
    if sha(archive) != expected:
        raise ValueError('export archive differs from its manifest')
    if not manifest.get('timing_eligible', manifest.get('accuracy_eligible', False)):
        raise ValueError('structure-only exports are ineligible for performance')
    return torch.export.load(archive), manifest


def make_inputs(program, batch, seed):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(seed)
    inputs = []
    for value in program.example_inputs[0]:
        shape = (batch, *value.shape[1:])
        inputs.append(torch.randn(shape, generator=generator).to(value.dtype))
    keywords = {}
    for name, value in program.example_inputs[1].items():
        shape = (batch, *value.shape[1:])
        if name == 'input_ids':
            # BERT-base's input vocabulary is part of its standard architecture.
            keywords[name] = torch.randint(30522, shape, generator=generator)
        elif name == 'attention_mask':
            keywords[name] = torch.ones(shape, dtype=value.dtype)
        else:
            keywords[name] = torch.zeros(shape, dtype=value.dtype)
    return tuple(inputs), keywords


def leaves(value):
    import torch
    return [item for item in torch.utils._pytree.tree_flatten(value)[0]
            if isinstance(item, torch.Tensor)]


def convert_inputs(arguments, keywords, *, dtype=None, device='cpu'):
    def convert(value):
        return value.to(device=device, dtype=dtype if dtype is not None and
                        value.is_floating_point() else value.dtype)
    return tuple(convert(x) for x in arguments), {k: convert(v) for k, v in keywords.items()}


def portable_module(source, *, dtype, device='cpu'):
    import torch
    # Factory dtype/device literals are not buffers: Module.to() cannot move
    # BERT's exported attention-mask factories or promote them for the oracle.
    module = torch.fx.GraphModule(source, copy.deepcopy(source.graph)).to(dtype=dtype, device=device)
    def convert(value):
        if value is torch.bfloat16:
            return dtype
        if isinstance(value, torch.device) and value.type == 'cpu':
            return torch.device(device)
        return value
    for node in module.graph.nodes:
        node.args = torch.utils._pytree.tree_map(convert, node.args)
        node.kwargs = torch.utils._pytree.tree_map(convert, node.kwargs)
    module.recompile()
    return module.eval()


def prepare_onnx(directory, batch, seed):
    import torch
    import onnx
    program, manifest = load_export(directory)
    dynamo = manifest['model'].startswith('bert')
    module = portable_module(program.module(), dtype=torch.float32)
    args, kwargs = make_inputs(program, batch, seed)
    args, kwargs = convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        outputs = leaves(module(*args, **kwargs))
    input_names = [f'input_{i}' for i in range(len(args))] + list(kwargs)
    output_names = [f'output_{i}' for i in range(len(outputs))]
    from safetensors.torch import save_file
    flat_inputs = torch.utils._pytree.tree_flatten((args, kwargs))[0]
    native_names = [spec.arg.name for spec in program.graph_signature.input_specs
                    if spec.kind.name == 'USER_INPUT']
    native_inputs = directory / f'inputs_b{batch}.safetensors'
    save_file({name: value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
               for name, value in zip(native_names, flat_inputs)}, str(native_inputs))
    path = directory / f'model_b{batch}_fp32.onnx'
    torch.onnx.export(module, args, str(path), kwargs=kwargs, input_names=input_names,
                      output_names=output_names, opset_version=18, dynamo=dynamo,
                      external_data=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    metadata = dict(evidence='verified', path=str(path.resolve()), sha256=sha(path),
        batch=batch, initialization_seed=manifest.get('initialization_seed'), input_seed=seed,
        exported_program_sha256=manifest['artifacts']['exported_program.pt2'],
        weight_semantics='exported BF16 weights promoted to FP32; identical stored values',
        accuracy_eligible=manifest.get('accuracy_eligible', False),
        onnx_exporter='torch.export/dynamo' if dynamo else 'legacy TorchScript',
        native_inputs_file=str(native_inputs.resolve()), native_inputs_sha256=sha(native_inputs),
        inputs=[dict(name=n, shape=list(t.shape), dtype=str(t.dtype))
                for n, t in zip(input_names, [*args, *kwargs.values()])],
        outputs=[dict(name=n, shape=list(t.shape), dtype=str(t.dtype))
                 for n, t in zip(output_names, outputs)])
    (directory / f'onnx_b{batch}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def numerical_metrics(actual, reference, name):
    import torch
    if len(actual) != len(reference):
        raise AssertionError('baseline output count differs from the exported reference')
    rows = []
    for index, (value, expected) in enumerate(zip(actual, reference)):
        value, expected = value.detach().float().cpu(), expected.detach().float().cpu()
        if value.shape != expected.shape or not torch.isfinite(value).all():
            raise AssertionError('baseline output has a wrong shape or nonfinite values')
        error = value - expected
        cosine = torch.nn.functional.cosine_similarity(value.reshape(1, -1),
                                                       expected.reshape(1, -1)).item()
        row = dict(output=index, shape=list(value.shape), cosine_vs_fp32=cosine,
                   max_abs_vs_fp32=error.abs().max().item(), mse_vs_fp32=error.square().mean().item())
        flattened_rows = value.reshape(-1, value.shape[-1])
        expected_rows = expected.reshape(-1, expected.shape[-1])
        row['cosine_min_vs_fp32'] = torch.nn.functional.cosine_similarity(
            flattened_rows, expected_rows, dim=1).min().item()
        if name in CLASSIFICATION_MODELS:
            row['top1_agreement_vs_fp32'] = (value.argmax(-1) == expected.argmax(-1)).float().mean().item()
        if name == 'nafnet':
            row['psnr_vs_fp32_db'] = -10 * math.log10(max(row['mse_vs_fp32'], 1e-30))
            mse = error.square().flatten(1).mean(-1)
            row['psnr_per_image_vs_fp32_db'] = (-10 * torch.log10(mse.clamp_min(1e-30))).tolist()
            row['psnr_min_vs_fp32_db'] = min(row['psnr_per_image_vs_fp32_db'])
            row['psnr_peak'] = 1.0
        rows.append(row)
    return rows


def correctness_gate(name, rows):
    if not rows:
        raise BaselineFailure('baseline has no output metrics', 'correctness', rows)
    for row in rows:
        if name == 'nafnet':
            passed = row.get('psnr_min_vs_fp32_db', row['psnr_vs_fp32_db']) >= 40
        else:
            threshold = .999 if name.startswith('bert') else .9999
            minimum = row.get('cosine_min_vs_fp32', row['cosine_vs_fp32'])
            passed = math.isfinite(minimum) and minimum >= threshold
            if name in CLASSIFICATION_MODELS:
                passed = passed and row['top1_agreement_vs_fp32'] == 1.0
        if not passed:
            raise BaselineFailure('baseline failed the FP32 equivalence gate: ' +
                                  json.dumps(rows, sort_keys=True), 'correctness', rows)
    return True


def inputs_digest(args, kwargs):
    import torch
    rows = []
    for name, value in [(str(i), x) for i, x in enumerate(args)] + list(kwargs.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError('DNN inputs must be tensors')
        value = value.detach().cpu().contiguous()
        rows.append(dict(name=name, dtype=str(value.dtype), shape=list(value.shape),
                         sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), rows


def torch_library_identity(metadata):
    import torch
    path = Path(torch.__file__).parent / 'lib/libtorch_cuda.so'
    metadata['library_path'] = str(path.resolve())
    metadata['library_sha256'] = sha(path)
    from torch._inductor.codecache import PyCodeCache
    modules = getattr(PyCodeCache, 'modules', [])
    if isinstance(modules, dict):
        modules = modules.values()
    files = sorted({str(Path(m.__file__).resolve()) for m in modules
                    if getattr(m, '__file__', None) and Path(m.__file__).is_file()})
    metadata['compiled_wrappers'] = [dict(path=p, sha256=sha(p)) for p in files]
    metadata['compiled_wrapper_digest'] = hashlib.sha256(
        json.dumps(metadata['compiled_wrappers'], sort_keys=True).encode()).hexdigest()


def torch_runner(module, args, kwargs, mode, metadata, *, freezing=False, fullgraph=True):
    import torch
    # Exported metadata assertions compare the device index as well as its type.
    device = torch.device('cuda', torch.cuda.current_device())
    module = portable_module(module, device=device, dtype=torch.bfloat16)
    args, kwargs = convert_inputs(args, kwargs, dtype=torch.bfloat16, device=device)
    torch._inductor.config.freezing = bool(freezing)
    compiled = torch.compile(module, mode=mode, fullgraph=fullgraph)
    metadata.update(compile_mode=mode, fullgraph=fullgraph,
                    freezing=bool(torch._inductor.config.freezing), freezing_requested=bool(freezing),
                    cuda_graph='enabled by official compile mode; actual capture is backend controlled')
    return lambda: compiled(*args, **kwargs)


def tensorrt_runner(directory, args, kwargs, precision, batch, out, metadata, engine_cache=None):
    import torch
    import tensorrt as trt
    info = json.loads((directory / f'onnx_b{batch}.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    onnx_path = Path(info['path'])
    if (sha(onnx_path) != info['sha256'] or info['batch'] != batch or
            info['exported_program_sha256'] != manifest['artifacts']['exported_program.pt2']):
        raise ValueError('ONNX identity or static batch differs')
    logger = trt.Logger(trt.Logger.WARNING)
    flags = dict(network_creation='EXPLICIT_BATCH', builder_optimization_level=5,
                 profiling_verbosity='DETAILED', precision_flag=precision.upper(),
                 workspace_limit='official default', tactic_sources='official default',
                 remaining_builder_configuration='official defaults')
    def build():
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError('TensorRT ONNX parse failed: ' + '\n'.join(
                str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.builder_optimization_level = 5
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.BF16 if precision == 'bf16' else trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT engine build failed')
        return bytes(serialized)
    engine_path = out / f'model_{precision}.plan'
    if engine_cache is None:
        serialized = build()
        engine_path.write_bytes(serialized)
    else:
        import subprocess
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        uuid = subprocess.check_output(['nvidia-smi', '--id=' + require_guard()[0],
            '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True).strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:
            raise ValueError('TensorRT engine cache requires a unique physical GPU UUID')
        identity = dict(schema_version=1, onnx_sha256=info['sha256'],
            exported_program_sha256=manifest['artifacts']['exported_program.pt2'],
            source_checkpoint_sha256=sha(directory / 'checkpoint/model.safetensors'),
            precision=precision, tensorrt_version=trt.__version__, gpu_uuid=uuid,
            compute_capability=[properties.major, properties.minor],
            builder_flags=flags, static_batch=batch)
        serialized, cached_path, cache_metadata = serialized_engine_cache(engine_cache, identity, build)
        if engine_path.exists() or engine_path.is_symlink():
            raise ValueError('refusing to replace an existing TensorRT output engine')
        engine_path.symlink_to(cached_path)
        metadata['engine_cache'] = cache_metadata
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('TensorRT deserialization failed')
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('TensorRT execution-context creation failed')
    buffers = {}
    incoming = dict(zip([x['name'] for x in info['inputs']], [*args, *kwargs.values()]))
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
                 trt.bfloat16: torch.bfloat16, trt.int32: torch.int32,
                 trt.int64: torch.int64, trt.bool: torch.bool}[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            buffers[name] = incoming[name].to(device='cuda', dtype=dtype).contiguous()
        else:
            buffers[name] = torch.empty(tuple(context.get_tensor_shape(name)), dtype=dtype, device='cuda')
        context.set_tensor_address(name, buffers[name].data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT initial execution failed')
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT CUDA graph capture failed')
    outputs = [buffers[item['name']] for item in info['outputs']]
    inspector = engine.create_engine_inspector()
    (out / 'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    metadata.update(precision=precision, builder_optimization_level=5, cuda_graph=True,
                    engine_sha256=sha(engine_path), onnx_sha256=info['sha256'],
                    precision_semantics='BF16/FP16 tactics permitted; layer precision recorded, FP32 fallback possible',
                    trtexec_crosscheck='unavailable: installed official TensorRT wheel has no trtexec binary')
    # Engine/context/runtime and binding buffers must outlive every graph replay.
    def run():
        _ = engine, context, runtime, buffers
        graph.replay()
        return outputs
    return run


def timed(run, warmup, iterations):
    import torch
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    values = [a.elapsed_time(b) for a, b in zip(starts, ends)]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise AssertionError('CUDA event returned an invalid duration')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--system', choices=('torchcompile', 'tensorrt', 'export-onnx', 'tvm'), required=True)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--compile-mode', choices=('max-autotune', 'reduce-overhead'), default='max-autotune')
    parser.add_argument('--freezing', action='store_true')
    parser.add_argument('--fullgraph', choices=('true', 'false'), default='true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--tvm-work-dir', type=Path)
    parser.add_argument('--tvm-max-trials', type=int, choices=(2000,), default=2000)
    parser.add_argument('--tvm-timeout-s', type=int, choices=(1800,), default=1800)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--engine-cache', type=Path,
                        help='explicit shared serialized TensorRT engine cache; strict input/GPU identity')
    parser.add_argument('--out', type=Path)
    options = parser.parse_args()
    _run_context.update(options={k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(options).items()}, out=options.out, phase='arguments')
    if not 1 <= options.batch <= 64 or options.warmup < 1 or options.iterations < 1:
        parser.error('invalid batch, warmup or iterations')
    if options.engine_cache is not None and options.system != 'tensorrt':
        parser.error('--engine-cache is only valid with --system tensorrt')
    if options.freezing and options.system != 'torchcompile':
        parser.error('--freezing is only valid with --system torchcompile')
    if options.system == 'tvm' and options.tvm_work_dir is None:
        parser.error('--system tvm requires --tvm-work-dir')
    import torch
    torch.set_num_threads(4)
    if options.system == 'export-onnx':
        print(json.dumps(prepare_onnx(options.export, options.batch, options.seed)), flush=True)
        return
    require_guard()
    if options.out is None:
        parser.error('GPU measurements require --out')
    if options.system == 'torchcompile' and options.precision != 'bf16':
        parser.error('torch.compile uses BF16 in the R16 protocol')
    options.out.mkdir(parents=True, exist_ok=True)
    if (options.out / 'measurements.json').exists() or (options.out / 'failure.json').exists():
        raise FileExistsError('retain prior baseline process evidence')
    _run_context['phase'] = 'initialization'
    guard_log = options.out / 'guard.jsonl'
    observe(guard_log, 'before-initialization')
    program, manifest = load_export(options.export)
    module = program.module()
    args, kwargs = make_inputs(program, options.batch, options.seed)
    input_hash, input_rows = inputs_digest(args, kwargs)
    reference_args, reference_kwargs = convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        reference = [x.detach().clone() for x in leaves(portable_module(copy.deepcopy(module), dtype=torch.float32)(
            *reference_args, **reference_kwargs))]
    metadata = dict(input_seed=options.seed, inputs='fixed synthetic tensors resident on device',
                    inputs_sha256=input_hash, input_tensors=input_rows,
                    reference_outputs_sha256=inputs_digest(reference, {})[0],
                    driver_sha256=sha(__file__),
                    reference='exported BF16 weights and inputs promoted to FP32; no accuracy certification')
    _run_context['configuration'] = metadata
    with torch.inference_mode():
        _run_context['phase'] = 'compile'
        if options.system == 'torchcompile':
            run = torch_runner(module, args, kwargs, options.compile_mode, metadata,
                               freezing=options.freezing, fullgraph=options.fullgraph == 'true')
        elif options.system == 'tensorrt':
            run = tensorrt_runner(options.export, args, kwargs, options.precision,
                                  options.batch, options.out, metadata, options.engine_cache)
        else:
            import importlib.util
            adapter_path = Path(__file__).with_name('tvm_baseline.py')
            spec = importlib.util.spec_from_file_location('dnn1_tvm_baseline', adapter_path)
            adapter = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = adapter
            spec.loader.exec_module(adapter)
            run = adapter.tvm_runner(options.export, program, args, kwargs, options.precision,
                                     options.batch, options.out, metadata,
                                     work_dir=options.tvm_work_dir,
                                     max_trials_global=options.tvm_max_trials,
                                     timeout_s=options.tvm_timeout_s)
        for index in range(options.warmup):
            run()
            _run_context['phase'] = 'warmup'
        torch.cuda.synchronize()
        _run_context['phase'] = 'correctness'
        actual = [x.detach().clone() for x in leaves(run())]
        torch.cuda.synchronize()
        correctness = numerical_metrics(actual, reference, manifest['model'])
        metadata['actual_outputs_sha256'] = inputs_digest(actual, {})[0]
        metadata['output_ranges'] = [dict(min=float(x.min()), max=float(x.max()),
            abs_max=float(x.abs().max()), nonzero=int(x.count_nonzero()), elements=x.numel())
            for x in actual]
        metadata['reference_output_ranges'] = [dict(min=float(x.min()), max=float(x.max()),
            abs_max=float(x.abs().max()), nonzero=int(x.count_nonzero()), elements=x.numel())
            for x in reference]
        if os.environ.get('TILEMEGA_DNN1_SAVE_CORRECTNESS') == '1':
            from safetensors.torch import save_file
            tensors = {f'{kind}_{i}': value.detach().cpu().contiguous()
                       for kind, values in [('actual', actual), ('reference', reference)]
                       for i, value in enumerate(values)}
            save_file(tensors, str(options.out / 'correctness_outputs.safetensors'))
        correctness_gate(manifest['model'], correctness)
        _run_context['phase'] = 'measurement'
        observe(guard_log, 'before-measurement')
        values = [] if options.correctness_only else timed(run, 0, options.iterations)
        observe(guard_log, 'after-measurement')
    if options.system == 'torchcompile':
        torch_library_identity(metadata)
    ordered = sorted(values)
    result = dict(schema_version=1, evidence='verified', system=options.system,
        model=manifest['model'], batch=options.batch, precision=options.precision,
        round_index=options.round_index, pid=os.getpid(), gpu_index=int(require_guard()[0]),
        process_role='correctness' if options.correctness_only else 'measurement',
        timing_status='not_measured' if options.correctness_only else 'measured',
        versions=versions(), accuracy_eligible=manifest.get('accuracy_eligible', False),
        random_weights=manifest.get('random_weights', False),
        exported_program_sha256=manifest['artifacts']['exported_program.pt2'],
        source_checkpoint_sha256=sha(options.export / 'checkpoint/model.safetensors'),
        measurement=None if options.correctness_only else dict(
            warmup=options.warmup, iterations=options.iterations,
            clock='CUDA events', fresh_processes='outer scheduler; this receipt is one process'),
        correctness_passed=True,
        correctness_protocol=dict(warmup=options.warmup, input_seed=options.seed,
                                  reference='same exported BF16 weights promoted to FP32'),
        median_ms=statistics.median(values) if values else None,
        p90_ms=ordered[math.ceil(0.9 * len(ordered)) - 1] if values else None,
        samples_ms=values, correctness=correctness, configuration=metadata,
        guard_log=str(guard_log.resolve()))
    (options.out / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples_ms'}), flush=True)


def entrypoint():
    try:
        main()
    except Exception as error:
        out = _run_context.get('out')
        if out is not None:
            out = Path(out)
            out.mkdir(parents=True, exist_ok=True)
            kind = getattr(error, 'failure_kind', None)
            phase = _run_context.get('phase', 'unknown')
            if kind is None:
                kind = phase if phase in ('compile', 'correctness') else 'runtime'
            failure = dict(evidence='verified failure', failure_kind=kind, phase=phase,
                           error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc(), options=_run_context.get('options'),
                           correctness=getattr(error, 'metrics', None),
                           configuration=_run_context.get('configuration', {}))
            path = out / 'failure.json'
            if not path.exists():
                path.write_text(json.dumps(failure, indent=2) + '\n')
        raise


if __name__ == '__main__':
    guarded_main(entrypoint)
