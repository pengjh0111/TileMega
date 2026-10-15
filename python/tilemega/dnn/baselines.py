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
        if name in ('resnet18', 'mbv1', 'mbv2'):
            row['top1_agreement_vs_fp32'] = (value.argmax(-1) == expected.argmax(-1)).float().mean().item()
        if name == 'nafnet':
            row['psnr_vs_fp32_db'] = -10 * math.log10(max(row['mse_vs_fp32'], 1e-30))
            row['psnr_peak'] = 1.0
        rows.append(row)
    return rows


def torch_runner(module, args, kwargs, mode, metadata):
    import torch
    module = portable_module(module, device='cuda', dtype=torch.bfloat16)
    args, kwargs = convert_inputs(args, kwargs, dtype=torch.bfloat16, device='cuda')
    compiled = torch.compile(module, mode=mode, fullgraph=True)
    metadata.update(compile_mode=mode, fullgraph=True,
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
        uuid = subprocess.check_output(['nvidia-smi', '--id=' + require_guard(),
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
    parser.add_argument('--system', choices=('torchcompile', 'tensorrt', 'export-onnx'), required=True)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--compile-mode', choices=('max-autotune', 'reduce-overhead'), default='max-autotune')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--engine-cache', type=Path,
                        help='explicit shared serialized TensorRT engine cache; strict input/GPU identity')
    parser.add_argument('--out', type=Path)
    options = parser.parse_args()
    if not 1 <= options.batch <= 64 or options.warmup < 1 or options.iterations < 1:
        parser.error('invalid batch, warmup or iterations')
    if options.engine_cache is not None and options.system != 'tensorrt':
        parser.error('--engine-cache is only valid with --system tensorrt')
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
    guard_log = options.out / 'guard.jsonl'
    observe(guard_log, 'before-initialization')
    program, manifest = load_export(options.export)
    module = program.module()
    args, kwargs = make_inputs(program, options.batch, options.seed)
    reference_args, reference_kwargs = convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        reference = [x.detach().clone() for x in leaves(portable_module(copy.deepcopy(module), dtype=torch.float32)(
            *reference_args, **reference_kwargs))]
    metadata = dict(input_seed=options.seed, inputs='fixed synthetic tensors resident on device',
                    reference='exported BF16 weights and inputs promoted to FP32; no accuracy certification')
    with torch.inference_mode():
        run = (torch_runner(module, args, kwargs, options.compile_mode, metadata)
               if options.system == 'torchcompile' else tensorrt_runner(
                   options.export, args, kwargs, options.precision, options.batch, options.out, metadata,
                   options.engine_cache))
        for _ in range(options.warmup):
            run()
        torch.cuda.synchronize()
        actual = [x.detach().clone() for x in leaves(run())]
        torch.cuda.synchronize()
        correctness = numerical_metrics(actual, reference, manifest['model'])
        observe(guard_log, 'before-measurement')
        values = timed(run, 0, options.iterations)
        observe(guard_log, 'after-measurement')
    ordered = sorted(values)
    result = dict(schema_version=1, evidence='verified', system=options.system,
        model=manifest['model'], batch=options.batch, precision=options.precision,
        round_index=options.round_index, pid=os.getpid(), gpu_index=int(require_guard()),
        versions=versions(), accuracy_eligible=manifest.get('accuracy_eligible', False),
        random_weights=manifest.get('random_weights', False),
        exported_program_sha256=manifest['artifacts']['exported_program.pt2'],
        source_checkpoint_sha256=sha(options.export / 'checkpoint/model.safetensors'),
        measurement=dict(warmup=options.warmup, iterations=options.iterations,
                         clock='CUDA events', fresh_processes='outer scheduler; this receipt is one process'),
        median_ms=statistics.median(values), p90_ms=ordered[math.ceil(0.9 * len(ordered)) - 1],
        samples_ms=values, correctness=correctness, configuration=metadata,
        guard_log=str(guard_log.resolve()))
    (options.out / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples_ms'}), flush=True)


if __name__ == '__main__':
    guarded_main(main)
