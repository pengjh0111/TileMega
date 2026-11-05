#!/usr/bin/env python3
"""Isolated AITemplate import, CPU profiler build, and guarded fixed-recipe sample.

GPU sampling waits for primary completion and a source-bound TVM result/failure
for every cell, after the remaining TVM tuners drain.
Each invocation emits one process receipt; P5/P-DNN orchestration is external.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
AIT_ROOT = ROOT / "runs/dnn1/resume/aitemplate"
AIT_COMMIT = "23a2e348857591ac41f4946f0177c53d833a0104"
RECIPE_SCHEMA = 8
BF16_POOL_SOURCE_SHA = 'fc84ab0252c59f86e85de54484beb4edbe4f59afb817013cbf9ce4401f8b6243'
BF16_GATHER_SOURCE_SHA = {
    'aitemplate/backend/cuda/cuda_common.py': '23a8e2ae50568c2d1cbf43c9aa3349c7f7527f6f251d0ed8ffbfd2124e5f3405',
    'aitemplate/backend/cuda/tensor/gather.py': '64881dd7a6db38b7c41fea8153b98d1e4c885697077d47072cd8df45ceabfd0e',
}
RECIPE = dict(schema=RECIPE_SCHEMA, frontend="uniform-static-ATen-NHWC", precision="bf16",
              target="cuda-sm90a", use_fp16_acc=False, use_fast_math=True,
              allow_cutlass_sm90=True, force_cutlass_sm90=False,
              optimize_graph=True, profile_timeout_s=500,
              profile_devices="guarded physical card; model runtime local0",
              num_runtimes=1, graph_mode=True, warmup=50, iterations=200,
              profiler_link_libraries=["cuda"], shared_link_libraries=["cuda"],
              backend_patch="dnn1_aitemplate_bf16_maxpool.patch",
              backend_gather_patch="dnn1_aitemplate_bf16_gather.patch",
              runtime_outputs="IR names/shapes/dtypes checked against official runtime enums")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_graph_recipe(graph):
    RECIPE.update(graph_mode=graph,
                  schema=RECIPE_SCHEMA if graph else f'{RECIPE_SCHEMA}-no-graph-reference')


def guarded_profile_devices(device_index):
    if (type(device_index) is not int or device_index not in (0, 1, 2, 3, 4, 5, 7) or
            os.environ.get('CUDA_VISIBLE_DEVICES') != str(device_index) or
            os.environ.get('TILEMEGA_DEVICE_INDEX') != str(device_index)):
        raise ValueError('profilers require the guarded physical card and singleton CVD')
    return [device_index]


def allocate_outputs(engine, graph_outputs, torch_module, dtype_to_enum, string_to_dtype):
    specs = {output['name']: output for output in graph_outputs}
    runtime = engine.get_output_name_to_index_map()
    if set(specs) != set(runtime):
        raise ValueError('compiled output names differ from the unchanged imported graph')
    buffers = {}
    for name, index in runtime.items():
        spec = specs[name]
        if engine.get_output_dtype(index) != dtype_to_enum(spec['dtype']):
            raise ValueError('compiled output dtype differs from imported IR: ' + name)
        shape = engine.get_output_maximum_shape(index)
        if list(shape) != list(spec['shape']):
            raise ValueError('compiled output shape differs from static imported IR: ' + name)
        buffers[name] = torch_module.empty(shape, device='cuda', dtype=string_to_dtype(spec['dtype']))
    return buffers


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_identity():
    paths = [Path(__file__), *sorted((ROOT / "python/tilemega/dnn").glob("aitemplate_*.py")),
             ROOT / "python/tilemega/dnn/baselines.py"]
    pool = AIT_ROOT / 'python/aitemplate/backend/cuda/pool2d/max_pool2d.py'
    if sha(pool) != BF16_POOL_SOURCE_SHA:
        raise ValueError('AIT BF16 max-pool backend source differs from the verified patch')
    paths.append(ROOT / 'scripts/dnn1_aitemplate_bf16_maxpool.patch')
    for relative, digest in BF16_GATHER_SOURCE_SHA.items():
        if sha(AIT_ROOT / 'python' / relative) != digest:
            raise ValueError('AIT BF16 gather backend source differs from the verified patch')
    paths.append(ROOT / 'scripts/dnn1_aitemplate_bf16_gather.patch')
    commit = subprocess.check_output(["git", "-C", str(AIT_ROOT / "source"),
                                      "rev-parse", "HEAD"], text=True).strip()
    if commit != AIT_COMMIT:
        raise ValueError("AITemplate checkout differs from the pinned commit")
    installed = {str(p.relative_to(AIT_ROOT / "python")): sha(p)
                 for p in sorted((AIT_ROOT / "python").rglob("*"))
                 if p.is_file() and p.suffix in (".py", ".so", ".h", ".cuh", ".cpp")}
    return dict(commit=commit, files={str(p): sha(p) for p in paths}, recipe=RECIPE,
                installed_sources_sha256=hashlib.sha256(
                    json.dumps(installed, sort_keys=True).encode()).hexdigest(),
                installed_source_count=len(installed))


def load_dependencies(cpu):
    sys.path[:0] = [str(AIT_ROOT / "python"), str(ROOT / "python")]
    if not cpu:
        sys.path.insert(0, str(ROOT / "scripts"))
        import dnn1_device
        dnn1_device.install()
    import torch
    torch.set_num_threads(4 if not cpu else 1)
    if cpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU import/build requires explicitly empty CUDA_VISIBLE_DEVICES")
        if "reject_cuda.so" not in os.environ.get("LD_PRELOAD", ""):
            raise ValueError("CPU import/build requires the reject_cuda preload")
        torch.cuda.is_available = lambda: False

        def forbidden(*args, **kwargs):
            raise RuntimeError("GPU initialization forbidden in CPU AIT preparation")

        torch.cuda._lazy_init = forbidden
        torch._C._cuda_getDeviceCount = forbidden
    from tilemega.dnn import baselines
    from tilemega.dnn.aitemplate_frontend import import_program
    return torch, baselines, import_program


def target():
    from aitemplate.backend.cuda.target_def import CUDA

    class HopperCUDA(CUDA):
        def compile_options(self):
            # Builder places these flags before objects; retain the driver even
            # when the host linker's default is --as-needed.
            return (super().compile_options() +
                    " -Xlinker=--no-as-needed -lcuda -Xlinker=--as-needed")

        def compile_cmd(self, executable=False):
            command = super().compile_cmd(executable=executable)
            # SM90 TMA profilers call the driver API when encoding tensor maps.
            return command + " -lcuda" if executable else command

        def _build_nvcc_compiler_options(self):
            # Pinned FBCUDA uses 90a for CUTLASS SM90 TMA kernels; its OSS
            # sibling leaves codegen at 90 even when those kernels are enabled.
            return [option.replace("compute_90", "compute_90a").replace("sm_90", "sm_90a")
                    if option.startswith("-gencode=") else option
                    for option in super()._build_nvcc_compiler_options()]

    return HopperCUDA(arch="90", use_fp16_acc=False, use_fast_math=True,
                      allow_cutlass_sm90=True, force_cutlass_sm90=False)


def graph_info(outputs, inputs, counters):
    return dict(inputs=[dict(name=n, shape=list(v.shape), dtype=str(v.dtype))
                        for n, v in inputs.items()],
                outputs=[dict(name=v._attrs["name"], shape=[d.value() for d in v.shape()],
                              dtype=v.dtype()) for v in outputs], layout_counters=counters)


def prepare(options):
    torch, baseline, importer = load_dependencies(cpu=True)
    identity = source_identity()
    rows = []
    cells = ([(m, b) for m in ("resnet18", "mbv1", "mbv2", "bert", "nafnet")
              for b in (1, 2, 4, 8, 16)] if options.all_cells else
             [(options.model, options.batch)])
    for model, batch in cells:
        row = dict(model=model, batch=batch, precision="bf16", status="pending")
        try:
            program, manifest = baseline.load_export(ROOT / "runs/dnn1/exports" / model)
            args, kwargs = baseline.make_inputs(program, batch, options.seed)
            outputs, inputs, counters = importer(program, args, kwargs, options.out)
            row.update(status="ir_created", **graph_info(outputs, inputs, counters),
                       exported_program_sha256=manifest["artifacts"]["exported_program.pt2"])
            if options.command == "cpu-profiler":
                row["profiler"] = cpu_profilers(outputs, options)
        except Exception as error:
            row.update(status="error", error=repr(error), traceback=traceback.format_exc())
        rows.append(row)
        save(options.out / "prepare.json", dict(identity=identity, rows=rows,
             scope="CPU IR/profiler compilation only; no GPU numerical/performance claim"))
        print(json.dumps(dict(model=model, batch=batch, status=row["status"])), flush=True)
    return int(any(row["status"] != "ir_created" for row in rows))


def cpu_profilers(outputs, options):
    from aitemplate import compiler
    from aitemplate.backend import builder, codegen
    from aitemplate.compiler.base import DynamicProfileStrategy
    from aitemplate.utils import graph_utils
    directory = options.out / "profilers"
    directory.mkdir()
    with target():
        graph = compiler.transform.toposort(outputs)
        compiler.transform.bind_constants(graph, {})
        compiler.transform.remove_unused_ops(graph)
        compiler.transform.remove_no_ops(graph)
        compiler.transform.name_graph(graph)
        compiler.transform.dedup_symbolic_name(graph)
        compiler.transform.mark_param_tensor(graph)
        graph = compiler.transform.optimize_graph(graph, str(directory), optimize=True)
        compiler.transform.mark_special_views(graph)
        compiler.transform.refine_graph(graph)
        graph_utils.dump_graph_debug_str_to_file(graph, str(directory), "optimized")
        profilers = [p for p in codegen.gen_profiler(
            graph, str(directory), DynamicProfileStrategy.MAX) if p is not None]
        if options.compile_profilers:
            os.environ["NUM_BUILDERS"] = str(options.builders)
            builder.get_compile_engine().make_profilers(profilers, str(directory))
    files = [p for p in directory.rglob("*") if p.is_file()]
    return dict(profiler_groups=len(profilers), compiled=options.compile_profilers,
                files={str(p): sha(p) for p in files},
                scope="Profiler sources/binaries only; profiling and final engine not run")


def require_tvm_complete():
    path = ROOT / "runs/dnn1/resume_tvm/aitemplate_release.json"
    release = json.loads(path.read_text()) if path.exists() else {}
    expected = {(model, batch) for model in ("resnet18", "mbv1", "mbv2", "bert", "nafnet")
                for batch in (1, 2, 4, 8, 16)}
    cells = release.get("cells", [])
    if (release.get("schema") != 1 or release.get("primary_cells_complete") != 125 or
            len(cells) != 25 or {(r["model"], r["batch"]) for r in cells} != expected):
        raise RuntimeError("AIT waits for 125 primary cells and 25 validated TVM results/failures")
    for record in [release["primary_evidence"], *cells]:
        source = Path(record["path"])
        if sha(source) != record["sha256"]:
            raise ValueError("AIT release evidence changed: " + str(source))
    for record in cells:
        if record["status"] not in ("validated", "genuine_failure") or not record.get("source_version"):
            raise ValueError("AIT release requires a source version and terminal cell evidence")
        if record["status"] == "genuine_failure" and record.get("failure_kind") not in (
                "import", "tune", "compile", "runtime", "numerical"):
            raise ValueError("an infrastructure rejection does not release the AIT gate")
    if not release.get("dispatchers"):
        raise ValueError("AIT release requires drained TVM dispatcher evidence")
    for filename in release["dispatchers"]:
        state = json.loads(Path(filename).read_text())
        if state.get("workers") or state.get("status") not in ("primary_complete", "needs_diagnosis"):
            raise RuntimeError("AIT GPU execution waits for the remaining TVM tuners to drain")
    return dict(path=str(path), sha256=sha(path),
                policy="primary125; TVM25 validated or genuine failure; drained TVM tuners")


def sample(options):
    publication = require_tvm_complete()
    torch, baseline, importer = load_dependencies(cpu=False)
    device_index = int(baseline.require_guard()[0])
    if device_index == 6:
        raise ValueError("GPU6 is excluded")
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("the fixed AIT recipe requires SM90")
    identity = source_identity()
    guard_log = options.out / "guard.jsonl"
    baseline.observe(guard_log, "before-initialization")
    export = ROOT / "runs/dnn1/exports" / options.model
    program, manifest = baseline.load_export(export)
    args, kwargs = baseline.make_inputs(program, options.batch, options.seed)
    reference_args, reference_kwargs = baseline.convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        reference = [x.detach().clone() for x in baseline.leaves(
            baseline.portable_module(copy.deepcopy(program.module()), dtype=torch.float32)(
                *reference_args, **reference_kwargs))]
    properties = torch.cuda.get_device_properties(0)
    identity.update(model=options.model, batch=options.batch,
                    exported_program_sha256=manifest["artifacts"]["exported_program.pt2"],
                    source_checkpoint_sha256=sha(export / "checkpoint/model.safetensors"),
                    gpu_uuid=str(properties.uuid), torch_version=torch.__version__,
                    profile_devices=guarded_profile_devices(device_index), runtime_local_device=0)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = options.cache / key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compiled = cache / "compiled.json"
        from aitemplate.compiler import compile_model, Model
        outputs, inputs, counters = importer(program, args, kwargs, cache)
        info = graph_info(outputs, inputs, counters)
        if compiled.exists():
            saved = json.loads(compiled.read_text())
            if saved["identity"] != identity or sha(saved["library"]) != saved["library_sha256"]:
                raise ValueError("AIT compiled cache identity/content differs")
            engine = Model(saved["library"], num_runtimes=1)
            cache_hit = True
        else:
            attempt = cache / "attempt.json"
            if attempt.exists():
                raise RuntimeError("incomplete AIT cache retained; use a fresh cache directory")
            save(attempt, dict(identity=identity, started=time.time()))
            engine = compile_model(outputs, target(), str(cache), "engine",
                                   profile_devs=guarded_profile_devices(device_index),
                                   num_runtimes=1, do_optimize_graph=True, profile_timeout=500)
            save(compiled, dict(identity=identity, graph=info, library=engine.lib_path,
                                library_sha256=sha(engine.lib_path)))
            cache_hit = False
    runtime_inputs = {name: value.cuda().contiguous() for name, value in inputs.items()
                      if name in engine.get_input_name_to_index_map()}
    if set(runtime_inputs) != set(engine.get_input_name_to_index_map()):
        raise ValueError("runtime inputs differ from the compiled input ABI")
    from aitemplate.compiler.dtype import dtype_str_to_enum
    from aitemplate.utils.torch_utils import string_to_torch_dtype
    actual_outputs = allocate_outputs(engine, info['outputs'], torch,
                                      dtype_str_to_enum, string_to_torch_dtype)

    def run():
        engine.run_with_tensors(runtime_inputs, actual_outputs,
            stream_ptr=torch.cuda.current_stream().cuda_stream, sync=False,
            graph_mode=RECIPE['graph_mode'])
        return [actual_outputs[f"output_{i}"] for i in range(len(actual_outputs))]

    with torch.inference_mode():
        for _ in range(50):
            run()
        torch.cuda.synchronize()
        actual = [x.detach().clone() for x in run()]
        torch.cuda.synchronize()
        metrics = baseline.numerical_metrics(actual, reference, options.model)
        baseline.correctness_gate(options.model, metrics)
        baseline.observe(guard_log, "before-measurement")
        values = [] if options.correctness_only else baseline.timed(run, 0, 200)
        baseline.observe(guard_log, "after-measurement")
    ordered = sorted(values)
    graph_observation = dict(requested=RECIPE['graph_mode'],
        scope='official runtime graph mode; whole-model capture is not asserted')
    if options.profile_capture:
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                run()
                torch.cuda.synchronize()
            trace = options.out / 'capture_trace.json'
            profiler.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            graph_observation.update(profiler_trace=str(trace),
                cuda_graph_launch_events=sum('cudagraphlaunch' in e.get('name', '').lower() for e in events),
                trace_sha256=sha(trace))
            graph_observation['capture_verified'] = graph_observation['cuda_graph_launch_events'] > 0
        except Exception as error:
            graph_observation.update(profiler_error=repr(error), capture_verified=False)
    save(options.out / "measurements.json", dict(schema_version=1, evidence="verified",
         system="aitemplate", model=options.model, batch=options.batch, precision="bf16",
         pid=os.getpid(), gpu_index=device_index, round_index=options.round_index,
         process_role="correctness" if options.correctness_only else "measurement",
         correctness_passed=True, correctness=metrics, samples_ms=values,
         median_ms=statistics.median(values) if values else None,
         p90_ms=ordered[math.ceil(.9 * len(ordered))-1] if values else None,
         guard_log=str(guard_log.resolve()), identity=identity, tvm_publication=publication,
         cache_hit=cache_hit, compiled_receipt_sha256=sha(compiled),
         library_sha256=sha(engine.lib_path),
         graph_mode=graph_observation,
         protocol=dict(warmup=50, iterations=0 if options.correctness_only else 200,
                       input_seed=options.seed, reference="same exported BF16 weights promoted FP32"),
         inputs_sha256=baseline.inputs_digest(args, kwargs)[0],
         output_sha256=baseline.inputs_digest(actual, {})[0],
         reference_sha256=baseline.inputs_digest(reference, {})[0]))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "cpu-profiler", "sample"])
    parser.add_argument("--model", choices=["resnet18", "mbv1", "mbv2", "bert", "nafnet"])
    parser.add_argument("--batch", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=AIT_ROOT / "engines")
    parser.add_argument("--compile-profilers", action="store_true")
    parser.add_argument("--builders", type=int, default=2)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument('--graph', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--profile-capture', action='store_true')
    options = parser.parse_args()
    configure_graph_recipe(options.graph)
    if options.all_cells and options.command != "prepare":
        parser.error("--all-cells is CPU graph import only")
    if not options.all_cells and not options.model:
        parser.error("--model is required")
    options.out.mkdir(parents=True, exist_ok=False)
    try:
        return sample(options) if options.command == "sample" else prepare(options)
    except Exception as error:
        save(options.out / "failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
