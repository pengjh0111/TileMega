"""Independent DNN TVM adapter for the pinned DNN1 source build.

Importing this file initializes no framework or GPU. GPU entry points require
the external scheduler's inherited guard and device lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import contextlib
import fcntl
import hashlib
import importlib.metadata
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


TVM_COMMIT = "7b5acb32bd271342a24130f4ef439ff6c2e52c6c"
TARGET = "cuda -arch=sm_90a"
# v0.27 requires structured Target construction; this is the same fixed target.
TARGET_CONFIG = {"kind": "cuda", "arch": "sm_90a"}
RECIPE = dict(schema=6, builder_device_visibility="none", builder_target_scope="explicit", completed_measurements="commit-before-next-build", builder_workers="min(8, CPU affinity)", ffi="pinned-submodule-python-build", importer_output_dtype="aten-with-fp32-accumulation", target=TARGET, import_exported_program=True,
              keep_params_as_input=True, partition=["cublas", "cudnn"],
              run_codegen=True, pipeline="zero", tune_before_default_schedule=True,
              schedules=["Matmul", "GEMV", "Reduction", "GeneralReduction", "Fallback"],
              tuning=dict(max_trials_global=2000, timeout_s=1800,
                          search_strategy="evolutionary", cost_model="xgb"))


def _device_target_config(tvm):
    _guard()
    device = tvm.cuda(0)
    # This release's CUDA canonicalizer only fills arch. MetaSchedule requires
    # explicit hardware limits; obtain them from the already-guarded device.
    return dict(TARGET_CONFIG,
                max_threads_per_block=int(device.max_threads_per_block),
                max_shared_memory_per_block=int(device.max_shared_memory_per_block),
                thread_warp_size=int(device.warp_size))


class TvmBaselineError(RuntimeError):
    def __init__(self, message, failure_kind):
        super().__init__(message)
        self.failure_kind = failure_kind


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _guard():
    device = os.environ.get("TILEMEGA_DEVICE_INDEX", "")
    if (not device.isdigit() or os.environ.get("CUDA_VISIBLE_DEVICES") != device
            or os.environ.get("TILEMEGA_GPU_LOCK_HELD") != "1"):
        raise TvmBaselineError("TVM requires the externally guarded single-card session",
                               "tvm_cuda_build_or_selfcheck")


def _root():
    default = Path(__file__).resolve().parents[3] / "runs/dnn1/tvm"
    return Path(os.environ.get("TILEMEGA_DNN1_TVM_ROOT", default)).resolve()


def _source_identity():
    root = _root()
    repo = Path(__file__).resolve().parents[3]
    patches = [repo / "scripts" / name for name in
               ("dnn1_tvm_llvm23_compat.patch", "dnn1_tvm_buildinfo.patch", "dnn1_tvm_torch_dtype.patch")]
    files = [root / "source" / name for name in
             ("src/target/llvm/llvm_instance.cc", "python/tvm/support/libinfo.py",
              "python/tvm/relax/frontend/torch/base_fx_graph_translator.py",
              "python/tvm/relax/frontend/torch/exported_program_translator.py")]
    diff = subprocess.check_output(["git", "-C", str(root / "source"),
                                    "diff", "--binary", "HEAD"])
    ffi_commit = subprocess.check_output(["git", "-C", str(root / "source/3rdparty/tvm-ffi"),
                                          "rev-parse", "HEAD"], text=True).strip()
    return dict(commit=TVM_COMMIT, ffi_commit=ffi_commit,
                patches={str(path): _sha(path) for path in patches},
                modified_sources={str(path): _sha(path) for path in files},
                source_diff_sha256=hashlib.sha256(diff).hexdigest(),
                original_stub_source_sha256="c12245def0db46c5700cd8e5924027d12a96e3ed1298830de5c07cfc691f5b88")


def _load_tvm():
    root = _root()
    local = [str(root / "ffi_python"), str(root / "python_deps"), str(root / "source/python")]
    for entry in reversed(local):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    inherited = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    os.environ["PYTHONPATH"] = os.pathsep.join(local + [entry for entry in inherited if entry not in local])
    os.environ["TVM_LIBRARY_PATH"] = str(root / "build/lib")
    info = None
    try:
        import tvm
        import tvm_ffi
        if not Path(tvm_ffi.__file__).resolve().is_relative_to(root / "ffi_python"):
            raise RuntimeError("TVM FFI must use the Python build from the pinned C++ submodule")
        if not Path(tvm.__file__).resolve().is_relative_to(root / "source/python"):
            raise RuntimeError(f"TVM did not import from the isolated source tree: {tvm.__file__}")
        commit = subprocess.check_output(
            ["git", "-C", str(root / "source"), "rev-parse", "HEAD"], text=True).strip()
        if commit != TVM_COMMIT:
            raise RuntimeError(f"TVM checkout differs: {commit}, expected {TVM_COMMIT}")
        info = dict(tvm.support.libinfo())
        if tvm.base._RUNTIME_ONLY:
            raise RuntimeError("TVM source build imported as runtime-only; compiler library is unavailable")
        for key in ("USE_CUDA", "USE_CUBLAS", "USE_CUDNN"):
            if str(info.get(key, "OFF")).upper() in ("OFF", "0", "FALSE", "", "NONE"):
                raise RuntimeError(f"TVM source-build self-check {key}={info.get(key)!r}")
        return tvm, tvm_ffi, info
    except Exception as error:
        if isinstance(error, TvmBaselineError):
            raise
        wrapped = TvmBaselineError(str(error), "tvm_cuda_build_or_selfcheck")
        wrapped.libinfo = info
        raise wrapped from error


def _libraries():
    root = _root()
    candidates = sorted((root / "build/lib").glob("libtvm*.so"))
    candidates += sorted((root / "build").glob("libtvm*.so"))
    candidates += sorted((root / "ffi_python/tvm_ffi").glob("**/libtvm_ffi.so"))
    result = []
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        command = subprocess.run(["readelf", "-d", str(path)], text=True, capture_output=True)
        result.append(dict(path=str(path), sha256=_sha(path),
                           dynamic_section=command.stdout,
                           readelf_error=command.stderr if command.returncode else None))
    if not result:
        raise TvmBaselineError("no local TVM source-build libraries found",
                               "tvm_cuda_build_or_selfcheck")
    return result


def tvm_selfcheck(out=None, metadata=None):
    """Run only through the external GPU guard; return all raw libinfo values."""
    _guard()
    result = dict(tvm_version=None, tvm_commit=TVM_COMMIT, libinfo=None,
                  enabled=None, libraries=None, cuda_exist=False)
    try:
        tvm, _, info = _load_tvm()
        result.update(tvm_version=tvm.__version__, libinfo=info,
                      enabled={key: True for key in ("USE_CUDA", "USE_CUBLAS", "USE_CUDNN")},
                      libraries=_libraries())
        buildinfo_module = importlib.import_module("tvm.support.libinfo")
        result["original_stub_libinfo"] = buildinfo_module.original_stub_libinfo()
        result["buildinfo_evidence"] = buildinfo_module.buildinfo_evidence()
        result["source_identity"] = _source_identity()
        result["cuda_exist"] = bool(tvm.cuda(0).exist)
        if not result["cuda_exist"]:
            raise RuntimeError("tvm.cuda(0).exist is false")
    except Exception as error:
        result["libinfo"] = getattr(error, "libinfo", result["libinfo"])
        if result["libinfo"] is not None:
            result["enabled"] = {key: str(result["libinfo"].get(key, "OFF")).upper()
                                  not in ("OFF", "0", "FALSE", "", "NONE")
                                  for key in ("USE_CUDA", "USE_CUBLAS", "USE_CUDNN")}
        result["error"] = traceback.format_exc()
        if metadata is not None:
            metadata["tvm_selfcheck"] = result
        if out is not None:
            _json(Path(out) / "tvm_selfcheck.json", result)
        raise TvmBaselineError(str(error), "tvm_cuda_build_or_selfcheck") from error
    if metadata is not None:
        metadata["tvm_selfcheck"] = result
    if out is not None:
        _json(Path(out) / "tvm_selfcheck.json", result)
    return result


def _precision_program(program, precision):
    import torch
    if precision == "bf16":
        return program, dict(method="unchanged BF16 ExportedProgram; no re-export")
    if precision != "fp16":
        raise ValueError("TVM precision must be bf16 or fp16")
    result = copy.deepcopy(program)

    def convert(value):
        if value is torch.bfloat16:
            return torch.float16
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            converted = value.to(torch.float16)
            if isinstance(value, torch.nn.Parameter):
                converted = torch.nn.Parameter(converted, requires_grad=value.requires_grad)
            return converted
        return value

    for mapping in (result.state_dict, result.constants):
        for name, value in list(mapping.items()):
            mapping[name] = convert(value)
    for node in result.graph.nodes:
        node.args = torch.utils._pytree.tree_map(convert, node.args)
        node.kwargs = torch.utils._pytree.tree_map(convert, node.kwargs)
        if "val" in node.meta:
            node.meta["val"] = torch.utils._pytree.tree_map(convert, node.meta["val"])
        tensor_meta = node.meta.get("tensor_meta")
        if getattr(tensor_meta, "dtype", None) == torch.bfloat16:
            node.meta["tensor_meta"] = tensor_meta._replace(dtype=torch.float16)
    result.graph_module.recompile()
    return result, dict(method="copied EP BF16 state/constants/FX dtype metadata and literals to FP16; no re-export")


def _shape(value):
    info = value.ty
    return [int(item) for item in info.shape.values]


def _import_program(program, precision, batch):
    from tvm.relax.frontend.torch import from_exported_program
    program, dtype = _precision_program(program, precision)
    try:
        return from_exported_program(program, keep_params_as_input=True), dtype
    except Exception as error:
        node_info = None
        frame = error.__traceback__
        while frame is not None:
            node = frame.tb_frame.f_locals.get("node")
            if frame.tb_frame.f_code.co_name == "_translate_fx_graph" and hasattr(node, "format_node"):
                value = node.meta.get("val")
                node_info = dict(name=node.name, target=str(node.target),
                                 expression=node.format_node(),
                                 expected_dtype=str(getattr(value, "dtype", "tuple/non-tensor")),
                                 importer_file=frame.tb_frame.f_code.co_filename,
                                 importer_line=frame.tb_lineno)
            frame = frame.tb_next
        root = _root() / "source/python/tvm/relax/frontend/torch"
        sources = [root / name for name in
                   ("exported_program_translator.py", "base_fx_graph_translator.py")]
        wrapped = TvmBaselineError(str(error), "compile")
        wrapped.diagnostics = dict(
            failure_stage="from_exported_program", precision_conversion=dtype,
            aten_failure_node=node_info, importer_sources={str(path): _sha(path) for path in sources},
            static_binding=dict(status="not_reached", requested_batch=batch),
            byoc_function_count=None, scheduled_tir_function_count=None,
            cuda_source_count=None, byoc_mac_flop_fraction=None,
            tuning=dict(maximum_trials=2000, committed_trials=0, successful_trials=0,
                        reason="import failed before tuning"))
        raise wrapped from error


def _static_batch(mod, batch, input_count):
    import tvm
    from tvm import relax
    bindings = {}
    for parameter in list(mod["main"].params)[:input_count]:
        shape = parameter.ty.shape.values
        if not shape:
            continue
        first = shape[0]
        if isinstance(first, tvm.tirx.IntImm):
            if int(first) != batch:
                raise ValueError(f"export has fixed batch {int(first)}, requested {batch}")
        elif isinstance(first, tvm.tirx.Var):
            bindings[first] = batch
        else:
            raise ValueError(f"unsupported non-variable batch expression {first}")
    if bindings:
        mod = relax.transform.BindSymbolicVars(bindings, func_name="main")(mod)
    checked = [_shape(parameter) for parameter in list(mod["main"].params)[:input_count]]
    if any(shape and shape[0] != batch for shape in checked):
        raise ValueError("Relax batch did not bind statically")
    return mod, dict(bindings={str(key): value for key, value in bindings.items()},
                     input_shapes=checked, batch=batch, reexported=False)


def _partition_diagnostics(mod):
    import tvm
    from tvm import relax
    byoc = {}
    for name, function in mod.functions_items():
        if isinstance(function, relax.Function):
            backend = function.attrs.get("Codegen") if function.attrs else None
            if backend:
                byoc[name.name_hint] = str(backend)
    unknown = Counter()
    calls = Counter()
    weighted = Counter()

    def visit_function(function, backend=None, stack=()):
        env = {}

        def visit(value):
            if isinstance(value, relax.SeqExpr):
                for block in value.blocks:
                    for binding in block.bindings:
                        if isinstance(binding.value, relax.Function):
                            env[binding.var] = binding.value
                        else:
                            visit(binding.value)
                visit(value.body)
            elif isinstance(value, relax.Call):
                operation = value.op
                if isinstance(operation, tvm.ir.GlobalVar):
                    name = operation.name_hint
                    if name in stack:
                        raise ValueError("recursive Relax graph in FLOP diagnostics")
                    child = mod[operation]
                    if isinstance(child, relax.Function):
                        owner = byoc.get(name, backend)
                        calls[name] += 1
                        visit_function(child, owner, (*stack, name))
                elif isinstance(operation, relax.Var) and operation in env:
                    visit_function(env[operation], backend, stack)
                elif isinstance(operation, tvm.ir.Op):
                    name = operation.name
                    count = None
                    try:
                        if name == "relax.matmul":
                            count = 2 * math.prod(_shape(value)) * _shape(value.args[0])[-1]
                        elif name == "relax.nn.conv2d":
                            data, kernel = _shape(value.args[0]), _shape(value.args[1])
                            attrs = value.attrs
                            layout, kernel_layout = str(attrs.data_layout), str(attrs.kernel_layout)
                            count = (2 * math.prod(_shape(value)) * data[layout.index("C")]
                                     // int(attrs.groups) * kernel[kernel_layout.index("H")]
                                     * kernel[kernel_layout.index("W")])
                    except Exception as error:
                        unknown[f"{name}: {type(error).__name__}: {error}"] += 1
                    if count is None:
                        unknown[name] += 1
                    else:
                        weighted[backend or "tir"] += count
                else:
                    unknown[f"call:{type(operation).__name__}"] += 1
                for argument in value.args:
                    if isinstance(argument, (relax.Call, relax.Tuple, relax.SeqExpr)):
                        visit(argument)
            elif isinstance(value, relax.Tuple):
                for field in value.fields:
                    visit(field)
            elif isinstance(value, relax.TupleGetItem):
                if isinstance(value.tuple_value, relax.Call):
                    visit(value.tuple_value)
        visit(function.body)

    visit_function(mod["main"], stack=("main",))
    total = sum(weighted.values())
    offloaded = sum(value for name, value in weighted.items() if name != "tir")
    return dict(byoc_function_count=len(byoc), byoc_functions=byoc,
                byoc_function_count_by_backend=dict(Counter(byoc.values())),
                function_execution_counts=dict(calls), counted_mac_flops=dict(weighted),
                byoc_mac_flop_fraction=offloaded / total if total else None,
                byoc_total_flop_fraction=None,
                flop_coverage_status="unknown for full graph; static conv/matmul 2MAC subtotal follows DNN1 appendix convention",
                uncounted_operations=dict(unknown))


def _prepare_module(program, precision, batch, evidence_dir):
    from tvm import relax
    from tvm.relax.backend.cuda.cublas import partition_for_cublas
    from tvm.relax.backend.cuda.cudnn import partition_for_cudnn
    mod, dtype = _import_program(program, precision, batch)
    mod, detached = relax.frontend.detach_params(mod)
    params = detached.get("main", [])
    input_count = int(mod["main"].attrs["num_input"])
    mod, binding = _static_batch(mod, batch, input_count)
    input_names = [value.name for value in list(mod["main"].params)[:input_count]]
    parameter_dtypes = Counter(str(value.dtype) for value in params)
    mod = partition_for_cublas(mod)
    mod = partition_for_cudnn(mod)
    diagnostics = _partition_diagnostics(mod)
    diagnostics.update(precision_conversion=dtype, static_binding=binding,
                       input_names=input_names, parameter_dtypes=dict(parameter_dtypes))
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "partitioned.py").write_text(mod.script())
    mod = relax.transform.RunCodegen()(mod)
    mod = relax.get_pipeline("zero")(mod)
    (evidence_dir / "zero.py").write_text(mod.script())
    return mod, params, input_names, diagnostics


def _count_database(database):
    records = list(database.get_all_tuning_records())
    successful = 0
    for record in records:
        seconds = record.run_secs
        if seconds is not None and len(seconds) and all(float(value) < 1e9 for value in seconds):
            successful += 1
    return dict(committed_trials=len(records), successful_trials=successful,
                failed_trials=len(records) - successful)


@contextlib.contextmanager
def _cpu_builder_environment():
    # LocalBuilder creates and closes its PopenPool synchronously. Hide CUDA
    # before those interpreters import TVM, then restore it before the runner.
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _build_with_explicit_target(mod, target, params):
    from tvm.s_tir.meta_schedule.builder.local_builder import default_build
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("DNN1 LocalBuilder must compile without a visible CUDA device")
    # NVRTC's callback reads Target.current(); passing target as an argument
    # alone makes this release probe a GPU to infer its architecture.
    with target:
        return default_build(mod, target, params)


def _tune_worker(request):
    _guard()
    tvm, _, _ = _load_tvm()
    import torch
    from tvm.s_tir import meta_schedule as ms
    if request.get("bert"):
        import transformers.modeling_outputs  # registers the stored output pytree
    torch.set_num_threads(4)
    program = torch.export.load(request["export"])
    target = tvm.target.Target(_device_target_config(tvm))
    directory = Path(request["tune_dir"])
    started = time.monotonic()
    from tvm.ir.utils import derived_object
    workers = min(8, len(os.sched_getaffinity(0)))
    with _cpu_builder_environment():
        delegate = ms.builder.LocalBuilder(max_workers=workers, f_build=_build_with_explicit_target)
    scheduler = ms.task_scheduler.GradientBased()

    @derived_object
    class JournalBuilder(ms.builder.PyBuilder):
        def build(self, inputs):
            # GradientBased otherwise defers all initial-round commits until
            # every task has run. Retain finished samples before the wall cap.
            for task_id in range(len(scheduler.tasks_)):
                scheduler.touch_task(task_id)
            with _cpu_builder_environment():
                results = delegate.build(inputs)
            with (directory / "builder_results.jsonl").open("a") as log:
                for index, result in enumerate(results):
                    error = str(result.error_msg) if result.error_msg is not None else None
                    log.write(json.dumps(dict(time=time.time(), error=error,
                                              artifact=result.artifact_path)) + "\n")
                    if error and not (directory / "first_build_error.py").exists():
                        (directory / "first_build_error.py").write_text(inputs[index].mod.script())
            return results

    @derived_object
    class JournalMeasurement(ms.measure_callback.PyMeasureCallback):
        def apply(self, scheduler, task_id, candidates, builds, results):
            with (directory / "runner_results.jsonl").open("a") as log:
                for result in results:
                    log.write(json.dumps(dict(time=time.time(), task_id=task_id,
                        error=str(result.error_msg) if result.error_msg is not None else None,
                        run_secs=[float(x) for x in result.run_secs]
                                 if result.run_secs is not None else None)) + "\n")

    with target:
        mod, _, _, _ = _prepare_module(program, request["precision"], request["batch"],
                                      directory / "ir")
        extracted = ms.relax_integration.extract_tasks(mod, target, params={})
        if extracted:
            database = ms.relax_integration.tune_relax(
                mod, params={}, target=target, work_dir=str(directory),
                max_trials_global=request["max_trials_global"], builder=JournalBuilder(),
                task_scheduler=scheduler,
                measure_callbacks=[JournalMeasurement(), *ms.measure_callback.MeasureCallback.create("default")])
        else:
            database = ms.database.JSONDatabase(work_dir=str(directory))
    result = dict(_count_database(database), task_count=len(extracted),
                  elapsed_s=time.monotonic() - started, deadline_hit=False,
                  inflight_trials="unknown; committed trials are authoritative")
    _json(directory / "completed.json", result)


def _run_tuning(export_path, precision, batch, tune_dir, max_trials_global, timeout_s, bert):
    from tvm.s_tir import meta_schedule as ms
    tune_dir = Path(tune_dir)
    tune_dir.mkdir(parents=True, exist_ok=True)
    request = dict(export=str(export_path), precision=precision, batch=batch,
                   tune_dir=str(tune_dir), max_trials_global=max_trials_global, bert=bert)
    _json(tune_dir / "request.json", request)
    command = [sys.executable, str(Path(__file__).resolve()), "--tune-request",
               str(tune_dir / "request.json")]
    started = time.monotonic()
    timed_out = False
    with (tune_dir / "tuning.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = process.wait()
    elapsed = time.monotonic() - started
    if not timed_out and code:
        error_path = tune_dir / "error.txt"
        raw = error_path.read_text() if error_path.exists() else (tune_dir / "tuning.log").read_text()
        raise TvmBaselineError(f"TVM MetaSchedule worker exited {code}:\n{raw}", "tune")
    try:
        database = ms.database.JSONDatabase(work_dir=str(tune_dir), allow_missing=False)
    except Exception as error:
        raise TvmBaselineError(f"cannot recover tuning database: {error}", "tune") from error
    result = dict(_count_database(database), elapsed_s=elapsed, deadline_hit=timed_out,
                  maximum_trials=max_trials_global, timeout_s=timeout_s,
                  process_returncode=code,
                  inflight_trials="unknown at timeout" if timed_out else 0,
                  work_dir=str(tune_dir), log=str(tune_dir / "tuning.log"))
    if (tune_dir / "builder_results.jsonl").exists():
        builds = [json.loads(line) for line in (tune_dir / "builder_results.jsonl").read_text().splitlines()]
        result.update(attempted_builds=len(builds), build_errors=sum(bool(x["error"]) for x in builds),
                      builder_journal=str(tune_dir / "builder_results.jsonl"))
    if (tune_dir / "completed.json").exists():
        result["task_count"] = json.loads((tune_dir / "completed.json").read_text())["task_count"]
    _json(tune_dir / "receipt.json", result)
    return database, result


def _source_diagnostics(executable, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stack, visited = [executable.mod], set()
    modules, errors, cuda_sources, ptx_sources = [], [], [], []
    while stack:
        module = stack.pop()
        if module in visited:
            continue
        visited.add(module)
        index, kind = len(modules), str(module.kind)
        row = dict(index=index, kind=kind)
        for format_name, extension, collection in (("cuda", "cu", cuda_sources),
                                                   ("ptx", "ptx", ptx_sources)):
            if kind != "cuda":
                continue
            try:
                text = module.inspect_source(format_name)
                if text:
                    path = directory / f"module_{index}.{extension}"
                    path.write_text(text)
                    collection.append(text)
                    row[format_name] = dict(path=str(path), sha256=_sha(path))
            except Exception as error:
                errors.append(dict(module=index, format=format_name, error=str(error)))
        modules.append(row)
        stack.extend(module.imports)
    cuda, ptx = "\n".join(cuda_sources), "\n".join(ptx_sources)
    return dict(runtime_modules=modules, cuda_source_count=len(cuda_sources),
                cuda_wmma=("wmma" in cuda) if cuda_sources else None,
                cuda_mma_sync=("mma.sync" in cuda) if cuda_sources else None,
                ptx_wmma=("wmma" in ptx) if ptx_sources else None,
                ptx_mma_sync=("mma.sync" in ptx) if ptx_sources else None,
                cublas_offload_runtime=any("cublas" in row["kind"] for row in modules),
                cudnn_offload_runtime=any("cudnn" in row["kind"] for row in modules),
                source_extraction_errors=errors,
                external_library_tensor_core_instructions="unknown; generated TIR sources do not expose library kernels")


def _flatten_tvm(value):
    import torch
    if hasattr(value, "__dlpack__"):
        return [torch.from_dlpack(value)]
    if isinstance(value, (tuple, list)) or (hasattr(value, "__len__") and hasattr(value, "__getitem__")):
        result = []
        for item in value:
            result.extend(_flatten_tvm(item))
        return result
    if value is None:
        return []
    raise TypeError(f"unexpected TVM output {type(value).__name__}")


def tvm_runner(directory, program, args, kwargs, precision, batch, out, metadata, *,
               work_dir, max_trials_global=2000, timeout_s=1800):
    """Compile/tune once per identified cell, then return the VM forward callable."""
    _guard()
    if max_trials_global != 2000 or timeout_s != 1800:
        raise ValueError("DNN1 fixes TVM tuning to 2000 global trials and 1800 seconds")
    import torch
    directory, out, work_dir = map(Path, (directory, out, work_dir))
    out.mkdir(parents=True, exist_ok=True)
    diag = dict(status="initializing", precision=precision, batch=batch, recipe=RECIPE,
                byoc_total_flop_fraction=None, error=None)
    metadata["tvm_diag"] = diag

    def publish():
        _json(out / "tvm_diag.json", diag)

    phase = "tvm_cuda_build_or_selfcheck"
    try:
        tvm, tvm_ffi, info = _load_tvm()
        manifest = json.loads((directory / "manifest.json").read_text())
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_uuid = str(getattr(properties, "uuid", ""))
        if not gpu_uuid:
            raise TvmBaselineError("torch did not expose the selected GPU UUID",
                                   "tvm_cuda_build_or_selfcheck")
        libraries = _libraries()
        dependency_versions = {name: importlib.metadata.version(name) for name in
                               ("torch", "numpy", "apache-tvm-ffi", "xgboost", "ml_dtypes")}
        identity = dict(schema_version=1, precision=precision, batch=batch,
                        target_configuration=_device_target_config(tvm),
                        exported_program_sha256=_sha(directory / "exported_program.pt2"),
                        source_checkpoint_sha256=_sha(directory / "checkpoint/model.safetensors"),
                        adapter_sha256=_sha(__file__), recipe=RECIPE,
                        tvm_commit=TVM_COMMIT, tvm_version=tvm.__version__,
                        tvm_source_identity=_source_identity(),
                        dependency_versions=dependency_versions,
                        libraries={row["path"]: row["sha256"] for row in libraries},
                        gpu_uuid=gpu_uuid, compute_capability=[properties.major, properties.minor])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cell = f"{manifest['model']}_b{batch}_{precision}"
        entry = work_dir.resolve() / cell / key
        entry.mkdir(parents=True, exist_ok=True)
        diag.update(tvm_version=tvm.__version__, tvm_commit=TVM_COMMIT,
                    dependency_versions=dependency_versions,
                    libinfo=info, libraries=libraries, cache_key=key,
                    cache_directory=str(entry), gpu_uuid=gpu_uuid,
                    cache_identity=identity)
        publish()
        phase = "compile"
        from tvm import relax
        from tvm.s_tir import dlight, meta_schedule as ms
        target = tvm.target.Target(identity["target_configuration"])
        with (entry / "cache.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt_path, library_path = entry / "compiled.json", entry / "model.so"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["identity"] != identity or _sha(library_path) != receipt["model_sha256"]:
                    raise ValueError("TVM compiled cache identity or library hash differs")
                diag.update(receipt["diagnostics"], cache_hit=True)
                library = tvm.runtime.load_module(str(library_path))
                with target:
                    _, params, input_names, current = _prepare_module(
                        program, precision, batch, out / "cache_hit_ir")
                if current["static_binding"] != diag["static_binding"]:
                    raise ValueError("TVM cache static binding differs")
            else:
                if (entry / "attempt.json").exists():
                    raise ValueError("incomplete or failed TVM cache attempt; explicit diagnosis required before retry")
                _json(entry / "attempt.json", dict(identity=identity, pid=os.getpid(), state="started"))
                with target:
                    mod, params, input_names, preparation = _prepare_module(
                        program, precision, batch, entry / "ir")
                diag.update(preparation)
                publish()
                phase = "tune"
                database, tuning = _run_tuning(
                    directory / "exported_program.pt2", precision, batch, entry / "tune",
                    max_trials_global, timeout_s, manifest["model"].startswith("bert"))
                diag["tuning"] = tuning
                phase = "compile"
                with target, database:
                    mod = relax.transform.MetaScheduleApplyDatabase()(mod)
                    mod = dlight.ApplyDefaultSchedule(
                        dlight.gpu.Matmul(), dlight.gpu.GEMV(), dlight.gpu.Reduction(),
                        dlight.gpu.GeneralReduction(), dlight.gpu.Fallback())(mod)
                    primfuncs = [function for _, function in mod.functions_items()
                                 if isinstance(function, tvm.tirx.PrimFunc)]
                    diag["tir_function_count"] = len(primfuncs)
                    diag["scheduled_tir_function_count"] = sum(
                        bool(function.attrs.get("tirx.is_scheduled", False)) for function in primfuncs)
                    (entry / "ir/final.py").write_text(mod.script())
                    executable = tvm.compile(mod, target=target)
                diag.update(_source_diagnostics(executable, entry / "sources"))
                (entry / "export").mkdir(exist_ok=True)
                executable.export_library(str(library_path), workspace_dir=str(entry / "export"))
                library = tvm.runtime.load_module(str(library_path))
                diag.update(cache_hit=False, model_library=str(library_path),
                            model_sha256=_sha(library_path), status="compiled")
                _json(receipt_path, dict(identity=identity, model_sha256=_sha(library_path),
                                         diagnostics=diag))
        phase = "runtime"
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        flat = torch.utils._pytree.tree_flatten((args, kwargs))[0]
        expected_inputs = [spec.arg.name for spec in program.graph_signature.input_specs
                           if spec.kind.name == "USER_INPUT"]
        if len(flat) != len(expected_inputs):
            raise ValueError("TVM export input signature and fixed inputs disagree")
        ordered = dict(zip(expected_inputs, flat))
        input_values = []
        keepalive = []
        with tvm_ffi.use_torch_stream():
            for name in input_names:
                value = ordered[name]
                value = value.to(device="cuda", dtype=dtype if value.is_floating_point() else value.dtype).contiguous()
                keepalive.append(value)
                input_values.append(tvm.runtime.from_dlpack(value))
            for parameter in params:
                value = torch.from_dlpack(parameter).to(device="cuda").contiguous()
                keepalive.append(value)
                input_values.append(tvm.runtime.from_dlpack(value))
            vm = relax.VirtualMachine(library, tvm.cuda(0))
            vm.save_function("main", "dnn1_forward", *input_values, include_return=True)
        forward = vm["dnn1_forward"]
        diag.update(status="ready", runtime_input_count=len(input_names),
                    runtime_parameter_count=len(params),
                    dtype=str(dtype), stream="tvm_ffi.use_torch_stream around every VM forward")
        diag["library_sha256"] = diag["model_sha256"]
        diag["compiled_library_sha256"] = diag["model_sha256"]
        diag["library_path"] = diag["model_library"]
        metadata["library_sha256"] = diag["model_sha256"]
        metadata["library_path"] = diag["model_library"]
        publish()

        def run():
            # The VM, TVM arguments and torch owners survive the returned closure.
            _ = vm, input_values, keepalive
            try:
                with tvm_ffi.use_torch_stream():
                    return _flatten_tvm(forward())
            except Exception as error:
                raise TvmBaselineError(str(error), "runtime") from error
        return run
    except Exception as error:
        failure_kind = getattr(error, "failure_kind", phase)
        diag.update(getattr(error, "diagnostics", {}))
        diag.update(status="failed", failure_kind=failure_kind,
                    error=traceback.format_exc())
        publish()
        if isinstance(error, TvmBaselineError):
            raise
        raise TvmBaselineError(str(error), failure_kind) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-request", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--out", type=Path)
    options = parser.parse_args()
    if options.selfcheck:
        print(json.dumps(tvm_selfcheck(options.out), sort_keys=True), flush=True)
    elif options.tune_request:
        request = json.loads(options.tune_request.read_text())
        try:
            _tune_worker(request)
        except Exception:
            path = Path(request["tune_dir"]) / "error.txt"
            path.write_text(traceback.format_exc())
            raise
    else:
        parser.error("select --selfcheck or --tune-request")


if __name__ == "__main__":
    main()
