"""Shared static ATen import and NHWC propagation for pinned FX2AIT."""
def install_converters():
    import importlib
    import torch
    from aitemplate.compiler import ops
    from fx2ait.converters.converter_registry import AIT_CONVERTERS
    from tilemega.dnn.aitemplate_batchnorm import batch_norm, hardtanh

    if getattr(install_converters, "installed", False):
        return
    # LayerNorm consults this testing helper during IR construction. The fixed
    # recipe specifies SM90, so populate metadata without probing any device.
    detector = importlib.import_module("aitemplate.testing.detect_target")
    detector.IS_CUDA, detector.FLAG = True, "90"
    AIT_CONVERTERS[torch.ops.aten.batch_norm] = batch_norm
    AIT_CONVERTERS[torch.ops.aten.hardtanh.default] = hardtanh

    def zero_pad(target, args, kwargs, name):
        if any(args[1]):
            raise ValueError("nonzero constant padding requires an explicit converter")
        return args[0]

    def generic_permute(target, args, kwargs, name):
        return ops.permute()(args[0], args[1])

    original_slice = AIT_CONVERTERS[torch.ops.aten.slice.Tensor]

    def default_slice(target, args, kwargs, name):
        defaults = [args[0], 0, 0, 9223372036854775807, 1]
        defaults[:len(args)] = args
        for index, key in enumerate(("self", "dim", "start", "end", "step")):
            if key in kwargs:
                defaults[index] = kwargs[key]
        return original_slice(target, tuple(defaults), {}, name)

    AIT_CONVERTERS[torch.ops.aten.constant_pad_nd.default] = zero_pad
    AIT_CONVERTERS[torch.ops.aten.permute.default] = generic_permute
    AIT_CONVERTERS[torch.ops.aten.slice.Tensor] = default_slice
    from tilemega.dnn import aitemplate_aten  # registers static ATen schemas
    for opname in ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"):
        target = getattr(torch.ops.aten, opname).default
        AIT_CONVERTERS[target] = aitemplate_aten.identity_adaptive_avg_pool(
            AIT_CONVERTERS.get(target))
    from tilemega.dnn.aitemplate_layout import install
    install()
    install_converters.installed = True


def import_program(program, args, kwargs, directory):
    import torch
    from fx2ait.fx2ait import AITInterpreter
    from fx2ait.tensor_spec import TensorSpec
    from fx2ait.passes import lower_basic_pass_aten as passes
    install_converters()
    from tilemega.dnn.aitemplate_layout import COUNTS, nchw
    for key in COUNTS:
        COUNTS[key] = {} if key == "logical_boundary_by_op" else 0
    graph = program.run_decompositions().module()
    graph = passes.replace_aten_op_with_indices(graph).graph_module
    graph = passes.replace_native_layernorm_with_layernorm(graph).graph_module
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.sym_size.int:
            node.target = torch.ops.aten.sym_size
    graph = passes.remove_ops(graph).graph_module
    graph.graph.lint()
    graph.recompile()
    flat_inputs = torch.utils._pytree.tree_flatten((args, kwargs))[0]
    specs = [TensorSpec(list(value.shape), value.dtype) for value in flat_inputs]

    class StaticInterpreter(AITInterpreter):
        def _create_target(self):
            # A target is only needed at code generation, never for graph import.
            return None

    interpreter = StaticInterpreter(graph, specs, str(directory), "dnn1",
                                    keep_constants=True, use_fp16_acc=False)
    physical = torch.utils._pytree.tree_flatten(torch.fx.Interpreter.run(interpreter))[0]
    outputs = nchw(physical)
    # Layout conversion creates a new output tensor; don't expose its NHWC parent.
    for value in physical:
        value._attrs["is_output"] = False
        if not any(value is output for output in outputs):
            value._attrs["name"] = None
    for index, value in enumerate(outputs):
        value._attrs["is_output"] = True
        value._attrs["name"] = f"output_{index}"
    if len(interpreter._fx_input_names) != len(flat_inputs):
        raise ValueError("FX2AIT input order differs from the exported program")
    return outputs, dict(zip(interpreter._fx_input_names, flat_inputs)), dict(COUNTS)
