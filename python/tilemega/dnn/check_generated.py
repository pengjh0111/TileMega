"""Execute a graph-generated DNN library against its upstream exported graph.

This smoke check is separate from the dataset-level G-DNN gate. The FP32
oracle uses the exported BF16 checkpoint promoted to FP32, and is identified
as such in its receipt. No latency measurement or compilation occurs here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import importlib
import ctypes as C
from pathlib import Path

import torch

from tilemega.serving.plan import FORWARD, PlanLibrary
from tilemega.serving.weights import load_weights


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def _diagnose(library, plan, module, arguments, bridge, destination):
    """Copy test-only native intermediates and compare untouched FX values.

    The optional accessor is appended to a separate diagnostic CUDA artifact;
    it is not part of the serving ABI or normal generated model libraries.
    A diagnostic does not bypass the final output correctness assertion.
    """
    captured = {}
    class Capture(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if isinstance(value, torch.Tensor):
                captured[node.name] = value.detach().clone()
            return value
    Capture(module).run(*arguments)
    aliases, terminal = {}, {}
    names = {buffer.name for buffer in library.buffers}
    order = {node['name']: index for index,node in enumerate(bridge['nodes'])}
    unary = {'batch_norm', '_native_batch_norm_legit_no_training', 'relu', 'relu_',
             'hardtanh', 'hardtanh_', 'view', 'reshape', 'flatten', 'dropout',
             'clone', 'contiguous', 'tanh', 'gelu', 'silu'}
    for node in bridge['nodes']:
        name = node['name']; op = node['target'].split('.')[1] if node['target'].startswith('aten.') else ''
        if name in names:
            aliases[name] = name
        elif (op in unary or node['target'] == 'operator.getitem') and node['inputs']:
            source = node['inputs'][0]
            if source in aliases:
                aliases[name] = aliases[source]
        elif op in {'add','add_'}:
            sources = [aliases[source] for source in node['inputs'] if source in aliases]
            if sources:
                aliases[name] = max(sources, key=lambda source:order.get(source,-1))
        if name in aliases and name in captured:
            terminal[aliases[name]] = name
    accessor = library.lib.tm_dm_debug_buffer
    accessor.argtypes = [C.c_void_p,C.c_uint32,C.POINTER(C.c_uint64)]
    accessor.restype = C.c_void_p
    runtime = C.CDLL('/usr/local/cuda/lib64/libcudart.so')
    runtime.cudaMemcpy.argtypes = [C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
    runtime.cudaMemcpy.restype = C.c_int
    rows = []
    for index,buffer in enumerate(library.buffers):
        reference_name = terminal.get(buffer.name)
        if reference_name is None or buffer.dtype != 0:
            continue
        meta = (C.c_uint64*18)(); pointer = accessor(plan.handle,index,meta)
        kind,rank = meta[:2]; logical,physical = list(meta[2:6]),list(meta[6:10])
        count = buffer.elements_constant+int(logical[0])*buffer.elements_per_batch
        if not pointer or not rank or rank>4:
            continue
        # Layout metadata gives the physical allocation. External buffers may
        # be described independently of the internal tensor's per-batch size.
        count = 1
        for extent in physical[:rank]: count *= extent
        value = torch.empty(count,device='cuda',dtype=torch.bfloat16)
        status = runtime.cudaMemcpy(value.data_ptr(),pointer,value.numel()*2,3)
        if status:
            raise RuntimeError(f'diagnostic buffer copy failed: {status}')
        value=value.reshape(physical[:rank])
        if kind == 1:
            top,_,left,_ = meta[14:18]
            value=value[:,top:top+logical[1],left:left+logical[2],:logical[3]].permute(0,3,1,2)
        reference=captured[reference_name].float().to('cuda')
        # The body stores [batch, query, head, dim]; ATen SDPA returns
        # [batch, head, query, dim]. This is a layout comparison, not a reshape.
        if nodes_target := next((node['target'] for node in bridge['nodes']
                if node['name']==reference_name), None):
            if nodes_target=='aten.scaled_dot_product_attention.default' and reference.ndim==4:
                reference=reference.permute(0,2,1,3).contiguous()
        if value.numel()!=reference.numel():
            rows.append(dict(buffer=buffer.name,reference=reference_name,shape_mismatch=True,
                actual_shape=list(value.shape),reference_shape=list(reference.shape)))
            continue
        value=value.reshape_as(reference).float()
        cosine=torch.nn.functional.cosine_similarity(value.flatten(1),reference.flatten(1),dim=1)
        rows.append(dict(buffer=buffer.name,reference=reference_name,cosine_min=cosine.min().item(),
            token_cosine_min=torch.nn.functional.cosine_similarity(value,reference,dim=-1).min().item(),
            max_error=(value-reference).abs().max().item(),reference_max=reference.abs().max().item()))
    destination.write_text(json.dumps(dict(evidence='verified',scope='diagnostic only',buffers=rows),indent=2)+'\n')


def output_metrics(actual,fp32,bf16,*,restoration=False,elementwise=False):
    cosine=torch.nn.functional.cosine_similarity(
        actual.reshape(-1,actual.shape[-1]),fp32.reshape(-1,fp32.shape[-1]),dim=1)
    metrics=dict(cosine_min=cosine.min().item(),max_error=(actual-fp32).abs().max().item(),
        bf16_max_error=(actual-bf16.float()).abs().max().item())
    if elementwise:
        if not torch.all((actual-fp32).abs()<=.016+.016*fp32.abs()):
            raise AssertionError(f'elementwise error exceeds BF16 tolerance: {metrics["max_error"]}')
    elif restoration:
        from .check import psnr
        tm_psnr=psnr(actual,fp32);bf16_psnr=psnr(bf16,fp32)
        metrics.update(psnr_fp32=tm_psnr.tolist(),bf16_psnr_fp32=bf16_psnr.tolist())
        if not torch.all(tm_psnr>=bf16_psnr-1):
            raise AssertionError(f'NAFNet PSNR with FP32 {tm_psnr.tolist()} is below BF16 - 1 dB {bf16_psnr.tolist()}')
    elif cosine.min()<.999:
        raise AssertionError(f'cosine with FP32 oracle is {cosine.min().item()}')
    return metrics


def check(library_path, export_dir, bridge_path, batch, *, epochs=1, diagnostics=None,
          input_tensors=None,elementwise=False):
    torch.set_num_threads(4); torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32 = False
    export_dir, bridge_path = Path(export_dir), Path(bridge_path)
    restoration=json.loads((export_dir/'manifest.json').read_text())['model']=='nafnet'
    library = PlanLibrary(library_path)
    if library.info.phase != FORWARD or library.info.capacity != 0:
        raise ValueError('DNN correctness requires a stateless forward plan')
    bridge = json.loads(bridge_path.read_text())
    signature = bridge['signature']
    nodes = {node['name']: node for node in bridge['nodes']}
    user_inputs = [item for item in signature['inputs'] if item['kind'] == 'USER_INPUT']
    if any(nodes[item['name']]['dtype'] == 'torch.int64' for item in user_inputs):
        importlib.import_module('transformers.modeling_outputs')
    program = torch.export.load(export_dir/'exported_program.pt2')
    # CPU exports may contain literal CPU devices in shape-only mask graphs.
    # Keep the untouched exported reference on its original device.
    module = program.module()
    oracle = copy.deepcopy(module).float()
    tensors = load_weights(export_dir/'checkpoint', library, device='cuda')
    arguments = []
    image_input = False
    input_uses = {item['name']: [node for node in bridge['nodes']
        if item['name'] in node['inputs']] for item in user_inputs}
    supplied = None
    if input_tensors is not None:
        from safetensors.torch import load_file
        supplied = load_file(str(input_tensors), device='cpu')
        if set(supplied) != {item['name'] for item in user_inputs}:
            raise ValueError('diagnostic input names differ from the exported signature')
    for item in user_inputs:
        record = nodes[item['name']]
        shape = (batch, *(int(value) for value in record['shape'][1:]))
        if record['dtype'] == 'torch.bfloat16' and len(shape) == 4:
            value = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
            image_input = True
        elif record['dtype'] == 'torch.int64' and len(shape) == 2:
            embedding = [node for node in input_uses[item['name']]
                if node['target'] == 'aten.embedding.default']
            if embedding:
                if len(embedding) != 1:
                    raise ValueError('integer input has ambiguous embedding tables')
                weight = embedding[0]['args']['v'][0]['v']
                value = torch.randint(int(nodes[weight]['shape'][0]), shape, device='cuda')
            else:
                lengths = torch.randint(64, shape[1]+1, (batch,), device='cuda')
                value = (torch.arange(shape[1], device='cuda')[None, :] < lengths[:, None]).long()
        else:
            raise ValueError('DNN smoke entry needs BF16 NCHW images or int64 encoder inputs')
        if supplied is not None:
            original = supplied[item['name']]
            if original.shape != value.shape or original.dtype != value.dtype:
                raise ValueError('diagnostic input shape or dtype differs from the plan')
            value = original.cuda()
        tensors[item['name']] = value; arguments.append(value)
    # Export preserves the original positional/keyword pytree contract.
    # HF BERT's supplied keyword inputs must not become positional inputs.
    cpu_arguments=[value.cpu() for value in arguments]
    positional,keywords=torch.utils._pytree.tree_unflatten(cpu_arguments,program.call_spec.in_spec)
    expected_bf16 = module(*positional,**keywords)
    fp32_arguments=[value.float() if value.is_floating_point() else value for value in cpu_arguments]
    positional,keywords=torch.utils._pytree.tree_unflatten(fp32_arguments,program.call_spec.in_spec)
    expected_fp32 = oracle(*positional,**keywords)
    def leaves(value):
        return [item for item in torch.utils._pytree.tree_flatten(value)[0] if isinstance(item, torch.Tensor)]
    expected_bf16, expected_fp32 = [[item.to('cuda') for item in leaves(value)]
        for value in (expected_bf16,expected_fp32)]
    outputs = [item['name'] for item in signature['outputs'] if item['kind'] == 'USER_OUTPUT']
    if len(outputs) != len(expected_fp32):
        raise ValueError('exported output tree differs from its bridge signature')
    descriptors = {buffer.name:buffer for buffer in library.buffers}
    for name, reference in zip(outputs, expected_fp32):
        desc = descriptors[name]
        if desc.dtype != 0 or desc.elements_constant+batch*desc.elements_per_batch != reference.numel():
            raise ValueError('generated output allocation differs from the upstream output')
        tensors[name] = torch.empty_like(reference, dtype=torch.bfloat16)
    external = {buffer.name for buffer in library.buffers if buffer.role == 1}
    if external != set(tensors):
        raise ValueError('generated external buffers differ from checkpoint/input/output bindings')
    first = None; cases = []
    with library.create(batch, {name:value.data_ptr() for name,value in tensors.items()}, 0) as plan:
        plan.set_steps([0])
        for epoch in range(epochs):
            for mode in (1, 2):
                if not library.info.modes & mode:
                    continue
                for name in outputs:
                    tensors[name].fill_(float('nan'))
                plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                if diagnostics is not None and epoch == 0 and mode == 1:
                    _diagnose(library,plan,oracle,fp32_arguments,bridge,diagnostics)
                row = dict(epoch=epoch, mode=mode, outputs=[])
                for name, fp32, bf16 in zip(outputs, expected_fp32, expected_bf16):
                    actual = tensors[name].float()
                    if not torch.isfinite(actual).all():
                        raise AssertionError('generated DNN produced a nonfinite output')
                    metrics=dict(name=name,**output_metrics(actual,fp32,bf16,restoration=restoration,elementwise=elementwise))
                    if image_input and actual.ndim == 2:
                        metrics['top1_agreement'] = (actual.argmax(-1)==fp32.argmax(-1)).float().mean().item()
                    row['outputs'].append(metrics)
                if first is None:
                    first = [tensors[name].clone() for name in outputs]
                elif any(not torch.equal(value, tensors[name]) for name,value in zip(outputs,first)):
                    raise AssertionError('DNN executors or repeated launches differ bitwise')
                cases.append(row)
    identity_path = Path(str(library.path)+'.identity.json')
    identity = json.loads(identity_path.read_text())
    if identity['binary_sha256'] != _sha(library.path):
        raise ValueError('generated library no longer matches its build identity')
    return dict(evidence='verified', passed=True, scope='upstream DNN graph execution smoke; not G-DNN',
        reference='exported BF16 checkpoint promoted to FP32', batch=batch,
        criterion='elementwise BF16 tolerance' if elementwise else 'model output metric',
        checker_sha256=_sha(Path(__file__)),
        artifact_id=identity['artifact_id'], bridge_sha256=_sha(bridge_path),
        archive_sha256=_sha(export_dir/'exported_program.pt2'),
        input_tensors_sha256=_sha(input_tensors) if input_tensors is not None else None, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--elementwise',action='store_true',
        help='operator fixture gate: |error| <= .016 + .016*|FP32|')
    parser.add_argument('--diagnostics',type=Path,
        help='test-only intermediate receipt; requires the separate diagnostic accessor')
    parser.add_argument('--input-tensors',type=Path,
        help='identified safetensors inputs for an additional smoke diagnostic')
    args = parser.parse_args()
    result = check(args.library, args.export, args.bridge, args.batch,diagnostics=args.diagnostics,
                   input_tensors=args.input_tensors,elementwise=args.elementwise)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
