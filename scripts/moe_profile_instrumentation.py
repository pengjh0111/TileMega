"""Diagnostic-only hooks: retain actual selected experts on device until export."""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

STATE = dict(armed=False, current=None, counters={}, routes=[], gemms=[], selected=tuple(range(48)))
NVTX_HANDLES = []
NVTX_MODULES = []
NVTX_ACTIVE = True


def set_nvtx_active(active, model=None):
    global NVTX_ACTIVE
    NVTX_ACTIVE = bool(active)


def source_of(function):
    function = inspect.unwrap(function)
    path = Path(inspect.getsourcefile(function))
    return dict(path=str(path), line=inspect.getsourcelines(function)[1],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def install_helper_nvtx(framework):
    import torch

    def decorate(original, name, stages, choose=None):
        proof = source_of(original)
        labels = {stage: f'moe_profile/{stage}/{name}' for stage in stages}
        for stage, label in labels.items():
            NVTX_MODULES.append(dict(name=name, cls='source_function', stage=stage,
                                     label=label, source_evidence=proof))

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not NVTX_ACTIVE:
                return original(*args, **kwargs)
            stage = choose(args, kwargs) if choose else stages[0]
            tensors = [dict(shape=list(value.shape), dtype=str(value.dtype))
                       for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
            torch.cuda.nvtx.range_push(labels[stage] + '|inputs=' + json.dumps(tensors, separators=(',', ':')))
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        return wrapped

    if framework == 'trtllm':
        from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
        RenormalizeMoeRoutingMethod.apply = decorate(RenormalizeMoeRoutingMethod.apply,
            'RenormalizeMoeRoutingMethod.apply', ['router_topk_normalize'])
        return
    if framework == 'vllm':
        import vllm.model_executor.layers.fused_moe.fused_moe as fused
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        FusedMoE.select_experts = staticmethod(decorate(FusedMoE.select_experts,
            'FusedMoE.select_experts', ['router_topk_normalize']))
        targets = [(fused, 'moe_align_block_size', 'route_count_index_pad'),
                   (ops, 'moe_sum', 'combine')]
    else:
        import sglang.srt.layers.moe.fused_moe_triton.fused_moe as fused
        targets = [(fused, 'moe_align_block_size', 'route_count_index_pad'),
                   (fused, 'silu_and_mul', 'expert_activation'),
                   (fused, 'moe_sum_reduce', 'combine'),
                   (fused, 'moe_sum_reduce_torch_compile', 'combine'),
                   (fused, 'moe_sum_reduce_triton', 'combine')]
    functions = []
    for module, name, stage in targets:
        original = getattr(module, name, None)
        if original is None or not inspect.isfunction(original):
            continue
        functions.append((original, decorate(original, module.__name__ + '.' + name, [stage])))

    def gemm_stage(args, kwargs):
        weights = kwargs.get('B', args[1] if len(args) > 1 else None)
        shape = tuple(weights.shape[-2:])
        return ('expert_gate_up_gemm' if shape == (1536, 2048) else
                'expert_down_gemm' if shape == (2048, 768) else 'expert_gemm_other')

    original = fused.invoke_fused_moe_kernel
    functions.append((original, decorate(original, fused.__name__ + '.invoke_fused_moe_kernel',
        ['expert_gate_up_gemm', 'expert_down_gemm', 'expert_gemm_other'], gemm_stage)))
    for module in list(sys.modules.values()):
        if module and getattr(module, '__name__', '').startswith(framework + '.'):
            for name, value in list(vars(module).items()):
                for original, replacement in functions:
                    if value is original:
                        setattr(module, name, replacement)


def install_nvtx_vllm(model):
    import torch
    def wrap(module, name, stage):
        original = module.forward
        label = f'moe_profile/{stage}/{name}'
        proof = source_of(original)
        NVTX_MODULES.append(dict(name=name, cls=type(module).__name__, stage=stage,
                                 label=label, source_evidence=proof))

        @functools.wraps(original)
        def forward(*args, **kwargs):
            if not NVTX_ACTIVE:
                return original(*args, **kwargs)
            shapes = [dict(shape=list(value.shape), dtype=str(value.dtype))
                      for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
            torch.cuda.nvtx.range_push(label + '|inputs=' + json.dumps(shapes, separators=(',', ':')))
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        module.forward = forward

    for name, module in model.named_modules():
        stage = None
        if name.endswith('qkv_proj'):
            stage = 'qkv_projection'
        elif name.endswith('o_proj'):
            stage = 'output_projection'
        elif name.endswith('.attn') and type(module).__name__ in ('Attention', 'RadixAttention'):
            stage = 'attention_and_kv'
        elif type(module).__name__ == 'RMSNorm':
            stage = 'norm_and_residual'
        elif name.endswith('.mlp.gate'):
            stage = 'router_projection'
        elif name.endswith('.mlp.experts'):
            stage = 'fused_routing_experts_combine'
        elif name.endswith('.mlp.topk'):
            stage = 'router_topk_normalize'
        elif name.endswith('rotary_emb'):
            stage = 'rotary_embedding'
        elif name.endswith('embed_tokens'):
            stage = 'embedding'
        elif name.endswith(('logits_processor', 'lm_head')):
            stage = 'lm_head'
        if stage:
            wrap(module, name, stage)
        if type(module).__name__ == 'Qwen3Attention' and hasattr(module, 'attn'):
            wrap(module.attn, name + '.attn', 'attention_and_kv')
    package = type(model).__module__
    install_helper_nvtx('vllm' if package.startswith('vllm.') else
                        'sglang' if package.startswith('sglang.') else 'trtllm')
    return dict(modules=NVTX_MODULES, diagnostic_only=True,
                scope='Audited Qwen3 module source and actual eager NVTX runtime call ranges;'
                      'fused expert region intentionally not split into invented substage times.')


def arm(model=None):
    STATE.update(armed=True, counters={}, routes=[], gemms=[], current=None, replay=None)
    return dict(armed=True, diagnostic_only=True)


def disarm(model=None):
    STATE['armed'] = False


def begin_layer(layer, tokens, phase, metadata):
    counter = STATE['counters'].get(layer, 0)
    if STATE['armed'] and phase == 'decode':
        counter += 1
        STATE['counters'][layer] = counter
    STATE['current'] = dict(layer=layer, tokens=tokens, phase=phase,
                            decode_iteration=counter, metadata=metadata)


def sample_enabled():
    row = STATE['current']
    return bool(STATE['armed'] and row and row['layer'] in STATE['selected']
                and row['phase'] == 'decode' and 32 <= row['decode_iteration'] <= 48)


def record_ids(ids):
    if sample_enabled():
        row = dict(STATE['current'])
        row.update(ids=ids.detach().clone())
        STATE['routes'].append(row)


def install_vllm(model):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    import vllm.model_executor.layers.fused_moe.fused_moe as fused
    proof = dict(router=source_of(FusedMoE.select_experts), gemm=source_of(fused.invoke_fused_moe_kernel))
    handles = []
    for name, module in model.named_modules():
        if type(module).__name__ != 'Qwen3MoeSparseMoeBlock':
            continue
        match = re.search(r'layers\.(\d+)\.', name + '.')
        if not match:
            raise ValueError('cannot establish source layer identity')
        layer = int(match[1])
        if layer == 0:
            STATE['replay_experts'] = module.experts

        def before(module, inputs, layer=layer):
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                metadata = next(iter(metadata.values()))
            query = getattr(metadata, 'max_query_len', None)
            values = dict(max_query_len=query, max_seq_len=getattr(metadata, 'max_seq_len', None))
            sequence = getattr(metadata, 'seq_lens', None)
            next_iteration = STATE['counters'].get(layer, 0) + 1
            if (sequence is not None and STATE['armed'] and query == 1
                    and layer in STATE['selected'] and 32 <= next_iteration <= 48):
                values['seq_lens'] = sequence.detach().clone()
            begin_layer(layer, inputs[0].shape[0], 'decode' if query == 1 else 'prefill_or_unknown', values)

        handles.append(module.register_forward_pre_hook(before))
    original = FusedMoE.select_experts

    @functools.wraps(original)
    def select(*args, **kwargs):
        result = original(*args, **kwargs)
        record_ids(result[1])
        current = STATE['current']
        if sample_enabled() and current['layer'] == 0 and current['decode_iteration'] == 40:
            experts = STATE['replay_experts']
            STATE['replay'] = dict(hidden_states=kwargs['hidden_states'].detach().clone(),
                                   topk_weights=result[0].detach().clone(),
                                   topk_ids=result[1].detach().clone(),
                                   w1=experts.w13_weight.detach(), w2=experts.w2_weight.detach())
        return result

    FusedMoE.select_experts = staticmethod(select)
    invoke = fused.invoke_fused_moe_kernel
    signature = inspect.signature(invoke)

    @functools.wraps(invoke)
    def gemm(*args, **kwargs):
        if sample_enabled():
            values = signature.bind(*args, **kwargs).arguments
            row = dict(STATE['current'])
            row.update(a_shape=list(values['A'].shape), b_shape=list(values['B'].shape),
                       c_shape=list(values['C'].shape), a_stride=list(values['A'].stride()),
                       b_stride=list(values['B'].stride()), config=dict(values['config']),
                       top_k=values['top_k'],
                       sorted_token_ids_elements=values['sorted_token_ids'].numel(),
                       padded=values['num_tokens_post_padded'].detach().clone())
            STATE['gemms'].append(row)
        return invoke(*args, **kwargs)

    # Installed modular expert backends can hold an imported alias of this function.
    for module in list(sys.modules.values()):
        if module and getattr(module, '__name__', '').startswith('vllm.'):
            for name, value in list(vars(module).items()):
                if value is invoke:
                    setattr(module, name, gemm)
    STATE['handles'] = handles
    STATE['proof'] = proof
    return dict(installed=True, layers=len(handles), source_evidence=proof,
                scope='Diagnostic eager run only;actual routing outputs,not reconstructed top-k.')


def install_trtllm():
    from tensorrt_llm._torch.models.modeling_qwen3_moe import Qwen3MoE
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
    forward = Qwen3MoE.forward
    select = RenormalizeMoeRoutingMethod.apply
    STATE['proof'] = dict(router=source_of(select), moe_forward=source_of(forward))

    @functools.wraps(forward)
    def wrapped(module, hidden_states, attn_metadata, *args, **kwargs):
        layer = getattr(module.experts, 'layer_idx', None)
        values = dict(num_contexts=attn_metadata.num_contexts)
        next_iteration = STATE['counters'].get(layer, 0) + 1
        if (STATE['armed'] and layer in STATE['selected'] and attn_metadata.num_contexts == 0
                and 32 <= next_iteration <= 48 and attn_metadata.seq_lens is not None):
            values['query_seq_lens'] = attn_metadata.seq_lens.detach().clone()
            kv = getattr(attn_metadata, 'kv_lens_runtime', None)
            if kv is not None:
                values['kv_seq_lens'] = kv.detach().clone()
        begin_layer(layer, hidden_states.shape[0],
                    'decode' if attn_metadata.num_contexts == 0 else 'prefill', values)
        return forward(module, hidden_states, attn_metadata, *args, **kwargs)

    @functools.wraps(select)
    def routed(module, logits):
        result = select(module, logits)
        record_ids(result[0])
        return result

    Qwen3MoE.forward = wrapped
    RenormalizeMoeRoutingMethod.apply = routed
    return dict(installed=True, source_evidence=STATE['proof'])


def install_sglang(model, forward_metadata):
    import sglang.srt.layers.moe.fused_moe_triton.fused_moe as fused
    for name, module in model.named_modules():
        if type(module).__name__ != 'Qwen3MoeSparseMoeBlock':
            continue
        layer = module.layer_id

        def before(module, args, layer=layer):
            metadata = forward_metadata()
            begin_layer(layer, args[0].shape[0], 'decode' if metadata['is_decode'] else 'prefill',
                        dict(forward_index=metadata['index']))
        module.register_forward_pre_hook(before)
    original = fused.invoke_fused_moe_kernel
    signature = inspect.signature(original)
    STATE['proof'] = dict(gemm=source_of(original),
                          routing='Actual topk_ids consumed by first expert GEMM;independently checked by native stat dump')

    @functools.wraps(original)
    def invoke(*args, **kwargs):
        if sample_enabled():
            values = signature.bind(*args, **kwargs).arguments
            if tuple(values['B'].shape[-2:]) == (1536, 2048):
                record_ids(values['topk_ids'])
            row = dict(STATE['current'])
            row.update(a_shape=list(values['A'].shape), b_shape=list(values['B'].shape),
                       c_shape=list(values['C'].shape), a_stride=list(values['A'].stride()),
                       b_stride=list(values['B'].stride()), config=dict(values['config']),
                       top_k=values['top_k'],
                       sorted_token_ids_elements=values['sorted_token_ids'].numel(),
                       flags={key: values.get(key) for key in ('a_use_tma', 'b_use_tma', 'c_sorted',
                                                               'no_combine', 'mul_routed_weight')},
                       padded=values['num_tokens_post_padded'].detach().clone())
            STATE['gemms'].append(row)
        return original(*args, **kwargs)
    for module in list(sys.modules.values()):
        if module and getattr(module, '__name__', '').startswith('sglang.'):
            for name, value in list(vars(module).items()):
                if value is original:
                    setattr(module, name, invoke)
    return dict(installed=True, source_evidence=STATE['proof'])


def export(path, model=None):
    import torch
    disarm()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # No routing tensor is copied to the host until this post-ROI export.
    counts = [torch.bincount(row['ids'].reshape(-1).long(), minlength=128) for row in STATE['routes']]
    host_counts = torch.stack(counts).cpu().tolist() if counts else []
    routes = []
    for original, count in zip(STATE['routes'], host_counts):
        row = {key: value for key, value in original.items() if key != 'ids'}
        row['metadata'] = {key: value.cpu().tolist() if isinstance(value, torch.Tensor) else value
                           for key, value in row['metadata'].items()}
        row.update(n_e=count, active_experts=sum(value > 0 for value in count),
                   empty_experts=sum(value == 0 for value in count), max_n_e=max(count),
                   max_to_mean=max(count) / (sum(count) / 128),
                   route_assignments=sum(count), top_k=original['ids'].shape[-1])
        if sum(count) != original['ids'].numel():
            raise ValueError('route count does not conserve actual expert assignments')
        routes.append(row)
    gemms = []
    for original in STATE['gemms']:
        row = {key: value for key, value in original.items() if key not in ('padded', 'metadata')}
        row['actual_padded_rows'] = int(original['padded'].cpu().item())
        gemms.append(row)
    result = dict(diagnostic_only=True, route_samples=routes, gemm_samples=gemms,
                  source_evidence=STATE.get('proof'),
                  scope='Actual selected expert IDs retained on device;post-ROI histogram/export. '
                        'Eager diagnostic changes timing and is excluded from benchmark.csv. '
                        'No unobserved tile/padding/materialization values inferred.')
    torch.save([dict(layer=row['layer'], decode_iteration=row['decode_iteration'],
                     expert_ids=row['ids'].cpu()) for row in STATE['routes']], path / 'selected_experts.pt')
    if STATE.get('replay'):
        replay_path = Path('/dev/shm/tilemega-moe-profile-qwen3-ad44e777/diagnostic_replay')
        replay_path.mkdir(exist_ok=True)
        batch = STATE['replay']['hidden_states'].shape[0]
        target = replay_path / f'layer0_b{batch}.pt'
        torch.save({key: value.cpu() for key, value in STATE['replay'].items()}, target)
        result['real_replay'] = dict(path=str(target), layer=0, decode_iteration=40,
                                   scope='Actual pretrained layer weights/input/route. '
                                         'Deleted with checkpoint after validation;not a timed transfer.')
    (path / 'routing.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    return dict(samples=len(routes), gemm_samples=len(gemms), path=str(path / 'routing.json'))
