"""Correctness-only smoke for a generated complete MoE decoder.

The seeded small checkpoint is a unit fixture, never the real-weight G-MOE gate.
Layer outputs expose errors hidden by a final argmax or a small expert update.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

import torch
from tilemega.serving.plan import PlanLibrary, FORWARD
from tilemega.serving.state import rotary_tables
from tilemega.serving.weights import load_weights
from tilemega.moe.check_generated import internal, sha


def individual_experts(state):
    result = {}
    for name, value in state.items():
        if name.endswith('.experts.gate_up_proj'):
            prefix = name.removesuffix('gate_up_proj')
            for expert, matrix in enumerate(value):
                gate, up = matrix.chunk(2, dim=0)
                result[prefix+f'{expert}.gate_proj.weight'] = gate.contiguous()
                result[prefix+f'{expert}.up_proj.weight'] = up.contiguous()
        elif name.endswith('.experts.down_proj'):
            prefix = name.removesuffix('down_proj')
            for expert, matrix in enumerate(value):
                result[prefix+f'{expert}.down_proj.weight'] = matrix.contiguous()
        else:
            result[name] = value.contiguous()
    return result


@torch.inference_mode()
def attention_residual(model, layer_index, hidden, past, history):
    """Independent HF attention on the exact input received by a native layer."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        apply_rotary_pos_emb, eager_attention_forward)
    layer = model.model.layers[layer_index]
    attention = layer.self_attn
    normalized = layer.input_layernorm(hidden)
    shape = (*hidden.shape[:-1], -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(normalized).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
    value = attention.v_proj(normalized).view(shape).transpose(1, 2)
    positions = torch.arange(past, past+hidden.shape[1], device=hidden.device)[None, :]
    cos, sin = model.model.rotary_emb(hidden, positions)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if history is not None:
        key = torch.cat((history[layer_index][0], key), dim=2)
        value = torch.cat((history[layer_index][1], value), dim=2)
    keys = torch.arange(key.shape[2], device=hidden.device)
    mask = torch.zeros((1, 1, hidden.shape[1], key.shape[2]),
                       dtype=hidden.dtype, device=hidden.device)
    mask.masked_fill_(keys[None, :] > positions[:, :, None], float('-inf'))
    context, _ = eager_attention_forward(attention, query, key, value, mask,
                                         scaling=attention.scaling)
    projected = attention.o_proj(context.reshape(*hidden.shape[:-1], -1))
    return (hidden+projected).bfloat16()


@torch.inference_mode()
def check(library_path, config_path, bridge_path=None):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = json.loads(Path(str(library_path)+'.identity.json').read_text())
    if identity['binary_sha256'] != sha(library_path):
        raise ValueError('decoder binary differs from its identity')
    library = PlanLibrary(library_path)
    if library.info.phase == FORWARD:
        raise ValueError('complete decoder requires prefill or decode')
    config = Qwen3MoeConfig.from_json_file(str(config_path))
    if config.num_hidden_layers > 2 or config.hidden_size > 128:
        raise ValueError('synthetic decoder smoke is restricted to a small unit fixture')
    config._experts_implementation = 'grouped_mm'
    config._attn_implementation = 'eager'
    model = Qwen3MoeForCausalLM(config).eval().to(dtype=torch.bfloat16)
    for name, value in model.named_parameters():
        if '.experts.' in name:
            value.normal_(std=.7/value.shape[-1]**.5)
    state = individual_experts(model.state_dict())
    layers = config.num_hidden_layers
    batch, seq, capacity = library.info.batch_lo, library.info.seq, library.info.capacity
    past = library.info.past_lo
    if batch != library.info.batch_hi or past+seq >= capacity:
        raise ValueError('decoder smoke needs one bound batch and room for its output token')
    ids = torch.randint(config.vocab_size, (batch, past+seq), device='cuda')
    model = model.cuda()
    captured, gates, residuals = {}, {}, {}
    hooks = [layer.register_forward_hook(lambda module, args, value, i=i:
        captured.__setitem__(i, value.detach().clone())) for i, layer in enumerate(model.model.layers)]
    hooks += [layer.mlp.gate.register_forward_hook(lambda module, args, value, i=i:
        gates.__setitem__(i, tuple(v.detach().clone() for v in value)))
        for i, layer in enumerate(model.model.layers)]
    hooks += [layer.post_attention_layernorm.register_forward_pre_hook(lambda module, args, i=i:
        residuals.__setitem__(i, args[0].detach().clone()))
        for i, layer in enumerate(model.model.layers)]
    history = prefix = None
    if past:
        prefix = model(ids[:, :past], use_cache=True)
        history = [(item.keys.clone(), item.values.clone()) for item in prefix.past_key_values.layers]
        cache = prefix.past_key_values
    else:
        cache = None
    reference = model(ids[:, past:], past_key_values=cache, use_cache=True)
    expected_token = reference.logits[:, -1].argmax(-1).to(torch.int32)
    expected = [captured[i].clone() for i in range(layers)]
    expected_gates = [gates[i] for i in range(layers)]
    expected_residuals = [residuals[i] for i in range(layers)]
    routing_names=[]
    if bridge_path is not None:
        from tilemega.moe.check_generated import router_output
        bridge=json.loads(Path(bridge_path).read_text())
        calls=[n for n in bridge['nodes'] if 'moe_experts' in n['target']]
        if len(calls)!=layers:
            raise ValueError('decoder bridge must contain one MoE region per layer')
        routing_names=[(router_output(bridge,f'model.layers.{i}.mlp.gate.weight'),call['inputs'][1])
            for i,call in enumerate(calls)]
    for hook in hooks:
        hook.remove()
    cos, sin = rotary_tables(config, capacity, torch.device('cuda'))
    tokens = torch.empty((batch, capacity), dtype=torch.int32, device='cuda')
    kv = torch.empty((layers, 2, batch, config.num_key_value_heads, capacity, config.head_dim),
                     dtype=torch.bfloat16, device='cuda')
    del captured, gates, residuals, reference, prefix, cache
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='dm1-moe-decoder-') as directory:
        path = Path(directory)/'model.safetensors'
        save_file(state, str(path))
        source_sha256 = sha(path)
        del state
        tensors = load_weights(directory, library, device='cuda')
    tensors.update({'serving.tokens':tokens, 'serving.rope_cos':cos, 'serving.rope_sin':sin})
    for layer in range(layers):
        tensors[f'kv_cache.k.{layer}'] = kv[layer, 0]
        tensors[f'kv_cache.v.{layer}'] = kv[layer, 1]
    cases, first = [], None
    with library.create(batch, {name:value.data_ptr() for name,value in tensors.items()}, 0) as plan:
        plan.set_steps([past])
        for epoch in range(1):
            for mode in (1, 2):
                if not library.info.modes & mode:
                    continue
                tokens.fill_(-1)
                tokens[:, :past+seq] = ids.to(torch.int32)
                kv.fill_(float('nan'))
                if history is not None:
                    for layer, (key, value) in enumerate(history):
                        kv[layer, 0, :, :, :past].copy_(key)
                        kv[layer, 1, :, :, :past].copy_(value)
                plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                outputs = [internal(library, plan, f'l{i}.moe.output',
                    (batch,seq,config.hidden_size), torch.bfloat16, batch=batch) for i in range(layers)]
                errors = [(a.float()-b.float()).abs() for a,b in zip(outputs,expected)]
                numerical = all(bool(torch.isfinite(a).all() and
                    torch.all(error <= .016+.016*b.float().abs()))
                    for a,b,error in zip(outputs,expected,errors))
                token = tokens[:, past+seq].clone()
                routing=[]
                components_ok=True
                for i,(logit_name,index_name) in enumerate(routing_names):
                    logits=internal(library,plan,logit_name,(batch,seq,config.num_experts),torch.bfloat16,batch=batch)
                    indices=internal(library,plan,index_name,(batch,seq,config.num_experts_per_tok),torch.int32,batch=batch).long()
                    ref_logits,_,ref_indices=expected_gates[i]
                    ref_logits=ref_logits.reshape_as(logits);ref_indices=ref_indices.reshape_as(indices)
                    same=indices.sort(-1).values.eq(ref_indices.sort(-1).values).all(-1)
                    residual=internal(library,plan,f'l{i}.attn.residual',
                        (batch,seq,config.hidden_size),torch.bfloat16,batch=batch)
                    layer=model.model.layers[i]
                    normalized=layer.post_attention_layernorm(residual)
                    local_logits,_,local_indices=layer.mlp.gate(normalized)
                    local_logits=local_logits.reshape_as(logits)
                    local_indices=local_indices.reshape_as(indices)
                    local_reference=(residual+layer.mlp(normalized)).bfloat16()
                    local_same=indices.sort(-1).values.eq(local_indices.sort(-1).values).all(-1)
                    local_error=(outputs[i].float()-local_reference.float()).abs()
                    local_ok=bool(torch.isfinite(outputs[i]).all() and
                        torch.all((local_error<=.016+.016*local_reference.float().abs())[local_same]))
                    boundary=local_logits.gather(-1,local_indices).float().min(-1).values
                    rounded=boundary.bfloat16()
                    ulp=torch.maximum(*[(torch.nextafter(rounded,torch.full_like(rounded,direction)).float()-boundary).abs()
                        for direction in (float('inf'),float('-inf'))])
                    added=(indices[:,:,:,None]!=local_indices[:,:,None,:]).all(-1)
                    removed=(local_indices[:,:,:,None]!=indices[:,:,None,:]).all(-1)
                    added_gap=(local_logits.float().gather(-1,indices)-boundary[:,:,None]).abs()
                    removed_gap=(local_logits.float().gather(-1,local_indices)-boundary[:,:,None]).abs()
                    route_ok=bool(torch.all((added_gap<=2*ulp[:,:,None])|~added) and
                        torch.all((removed_gap<=2*ulp[:,:,None])|~removed))
                    layer_input=model.model.embed_tokens(ids[:,past:]) if i==0 else outputs[i-1]
                    local_attention=attention_residual(model,i,layer_input,past,history)
                    residual_error=(residual.float()-local_attention.float()).abs()
                    attention_ok=bool(torch.isfinite(residual).all() and
                        torch.all(residual_error<=.016+.016*local_attention.float().abs()))
                    components_ok &= local_ok and route_ok and attention_ok
                    routing.append(dict(layer=i,set_equal_fraction=float(same.float().mean()),
                        router_max_error=float((logits.float()-ref_logits.float()).abs().max()),
                        residual_max_error=float((residual.float()-expected_residuals[i].float()).abs().max()),
                        output_max_error_same_routes=float(errors[i][same].max()) if same.any() else None,
                        different_route_tokens=int((~same).sum()),
                        local_moe_numerical=local_ok,local_routing=route_ok,attention_residual_numerical=attention_ok,
                        local_moe_max_error=float(local_error.max()),
                        local_attention_max_error=float(residual_error.max()),
                        local_set_equal_fraction=float(local_same.float().mean())))
                bits = first is None or all(torch.equal(a,b) for a,b in zip([*outputs,token],first))
                if first is None:
                    first = [value.clone() for value in [*outputs,token]]
                # Discrete expert selection can amplify a small upstream BF16
                # difference. Check each region on its actual native input;
                # retain the independent whole-decoder error as a diagnostic.
                local_token=model.lm_head(model.model.norm(outputs[-1]))[:, -1].argmax(-1).to(torch.int32)
                head_ok=torch.equal(token,local_token)
                cases.append(dict(epoch=epoch, mode=mode, passed=(components_ok and head_ok if routing_names else numerical and
                    torch.equal(token,expected_token)) and bits, layer_max_errors=[float(e.max()) for e in errors],
                    whole_decoder_elementwise=numerical,
                    token_equal=torch.equal(token,expected_token), local_head_token_equal=head_ok, same_binary_bits=bits,routing=routing))
    return dict(evidence='verified', passed=all(row['passed'] for row in cases),
        scope='synthetic complete MoE decoder smoke; not the real-weight G-MOE gate',
        artifact_id=identity['artifact_id'], checkpoint_sha256=source_sha256,
        config_sha256=sha(config_path), bridge_sha256=sha(bridge_path) if bridge_path else None,
        checker_sha256=sha(Path(__file__)), seed=20261010, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.library, args.config, args.bridge)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
