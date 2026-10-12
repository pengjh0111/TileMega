"""Manifest allocation accounting shared by dense and MoE serving."""
import json


def memory_report(config, manifests, batch, capacity):
    """Account for shared recipes, request state and both plans' buffers.

    CUDA context, event/queue tables and runtime-inserted split-K storage
    are not manifest buffers. Content-equal weights can also be deduplicated
    by the loader. This estimate cannot certify that a plan fits.
    """
    widths = {'bf16': 2, 'f32': 4, 'i32': 4, 'i64': 8}
    weights, names, phases = {}, {}, []
    for manifest in manifests:
        if not manifest['batch_lo'] <= batch <= manifest['batch_hi'] or \
                manifest['capacity'] != capacity:
            raise ValueError('memory report workload differs from the plan')
        seq = manifest['seq']; past = manifest['past_hi']
        internal = int(manifest.get('memory_arena_bytes', 0))
        phase_weights={}
        for buffer in manifest['buffers']:
            elements = sum(int(buffer.get(k, 0))*v for k, v in (
                ('constant', 1), ('per_seq', seq), ('per_past', past),
                ('per_total', seq+past), ('per_batch', batch)))
            size = elements*widths[buffer['dtype']]
            if 'recipe' in buffer:
                recipe = json.dumps(buffer['recipe'], sort_keys=True, separators=(',', ':'))
                key = (recipe, buffer['dtype'], size)
                prior = names.setdefault(buffer['name'], key)
                if prior != key:
                    raise ValueError('phase plans disagree on weight '+buffer['name'])
                weights[key] = size
                phase_weights[key]=size
            elif buffer['role'] == 'internal' and 'dm_arena_offset' not in buffer:
                internal += size
        phases.append(dict(phase=manifest['phase'], internal_bytes=internal,
                           weight_bytes=sum(phase_weights.values())))
    layers = int(config['num_hidden_layers']); heads = int(config['num_key_value_heads'])
    dim = int(config.get('head_dim', config['hidden_size']//config['num_attention_heads']))
    kv=layers*2*batch*heads*capacity*dim*2
    state = kv + batch*capacity*4 + 2*capacity*dim*2
    return dict(evidence='inferred', scope='manifest estimate; excludes CUDA/event tables and inserted split-K storage; recipe-based weight deduplication',
        batch=batch, shared_weight_bytes=sum(weights.values()), request_state_bytes=state,
        kv_bytes=kv, workspace_bytes=sum(p['internal_bytes'] for p in phases),
        cross_phase_saved_weight_bytes=sum(p['weight_bytes'] for p in phases)-sum(weights.values()),
        phases=phases, estimated_allocation_bytes=sum(weights.values())+state+
            sum(phase['internal_bytes'] for phase in phases))

