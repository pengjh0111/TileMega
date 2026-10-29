"""CPU contract for a native MoE prompt/decode candidate's shared bindings."""
import hashlib
import json


def validate_plan_pair(prefill, decode, batch, expected_watchdog=None):
    if batch not in (1, 2, 4, 8, 16):
        raise ValueError('native MoE pair has an unsupported batch')
    for plan, phase, executor, seq in ((prefill, 'prefill', 'mx_static_prompt', 64),
                                      (decode, 'decode', 'mx_static', 1)):
        if plan.get('program_kind') != 'moe' or plan.get('phase') != phase or plan.get('executor') != executor or plan.get('seq') != seq:
            raise ValueError('native MoE pair requires prompt64 and decode native manifests')
        if plan.get('batch_lo') != batch or plan.get('batch_hi') != batch or plan.get('capacity') != 1088:
            raise ValueError('native MoE pair batch/capacity differs from workload')
        if plan.get('kv_layout') != 'vt' or plan.get('kv_storage') != 'block_fragments':
            raise ValueError('native MoE pair requires the selected shared KV layout')
        if plan.get('layers') != 24 or plan.get('stages_per_step') != 145 or plan.get('experts') != 128 or plan.get('top_k') != 8:
            raise ValueError('native MoE pair requires the specified MoE-L24 program')
        if expected_watchdog is not None and plan.get('watchdog') != expected_watchdog:
            raise ValueError('native MoE pair watchdog differs from requested build')
    for field in ('model', 'head_dim', 'heads_kv', 'q_per_kv'):
        if field not in prefill or prefill[field] != decode.get(field):
            raise ValueError('native MoE pair model geometry differs: ' + field)
    if prefill.get('prompt_barriers') != 9217 or decode.get('prompt_barriers') != 0:
        raise ValueError('native MoE pair has incorrect prompt/head barrier counts')

    def bindings(plan):
        rows = [row for row in plan['buffers'] if row['role'] == 'external']
        if len(rows) != 246 or len({row['name'] for row in rows}) != len(rows):
            raise ValueError('native MoE pair must retain all 246 unique external bindings')
        return [(row['name'], row['dtype'], row['constant'], row['per_batch'], row.get('recipe')) for row in rows]
    prompt_bindings, decode_bindings = bindings(prefill), bindings(decode)
    if prompt_bindings != decode_bindings:
        raise ValueError('native MoE candidate must use matching prompt/decode names, dtypes, sizes and recipes')
    digest = hashlib.sha256(json.dumps(prompt_bindings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(batch=batch, layers=24, external_buffers=246, binding_sha256=digest,
        prompt_barriers=9217, decode_stages=145, shared_binding_contract=True,
        not_correctness_or_solver_acceptance=True)
