#!/usr/bin/env python3
"""One guarded, fresh-process MoE DIAG intervention using the MX-5 protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path

MASKS = (0, 64, 65, 66, 68, 80, 96, 112, 120, 127)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'prefill-so', 'decode-so', 'prompt-ids', 'query', 'policy', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--batch', type=int, choices=(1, 16), required=True)
    p.add_argument('--skip', type=int, choices=MASKS, required=True)
    p.add_argument('--round', type=int, choices=(1, 2), required=True)
    a = p.parse_args()
    from tilemega.build.identity import verify
    from tilemega.moe.plan_pair import validate_plan_pair
    from tilemega.mx5_plan_occupancy import guard_evidence
    guard = guard_evidence(os.environ)
    di, pi = verify(a.decode_so), verify(a.prefill_so)
    decode, prefill = di['plan'], pi['plan']
    if not decode['diag'] or prefill['diag'] or decode['trace_stage'] or prefill['trace_stage']:
        raise ValueError('ledger requires DIAG decode, normal prefill, and no trace')
    validate_plan_pair(prefill, decode, a.batch, expected_watchdog=0)
    query = json.loads(a.query.read_text())
    if (query['bound_inputs']['binary'] != reference(a.decode_so) or query['resident_ctas'] != 1 or
            query['attributes']['localSizeBytes'] or query['launches'] or
            query['guard']['physical_gpu'] != guard['physical_gpu']):
        raise ValueError('ledger lacks the matching actual resource query on this card')
    if next(r for r in decode['buffers'] if r['name'] == 'mx.routing_weights')['dtype'] != 'bf16':
        raise ValueError('ledger requires the BF16 routing contract')
    os.environ['TILEMEGA_MX_DIAG_SKIP'] = str(a.skip)
    a.out.mkdir(parents=True, exist_ok=False)
    scope = dict(kind='mx6_moe_diag_skip_intervention', round=a.round, mask=a.skip,
        batch=a.batch, layers=decode['layers'], executor='L1_loop', implementation='mx_moe_v2',
        contract='bf16', PID=os.getpid(), guard=guard, query=reference(a.query),
        decode=reference(a.decode_so), prefill=reference(a.prefill_so),
        decode_artifact_id=di['artifact_id'], prefill_artifact_id=pi['artifact_id'],
        source_digest=di['source']['source_digest'], coordinates=decode['classes'],
        b_min=decode['b_min'], attn_prefetch=decode['attn_prefetch'],
        nonadditive=True, body_latency_claim=False, numerical_correctness_claim=False,
        production_acceptance=False, diag=1, watchdog=0, trace_stage=0, step_events=False,
        initialization='zero tokens before each generate; native normal prefill; unwritten scratch starts zero')
    (a.out / 'diagnostic_scope.json').write_text(json.dumps(scope, indent=2) + '\n')
    import torch
    from tilemega.serving.engine import ServingEngine
    from tilemega.serving.measure import _preflight_external_memory, measure
    from tilemega.serving.workload import validate_prompt_ids
    ids = json.loads(a.prompt_ids.read_text())
    if validate_prompt_ids(ids, a.batch) != 64:
        raise ValueError('ledger requires ISL64')
    prompts = torch.tensor(ids[:a.batch], dtype=torch.int32)
    raw = []
    _preflight_external_memory(a.out)
    with ServingEngine(a.model, a.prefill_so, a.decode_so, a.batch, prompt_len=64,
            max_new_tokens=1024, mode='L1', prefill_mode='L1', decode_loop=True, step_events=False) as engine:
        original = engine.generate

        def seeded_generate(prompt_ids, new_tokens=None):
            engine.state.tokens.zero_()
            torch.cuda.synchronize()
            result = original(prompt_ids, new_tokens)
            count = new_tokens or engine.max_new_tokens
            stamps = engine.decode.read_step_ns(0, count) if result.decode_loop_used and count > 1 else []
            if stamps and (result.step_ns_read or len(stamps) != count or
                           not all(y > x for x, y in zip(stamps, stamps[1:]))):
                raise ValueError('invalid diagnostic timestamps')
            raw.append(dict(invocation=len(raw), N=count, e2e_seconds=result.e2e_ms / 1e3,
                tokens=result.tokens.tolist(), step_ns=stamps, decode_loop_used=result.decode_loop_used,
                step_ns_read=result.step_ns_read, accepted=False, diagnostic_manual_timestamp_read=bool(stamps)))
            with (a.out / 'raw_timestamp_invocations.jsonl').open('a') as output:
                output.write(json.dumps(raw[-1]) + '\n')
            return result

        engine.generate = seeded_generate
        result = measure(engine, prompts, a.out, warmup=1, repeats=2, policy_path=a.policy)
    for run in result['runs']:
        matches = [v for v in raw if not v['accepted'] and all(v[k] == run[k] for k in
                   ('N', 'e2e_seconds', 'tokens', 'decode_loop_used', 'step_ns_read'))]
        if len(matches) != 1:
            raise ValueError('accepted measurement does not bind one raw invocation')
        matches[0].update(accepted=True, run=run['run'], warmup=run['warmup'])
    timed = [v for v in raw if v['accepted'] and v['N'] == 1024 and not v['warmup']]
    if len(timed) != 2 or any(len(v['step_ns']) != 1024 for v in timed):
        raise ValueError('ledger requires two accepted 1024-token requests')
    (a.out / 'raw_step_timestamps.json').write_text(json.dumps(raw) + '\n')
    (a.out / 'diagnostic_result.json').write_text(json.dumps(dict(scope=scope, measurement=result), indent=2) + '\n')
    print('DIAG_INTERVENTION_COMPLETE', a.batch, a.round, a.skip, flush=True)


if __name__ == '__main__':
    main()
