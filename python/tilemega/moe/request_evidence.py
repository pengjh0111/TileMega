"""Validate measured native MoE requests without granting model acceptance."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from tilemega.build.identity import bind_execution, digest, verify
from tilemega.mx5_select import record, validate_measurement

from .cost_provenance import Files, need
from .plan_pair import validate_plan_pair


def validate_report(report, decode_identity, prefill_identity, batch):
    need(type(batch) is int and batch in (1, 2, 4, 8, 16), 'invalid request batch')
    pair = validate_plan_pair(prefill_identity['plan'], decode_identity['plan'], batch, 0)
    for identity in (decode_identity, prefill_identity):
        plan = identity['plan']
        need(not identity['trace'] and not plan.get('diag') and not plan.get('trace_stage'),
             'timed request uses an instrumented native program')
        kernels = [r for name, r in identity['resources'].items()
                   if ('mx_moe_kernel_v3' if plan.get('program_version',2)==3 else 'mx_moe_kernel_v2')
                   in name and '__device_stub__' not in name]
        need(len(kernels) == 1 and kernels[0]['stack_bytes'] == 0
             and kernels[0]['static_smem_bytes'] == 0
             and 0 < kernels[0]['registers'] <= (227 if plan['threads'] == 576 else 255),
             'timed request artifact fails the native resource gate')
    need(not report.get('synthetic_fixture'), 'synthetic request is not measurement evidence')
    value = validate_measurement(report, dict(request=dict(batch=batch)),
        decode_identity['artifact_id'], prefill_identity['artifact_id'])

    def execution(identity, loop):
        expected = bind_execution(identity, 'L1', loop=loop)
        expected['shared_memory_bytes'] = identity['shared_memory_bytes']
        expected['execution_id'] = digest({k: v for k, v in expected.items() if k != 'execution_id'})
        return expected

    need(report['execution_identity'] == execution(decode_identity, True)
         and report['prefill_execution_identity'] == execution(prefill_identity, False),
         'actual execution identity differs from the native kernel or shared memory')
    for name in ('execution_identity', 'prefill_execution_identity'):
        actual = report[name]
        need(actual.get('spill') is False and actual.get('trace') is False
             and actual.get('pdl') is False and actual.get('executor') == 'L1',
             'timed execution uses spill, trace, PDL or a different executor')
    need(report['prefill_execution_identity'].get('loop') is False,
         'native prompt request must not use a serving decode loop')
    timed = {count: [] for count in (1, 1024)}
    for run in report['runs']:
        need(type(run['N']) is int and type(run['run']) is int and type(run['warmup']) is bool,
             'request run coordinates have invalid types')
        need(run['execution_identity'] == execution(decode_identity, run['N'] == 1024),
             'request run kernel identity differs from the actual decode artifact')
        elapsed = run['e2e_seconds']
        need(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
             'request latency is invalid')
        tokens = run['tokens']
        need(len(tokens) == batch and all(len(row) == run['N'] and
             all(type(t) is int and 0 <= t < 151936 for t in row) for row in tokens),
             'raw generated token dimensions or vocabulary differ')
        if not run['warmup']:
            timed[run['N']].append(run)
    full, first = (statistics.median(r['e2e_seconds'] for r in timed[n]) for n in (1024, 1))
    need(report['e2e_seconds'] == full and report['ttft_seconds'] == first,
         'request summaries differ from the two actual timed latencies')
    need(timed[1024][0]['tokens'] == timed[1024][1]['tokens'],
         'actual timed generations differ despite their summary flag')
    initial = [[row[0]] for row in timed[1024][0]['tokens']]
    need(all(r['tokens'] == initial for r in timed[1]), 'TTFT tokens differ from the full request')
    throughput = report['output_tokens_per_second']
    need(type(throughput) in (int, float) and math.isfinite(throughput)
         and math.isclose(throughput, batch * 1024 / full, rel_tol=1e-12),
         'request throughput differs from the actual end-to-end latency')
    return dict(tpot_ns=value, ttft_ns=first * 1e9, e2e_ns=full * 1e9,
        output_tokens_per_second=throughput, plan_pair_contract=pair,
        model_acceptance_implied=False, grants_512_exception=False,
        fresh_process_or_guard_qualification_implied=False, final=False, default_enabled=False)


def validate_request(path, decode, prefill, batch):
    files = Files()
    paths = [Path(path), Path(decode), Path(prefill)]
    refs = [record(p) for p in paths]
    for ref in refs:
        files.check(ref)
    identities = [verify(p) for p in paths[1:]]
    for binary in paths[1:]:
        for suffix in ('.identity.json', '.cu', '.plan.json', '.ptxas.log', '.build_command.txt'):
            files.check(record(str(binary) + suffix))
    report = json.loads(paths[0].read_text())
    result = validate_report(report, *identities, batch)
    for run in report['runs']:
        raw = paths[0].parent / ('tokens_N%d_run%d.json' % (run['N'], run['run']))
        files.check(record(raw))
        need(json.loads(raw.read_text()) == run['tokens'], 'raw token file differs from the request report')
    files.unchanged()
    return dict(result, measurement=refs[0], decode=refs[1], prefill=refs[2])
