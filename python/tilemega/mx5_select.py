"""Explicit diagnostic MX-5 top-four orchestration; never changes target defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

CLASSES = ('qkv', 'o', 'gate_up', 'down', 'head', 'attention')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_file(record):
    path = Path(record['path']).resolve()
    if sha(path) != record['sha256']:
        raise ValueError('immutable input changed: ' + str(path))
    return path


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def _validate_schema3_diagnostic(doc, table):
    if (doc.get('preview_mode') != 'schema3_qualified_domain_diagnostic'
            or doc.get('source_schema_version') != 3 or doc.get('schema3_diagnostic_only') is not True
            or doc.get('execution_authorized') is not False or doc.get('results_consumption_allowed') is not False):
        raise ValueError('requires explicit audit-only schema3 diagnostic mode')
    if (table.get('qualification_scope') != 'complete_outcomes_single_request_v1'
            or table.get('outcome_domain_complete') is not True or table.get('full_table') is not False
            or any(table.get(key) is not False for key in
                   ('execution_512_approved', 'global_dt576_policy_approved', 'target_defaults_changed'))):
        raise ValueError('requires complete schema3 outcomes without policy approvals')
    report = doc['domain_report']
    for key in ('scope', 'coverage', 'resource_rejections', 'r_next_policy_audit'):
        if json.dumps(report[key], sort_keys=True) != json.dumps(table[key], sort_keys=True):
            raise ValueError('schema3 domain report differs from immutable table: ' + key)
    if any(report.get(key) is not False for key in ('execution_512_approved', 'final', 'default_enabled')):
        raise ValueError('schema3 domain report cannot grant execution/default approval')
    coverage = report['coverage']
    counts = (coverage.get('expected_keys'), coverage.get('costed_keys'), coverage.get('rejected_keys'))
    if (any(type(x) is not int or x < 0 for x in counts) or coverage.get('missing_keys') != 0
            or counts[0] != counts[1] + counts[2] or counts[1] != len(table['rows'])
            or counts[2] != len(table['resource_rejections'])):
        raise ValueError('schema3 complete outcome counts disagree with actual rows')
    scope, request = report['scope'], doc['request']
    for key in ('batch', 'threads', 'grid', 'past_lo', 'past_hi'):
        if type(request.get(key)) is not int or request[key] != scope[key]:
            raise ValueError('schema3 request differs from domain scope: ' + key)
    if request['capacity'] != scope['model']['capacity']:
        raise ValueError('schema3 capacity differs from domain scope')
    status, count = doc['status'], doc['available_candidate_count']
    if type(count) is not int or not 0 <= count <= 4:
        raise ValueError('schema3 diagnostic candidate count is invalid')
    if status == 'predicted_top4_resource_review_required':
        if count != 4 or report.get('ranking_claim') != 'exact_top4_of_certified_executable_stage_domain':
            raise ValueError('schema3 top4 ranking/count claim is invalid')
    elif status in ('no_executable_plan', 'insufficient_top4'):
        if (doc['candidates'] or (count != 0 if status == 'no_executable_plan' else not 1 <= count <= 3)
                or report.get('ranking_claim') != 'no_top4_claim'):
            raise ValueError('schema3 incomplete top4 must not serialize candidate pins')
    else:
        raise ValueError('unknown schema3 diagnostic domain status')


def validate_preview(path, *, allow_schema3_diagnostic=False):
    path = Path(path).resolve()
    doc = load(path)
    if (doc.get('schema_version') != 1 or doc.get('kind') != 'mx5_decode_candidate_preview'
            or doc.get('diagnostic') is not True or doc.get('final') is not False
            or doc.get('solver_selected') is not False or doc.get('default_enabled') is not False):
        raise ValueError('requires an explicit diagnostic native-solver preview')
    table_path = check_file(doc['stage_table'])
    check_file(doc['input']); check_file(doc['producer'])
    table = load(table_path)
    schema3 = table.get('schema_version') == 3
    if schema3 and not allow_schema3_diagnostic:
        raise ValueError('schema3 preview is diagnostic only; explicit allow_schema3_diagnostic is required, not measurement/default consumption')
    if (table.get('schema_version') not in (2, 3) or table.get('table_kind') != 'mx5_decode_stage_costs'
            or table.get('qualified') is not True or table.get('diagnostic_reasons')
            or table.get('synthetic_fixture')):
        raise ValueError('requires qualified schema-v2 costs; synthetic/diagnostic tables cannot be measured')
    if schema3:
        _validate_schema3_diagnostic(doc, table)
        if doc['status'] != 'predicted_top4_resource_review_required':
            return doc
    rows = {json.dumps(row, sort_keys=True, separators=(',', ':')) for row in table['rows']}
    candidates = doc['candidates']
    if len(candidates) != 4 or [c['index'] for c in candidates] != list(range(4)):
        raise ValueError('requires exactly four ordered native-solver candidates')
    for candidate in candidates:
        if schema3 and (candidate.get('diagnostic') is not True or any(candidate.get(key) is not False
                for key in ('final', 'solver_selected', 'default_enabled', 'execution_authorized'))):
            raise ValueError('schema3 candidates must remain explicit audit-only manual diagnostics')
        pin = candidate['pin']['mx_decode_v2']
        if schema3 and pin.get('default_enabled') is not False:
            raise ValueError('schema3 manual pins cannot enable defaults')
        if (pin.get('purpose') != 'manual' or pin.get('diagnostic') is not True
                or pin.get('final') is not False or pin.get('solver_selected') is not False):
            raise ValueError('candidate pins must remain diagnostic manual inputs')
        if load(check_file(candidate['pin_file'])) != candidate['pin']:
            raise ValueError('candidate pin file differs from native-solver preview')
        for cls in CLASSES:
            evidence = candidate['evidence'][cls]
            if len(evidence) != (5 if cls == 'attention' else 1):
                raise ValueError('candidate lacks exact selected stage evidence: ' + cls)
            for row in evidence:
                if json.dumps(row, sort_keys=True, separators=(',', ':')) not in rows:
                    raise ValueError('selected evidence is absent from the immutable qualified table')
                key = row['key']
                if key['threads'] != pin['threads'] or key['b'] != doc['request']['batch']:
                    raise ValueError('selected evidence threads/batch mismatch')
                if cls == 'attention':
                    if key['b_min'] != pin['b_min'] or key['prefetch_kind'] != pin['attn_prefetch']:
                        raise ValueError('attention evidence differs from pin')
                else:
                    c = pin[cls]
                    if any(key[a] != c[b] for a, b in (('tn', 'tn'), ('wn', 'wn'), ('depth', 'd'), ('prefetch', 'prefetch'))):
                        raise ValueError('GEMM evidence differs from pin: ' + cls)
                    if row['r_next'] != c['r_next'] or (cls != 'head' and key['part'] != int(c['part'] == 'kTileRange')):
                        raise ValueError('GEMM R_next/Part evidence mismatch: ' + cls)
    return doc


def verify_stage_sources(candidate, identity, source_map):
    mappings = {}
    for item in source_map['sources']:
        key = item['source_sha256']
        if key in mappings:
            raise ValueError('duplicate stage source-map digest')
        mappings[key] = Path(item['archive']).resolve()
    files = identity['source']['source_files']
    checked = set()
    for cls in CLASSES:
        for row in candidate['evidence'][cls]:
            digest = row['provenance']['source_sha256']
            if digest in checked:
                continue
            if digest not in mappings:
                raise ValueError('missing archived stage source proof for ' + digest)
            archive = mappings[digest]
            h = hashlib.sha256(); shared = 0
            with tarfile.open(archive, 'r:gz') as source:
                members = [m for m in source.getmembers() if m.isfile()]
                if len({m.name for m in members}) != len(members):
                    raise ValueError('duplicate stage source archive member')
                for member in sorted(members, key=lambda m: m.name):
                    body = source.extractfile(member).read()
                    h.update(member.name.encode() + b'\0' + body + b'\0')
                    if member.name.startswith('include/tilemega/'):
                        shared += 1
                        if files.get(member.name) != hashlib.sha256(body).hexdigest():
                            raise ValueError('measured stage headers differ from built plan: ' + member.name)
            if h.hexdigest() != digest or not shared:
                raise ValueError('archived stage source identity mismatch: ' + str(archive))
            checked.add(digest)
    return [dict(source_sha256=key, **record(mappings[key])) for key in sorted(checked)]


def verify_candidate(so, candidate, preview, source_map):
    from .build.identity import verify, validate_snapshot_inputs
    from .mx5_resources import inspect
    identity = verify(so)
    validate_snapshot_inputs(Path(identity['source']['root']), identity['source'])
    plan = identity['plan']; pin = candidate['pin']['mx_decode_v2']
    if (plan.get('version') != 5 or plan.get('executor') != 'mx_static' or plan.get('threads') != pin['threads']
            or plan.get('watchdog') != 0 or plan.get('trace_stage') or plan.get('diag')):
        raise ValueError('candidate artifact is not an uninstrumented native v2 plan')
    if identity['source']['compiler_sha256'] != preview['producer']['sha256']:
        raise ValueError('candidate compiler differs from native solver producer')
    if plan.get('b_min') != pin['b_min'] or plan.get('attn_prefetch') != pin['attn_prefetch']:
        raise ValueError('artifact attention configuration differs from pin')
    for cls in CLASSES[:-1]:
        if any(plan['classes'][cls].get(key) != value for key, value in pin[cls].items()):
            raise ValueError('artifact geometry/prefetch/R_next differs from pin: ' + cls)
    if any(plan.get(key) != preview['request'][key] for key in ('capacity', 'past_lo', 'past_hi')):
        raise ValueError('artifact decode domain differs from solver request')
    if plan.get('batch_lo') != preview['request']['batch'] or plan.get('batch_hi') != preview['request']['batch']:
        raise ValueError('artifact batch differs from solver request')
    resources = inspect(so, str(so) + '.ptxas.log', pin['threads'])
    if resources['violations']:
        raise ValueError('actual candidate resource violations: ' + '; '.join(resources['violations']))
    source_proofs = verify_stage_sources(candidate, identity, source_map)
    return dict(identity=identity, resources=resources, stage_sources=source_proofs,
                sidecars=[record(str(so) + suffix) for suffix in ('', '.identity.json', '.plan.json', '.resources.json', '.ptxas.log')])


def validate_measurement(report, preview, artifact_id, prefill_id):
    if (report.get('batch') != preview['request']['batch'] or report.get('max_tokens') != 1024
            or report.get('decode_loop_used') is not True or report.get('step_ns_read') is not False
            or report.get('timed_tokens_identical') is not True):
        raise ValueError('measurement does not satisfy native decode protocol')
    execution = report['execution_identity']
    if execution.get('artifact_id') != artifact_id or execution.get('executor') != 'L1' or execution.get('loop') is not True:
        raise ValueError('measured execution differs from candidate artifact')
    if report['prefill_execution_identity']['artifact_id'] != prefill_id:
        raise ValueError('measured prefill artifact changed')
    runs = report['runs']
    if len(runs) != 6 or sorted((r['N'], r['run'], r['warmup']) for r in runs) != sorted(
            (n, i, i == 0) for n in (1024, 1) for i in range(3)):
        raise ValueError('measurement requires warmup=1 and timed=2 for full request and TTFT')
    if any(r['step_ns_read'] or len(r['gpu_step_ms']) != 1
           or not math.isfinite(r['gpu_step_ms'][0]) or r['gpu_step_ms'][0] < 0 for r in runs):
        raise ValueError('step-events=0 retains the prefill event only; decode step timing must be absent')
    for r in runs:
        if r['execution_identity'].get('artifact_id') != artifact_id or r['execution_identity'].get('executor') != 'L1':
            raise ValueError('request execution identity differs from candidate artifact')
        if r['N'] == 1 and (r['decode_loop_used'] or r['execution_identity'].get('loop')):
            raise ValueError('TTFT must execute prefill without the decode loop')
        if r['N'] == 1024 and (not r['decode_loop_used'] or r['execution_identity'] != execution):
            raise ValueError('full request did not execute the identified native decode loop')
    if any(key in report for key in ('tpot_mean_seconds', 'tpot_p50_seconds', 'tpot_p90_seconds')):
        raise ValueError('step quantiles were not measured under step-events=0')
    value = report['tpot_seconds']
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError('invalid measured TPOT')
    expected = (report['e2e_seconds'] - report['ttft_seconds']) / 1023
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('TPOT differs from request-minus-TTFT protocol')
    return value * 1e9


def validate_results(path, preview_path):
    from .build.identity import verify
    results = load(path)
    if 'execution_manifest' not in results:
        raise ValueError('measured pairs lack fresh-process/guard/artifact provenance')
    manifest = results['execution_manifest']
    preview = validate_preview(preview_path)
    if (manifest.get('kind') != 'mx5_top4_execution' or manifest.get('final') is not False
            or results['candidate_preview_sha256'] != sha(preview_path)
            or results['stage_table_sha256'] != preview['stage_table']['sha256']):
        raise ValueError('measured result input identities differ from candidate preview')
    for key in ('candidate_preview', 'prompt_ids', 'policy', 'stage_source_map', 'model_config',
                'guard_wrapper', 'guard_policy', 'orchestrator'):
        check_file(manifest[key])
    if any(c['pin']['mx_decode_v2']['threads'] != 576 for c in preview['candidates']):
        raise ValueError('unresolved per-plan 512-thread fallback policy')
    attempts = manifest['resource_attempts']
    if len(attempts) != 4 or {a['candidate'] for a in attempts} != set(range(4)):
        raise ValueError('results lack four actual candidate resource attempts')
    sources = load(check_file(manifest['stage_source_map'])); artifacts = {}
    for attempt in attempts:
        index = attempt['candidate']
        argv = attempt['process']['argv']
        if (attempt['resource_step'] != 0 or attempt['process']['returncode'] != 0
                or attempt['actual_pin'] != preview['candidates'][index]['pin']):
            raise ValueError('unresolved or mismatched actual plan resource attempt')
        proof = verify_candidate(Path(argv[3]), preview['candidates'][index], preview, sources)
        artifacts[index] = proof['identity']['artifact_id']
        if artifacts[index] != attempt['verified']['identity']['artifact_id']:
            raise ValueError('resource-checked artifact identity changed')
    pairs = results['process_tpot_ns']
    if len(pairs) != 4 or any(len(pair) != 2 for pair in pairs):
        raise ValueError('results require four pairs of fresh-process measurements')
    seen, pids = set(), set()
    for process in manifest['processes']:
        index, round_number = process['candidate'], process['round']
        pair = index, round_number
        if index not in range(4) or round_number not in range(2) or pair in seen:
            raise ValueError('duplicate or invalid candidate process coordinates')
        seen.add(pair)
        argv = process['process']['argv']
        if (process['process']['returncode'] != 0 or argv[2] != '900'
                or process['measurement_pid'] in pids):
            raise ValueError('measurement lacks a distinct successful 900-second guarded process')
        pids.add(process['measurement_pid'])
        guard = Path(argv[3])
        if int((guard / 'pgid').read_text()) != process['measurement_pid']:
            raise ValueError('guard child PID differs from measured process')
        if load(check_file(process['guard_result'])).get('code') != 0:
            raise ValueError('GPU guard did not accept measured process')
        prefill = Path(argv[argv.index('--prefill-so') + 1])
        if verify(prefill)['artifact_id'] != manifest['prefill_identity']:
            raise ValueError('prefill identity changed before selection')
        report = load(check_file(process['measurements']))
        if report.get('measurement_policy', {}).get('guard') is not True:
            raise ValueError('measurement power/exclusivity guard was disabled')
        value = validate_measurement(report, preview, artifacts[index], manifest['prefill_identity'])
        if value != pairs[index][round_number] or value != process['tpot_ns']:
            raise ValueError('measured TPOT pair differs from hashed raw process output')
    if len(seen) != 8:
        raise ValueError('results require two fresh accepted processes for every top4 candidate')
    return results


def run_command(argv, directory, *, timeout=900, env=None):
    directory = Path(directory); directory.mkdir(parents=True)
    with (directory / 'stdout.log').open('x') as out, (directory / 'stderr.log').open('x') as err:
        started = time.time()
        process = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            from .bench.gpu_guard import stop, process_start
            stop(process, {process.pid: process_start(process.pid)})
            code = 124
    result = dict(argv=argv, pid=process.pid, started=started, returncode=code, timeout_seconds=timeout)
    write(directory / 'process.json', result)
    return result


def measurement_command(args, so, output, guard):
    return [str(args.root / 'scripts/mx_gpu_job.sh'), str(args.gpu), '900', str(guard), sys.executable,
            '-m', 'tilemega.serving.measure', '--model', str(args.model), '--prefill-so', str(args.prefill_so),
            '--decode-so', str(so), '--prompt-ids', str(args.prompt_ids), '--batch', str(args.batch),
            '--mode', 'L1', '--prefill-mode', args.prefill_mode, '--decode-loop', '1', '--max-new-tokens', '1024',
            '--warmup', '1', '--repeats', '2', '--step-events', '0', '--policy', str(args.policy), '--out', str(output)]


def native_solve_command(preview, prefix, results=None):
    req = preview['request']
    argv = [str(check_file(preview['producer'])), 'compile', str(check_file(preview['input'])), str(prefix),
            '--serving', 'decode', '--emit', 'serving', '--frontend', 'decoder', '--batch', str(req['batch']),
            '--capacity', str(req['capacity']), '--past-range', str(req['past_lo']) + ':' + str(req['past_hi']),
            '--selection', 'predicted', '--runtime-target', req['runtime_target'],
            '--mx-decode-v2-solve', str(check_file(preview['stage_table'])), '--mx-decode-v2-threads', str(req['threads'])]
    if results is not None:
        argv += ['--mx-decode-v2-results', str(results)]
    return argv


def orchestrate(args):
    from .build.identity import verify
    preview = validate_preview(args.preview)
    if args.batch != preview['request']['batch']:
        raise ValueError('measurement batch differs from preview')
    if any(c['pin']['mx_decode_v2']['threads'] != 576 for c in preview['candidates']):
        raise ValueError('unresolved per-plan resource policy: 512-thread execution requires actual 576 failures, prescribed reductions and D7 evidence; no fallback is inferred')
    from .serving.workload import validate_prompt_ids
    validate_prompt_ids(load(args.prompt_ids), args.batch, 64)
    if any(preview['request'][key] != value for key, value in (('capacity', 1088), ('past_lo', 64), ('past_hi', 1086))):
        raise ValueError('top4 protocol requires ISL64/OSL1024 and decode past=64..1086')
    if load(args.policy).get('guard') is not True:
        raise ValueError('top4 measurement requires the declared power/exclusivity policy')
    prefill = verify(args.prefill_so)
    sources = load(args.stage_source_map)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, kind='mx5_top4_execution', status='preparing', final=False,
                    default_enabled=False, candidate_preview=record(args.preview), stage_table=preview['stage_table'],
                    prefill_identity=prefill['artifact_id'], model=str(args.model), prompt_ids=record(args.prompt_ids),
                    policy=record(args.policy), stage_source_map=record(args.stage_source_map),
                    model_config=record(args.model / 'config.json'),
                    guard_wrapper=record(args.root / 'scripts/mx_gpu_job.sh'),
                    guard_policy=record(args.root / 'configs/bench/guard_policy_c550.json'),
                    orchestrator=record(Path(__file__)),
                    protocol=dict(candidates=4, fresh_processes_per_candidate=2, warmup=1, timed=2,
                                  step_events=0, max_new_tokens=1024, guard_timeout_seconds=900),
                    resource_attempts=[], processes=[])
    def snapshot():
        # Each state is immutable, including failures and all earlier attempts.
        write(args.out / ('state%03d.json' % len(list(args.out.glob('state*.json')))), manifest)
    snapshot()
    env = dict(os.environ, PYTHONPATH=str(args.root / 'python'), NEEDS_MIB=str(args.needs_mib),
               TILEMEGA_GPU_JOB_LOG=str(args.out / 'gpu_jobs.log'))
    env.pop('TILEMEGA_MX_DIAG', None); env.pop('TILEMEGA_TRACE', None)
    artifacts = []
    try:
        prefix = args.out / 'native_revalidation' / 'preview'
        prefix.parent.mkdir()
        process = run_command(native_solve_command(preview, prefix), prefix.parent / 'process', env=env)
        manifest['native_revalidation'] = process; snapshot()
        if process['returncode']:
            raise RuntimeError('native solver could not revalidate exact complete-domain top4; no candidate is built')
        fresh = validate_preview(str(prefix) + '.mx5_candidates.json')
        for old, new in zip(preview['candidates'], fresh['candidates']):
            if any(old[key] != new[key] for key in ('index', 'pin', 'predicted_tpot_ns', 'predicted_stage_ns', 'ranks', 'evidence')):
                raise ValueError('preview differs from fresh exact native-solver top4')
        for candidate in preview['candidates']:
            index = candidate['index']; directory = args.out / ('candidate%d' % index); directory.mkdir()
            so = directory / 'plan.so'
            req = preview['request']
            command = [str(check_file(preview['producer'])), 'compile', str(check_file(preview['input'])), str(so),
                       '--serving', 'decode', '--emit', 'serving', '--frontend', 'decoder', '--batch', str(args.batch),
                       '--capacity', str(req['capacity']), '--past-range', str(req['past_lo']) + ':' + str(req['past_hi']),
                       '--runtime-target', req['runtime_target'], '--evaluate-configs', str(check_file(candidate['pin_file'])),
                       '--selection', 'predicted', '--watchdog', '0']
            attempt = run_command(command, directory / 'build', env=env)
            resource_attempt = dict(candidate=index, resource_step=0, actual_pin=candidate['pin'], process=attempt,
                                    automatic_reductions_performed=False, thread_fallback_performed=False)
            for suffix in ('.resource_failure.json', '.resources.json', '.ptxas.log', '.source.json'):
                if Path(str(so) + suffix).is_file():
                    resource_attempt.setdefault('evidence', []).append(record(str(so) + suffix))
            manifest['resource_attempts'].append(resource_attempt); snapshot()
            if attempt['returncode']:
                failure_path = Path(str(so) + '.resource_failure.json')
                failure = load(failure_path) if failure_path.is_file() else {'status': 'no_resource_failure_report'}
                resource_attempt['failure'] = failure; snapshot()
                if not failure.get('violations'):
                    raise RuntimeError('native candidate compile/report failed; no register/stack retry is justified: ' + str(failure))
                reason = 'unresolved per-plan resource policy: ' + '; '.join(failure['violations']) + '; retain exact failed build; largest-Pw reductions (at most three) and D7 before any 576-to-512 fallback are not implemented by this bounded route'
                raise RuntimeError(reason + '; candidate=%d; logs=%s' % (index, directory / 'build'))
            proof = verify_candidate(so, candidate, preview, sources)
            resource_attempt['verified'] = proof; snapshot()
            artifacts.append((so, proof))
        pairs = [[] for _ in range(4)]; pids = set()
        for round_number in range(2):
            for offset in range(4):
                index = (offset + round_number) % 4
                so, proof = artifacts[index]
                validate_preview(args.preview); verify_candidate(so, preview['candidates'][index], preview, sources)
                directory = args.out / ('candidate%d/process%d' % (index, round_number))
                output, guard = directory / 'measure', directory / 'guard'
                command = measurement_command(args, so, output, guard)
                process = run_command(command, directory, timeout=930, env=env)
                manifest['processes'].append(dict(candidate=index, round=round_number, process=process)); snapshot()
                if process['returncode']:
                    raise RuntimeError('guarded fresh measurement failed; candidate=%d round=%d code=%d' % (index, round_number, process['returncode']))
                if load(guard / 'guard_result.json').get('code') != 0:
                    raise ValueError('GPU guard did not record a successful child exit')
                pid = int((guard / 'pgid').read_text())
                if pid in pids:
                    raise ValueError('measurement processes are not distinct')
                pids.add(pid)
                measurement = load(output / 'measurements.json')
                if measurement.get('measurement_policy', {}).get('guard') is not True:
                    raise ValueError('measurement power/exclusivity policy is disabled')
                pairs[index].append(validate_measurement(measurement, preview, proof['identity']['artifact_id'], prefill['artifact_id']))
                manifest['processes'][-1].update(measurement_pid=pid, measurements=record(output / 'measurements.json'),
                    guard_result=record(guard / 'guard_result.json'), tpot_ns=pairs[index][-1])
                snapshot()
        validate_preview(args.preview)
        for index, (so, proof) in enumerate(artifacts):
            final = verify_candidate(so, preview['candidates'][index], preview, sources)
            if final['identity']['artifact_id'] != proof['identity']['artifact_id']:
                raise ValueError('candidate artifact changed during measurement')
        if verify(args.prefill_so)['artifact_id'] != prefill['artifact_id']:
            raise ValueError('prefill artifact changed during measurement')
        for key in ('candidate_preview', 'prompt_ids', 'policy', 'stage_source_map', 'model_config', 'guard_wrapper', 'guard_policy', 'orchestrator'):
            check_file(manifest[key])
        results = dict(candidate_preview_sha256=sha(args.preview), stage_table_sha256=preview['stage_table']['sha256'],
                       process_tpot_ns=pairs, execution_manifest=manifest)
        results_path = args.out / 'verified_measurements.json'; write(results_path, results)
        prefix = str(args.preview).removesuffix('.mx5_candidates.json')
        command = native_solve_command(preview, prefix, results_path)
        process = run_command(command, args.out / 'native_selection', env=env)
        if process['returncode']:
            raise RuntimeError('native solver measured-selection consumer failed')
        selection_path = Path(prefix + '.mx5_selection.json')
        selection = load(selection_path)
        if selection.get('kind') != 'mx5_decode_top4_selection' or selection.get('final') is not False:
            raise ValueError('native selection report is invalid')
        manifest.update(status='measured_candidate_selection', selection=record(selection_path),
                        selected_artifact=record(artifacts[selection['selected_index']][0]), selection_applied=False)
        snapshot()
        return manifest
    except BaseException as error:
        manifest.update(status='blocked', reason=str(error)); snapshot(); raise


def main():
    if sys.argv[1:2] == ['--verify-results']:
        parser = argparse.ArgumentParser(description='Verify immutable CPU-readable top4 process evidence')
        parser.add_argument('--verify-results', type=Path, required=True)
        parser.add_argument('--preview', type=Path, required=True)
        args = parser.parse_args()
        validate_results(args.verify_results, args.preview)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('preview', 'root', 'model', 'prefill-so', 'prompt-ids', 'stage-source-map', 'policy', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--gpu', type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument('--prefill-mode', choices=('L1', 'L2', 'auto'), default='L2')
    parser.add_argument('--needs-mib', type=int, default=20480)
    args = parser.parse_args()
    for name in ('preview', 'root', 'model', 'prefill_so', 'prompt_ids', 'stage_source_map', 'policy', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    orchestrate(args)


if __name__ == '__main__':
    main()
