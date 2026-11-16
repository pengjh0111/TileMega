#!/usr/bin/env python3
"""Validate explicit terminal AIT failures without fabricating timing evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import dnn1_baselines as base


KINDS = {'compiler_build', 'build_timeout', 'import_unsupported', 'runtime_unsupported',
         'model_correctness'}


def checked(record):
    path = Path(record['path'])
    if base.sha(path) != record['sha256']:
        raise ValueError('AIT terminal-failure evidence changed: ' + str(path))
    return path


def validate(value):
    if (value.get('schema') != 1 or value.get('system') != 'aitemplate' or
            value.get('status') != 'N/A' or value.get('final') is not True or
            value.get('repair_pending') is not False or value.get('failure_kind') not in KINDS):
        raise ValueError('AIT failure must have an explicit final, repaired-or-unresolved disposition')
    reason = value.get('reason', '')
    if not reason or any(word in reason.lower() for word in
            ('pending', 'waiting', 'interference', 'guard75', 'guard rejection', 'cancelled')):
        raise ValueError('AIT infrastructure or unfinished outcomes are not terminal failures')
    cell = base.read(checked(value['cell']))
    identity = base.read(checked(value['attempt_identity']))['identity']
    if (cell['model'] != value['model'] or cell['batch'] != value['batch'] or
            identity['model'] != value['model'] or identity['batch'] != value['batch'] or
            identity['recipe']['precision'] != 'bf16' or
            identity['recipe']['graph_mode'] is not True or cell['graph_mode'] is not True):
        raise ValueError('AIT failed-cell identity or uniform BF16 graph recipe differs')
    if any(cell['source_sha256'].get(path) != digest
           for path, digest in identity['files'].items()):
        raise ValueError('AIT failed-cell source binding differs from the actual build attempt')
    scheduler = base.read(checked(value['scheduler']))[f"ait-{value['model']}-b{value['batch']}"]
    if (scheduler['status'] != 'failed' or scheduler.get('exit_code') in (0, 75, 143, -15) or
            scheduler.get('attempts', 0) < 1 or not scheduler.get('started')):
        raise ValueError('AIT terminal failure requires an actual failed payload, not admission/cancellation')
    guard = base.read(checked(value['outer_guard']))
    if guard.get('code') != scheduler['exit_code']:
        raise ValueError('AIT failure scheduler and outer guard exit codes differ')
    command = base.read(checked(value['command']))
    if not command:
        raise ValueError('AIT failure lacks the original payload command')
    stderr = checked(value['stderr']).read_text(errors='replace')
    failure = base.read(checked(value['original_failure'])) if value.get('original_failure') else {}
    if value['failure_kind'] == 'build_timeout':
        if (scheduler['exit_code'] != 124 or scheduler.get('run_s', 0) < 14400 or
                (checked(value['attempt_identity']).parent / 'compiled.json').exists()):
            raise ValueError('AIT build timeout requires the genuine four-hour boundary')
    elif not failure.get('error'):
        raise ValueError('AIT failure lacks its original exception')
    if value['failure_kind'] == 'compiler_build' and not any(
            word in stderr.lower() for word in ('internal compiler error', 'error:', 'build has failed')):
        raise ValueError('AIT compiler failure lacks actual compiler stderr')
    for evidence in value.get('extra_evidence', []):
        checked(evidence)
    return cell, identity, scheduler, guard


def row(value, fields, declaration):
    cell, identity, scheduler, _ = validate(value)
    compact = lambda data: json.dumps(data, sort_keys=True, separators=(',', ':'))
    result = {field: '' for field in fields}
    result.update(model=value['model'], batch=value['batch'], system='aitemplate',
        arm=f"aitemplate-bf16-graph-v{identity['recipe']['schema']}", precision='bf16',
        gpu=cell['gpu'], status='N/A', reason=value['reason'], versions=compact(identity),
        source=compact([str(Path(declaration).resolve())]), correctness_passes='not reached',
        strict_v1_status='N/A', p_cell_complete=False,
        exported_program_sha256=identity['exported_program_sha256'],
        source_checkpoint_sha256=identity['source_checkpoint_sha256'])
    diag = dict(model=value['model'], batch=value['batch'], status='N/A', physical_gpu=cell['gpu'],
        gpu_uuid=identity['gpu_uuid'], precision='bf16', graph_requested=True,
        graph_observations=compact(dict(requested=True, capture_verified=False,
            scope='no accepted whole-model timing; graph execution not established')),
        recipe=compact(identity['recipe']),
        source_identity_sha256=hashlib.sha256(compact(identity).encode()).hexdigest(),
        installed_sources_sha256=identity['installed_sources_sha256'],
        library='', library_sha256='', compiled_receipt='', compiled_receipt_sha256='',
        sm_clock_start_mhz='', sm_clock_end_mhz='', session='', session_sha256='',
        outer_guard=value['outer_guard']['path'], outer_guard_sha256=value['outer_guard']['sha256'],
        profile_cache_before='', profile_cache_after='',
        error_chain=compact(dict(declaration=str(Path(declaration).resolve()),
            declaration_sha256=base.sha(declaration), validator_sha256=base.sha(__file__),
            failure_kind=value['failure_kind'],
            reason=value['reason'], scheduler_exit=scheduler['exit_code'],
            evidence={name: value.get(name) for name in
                ('cell', 'attempt_identity', 'scheduler', 'outer_guard', 'command', 'stderr',
                 'original_failure', 'extra_evidence')})))
    return result, diag
