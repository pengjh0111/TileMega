#!/usr/bin/env python3
"""Versioned CPU control for TVM infrastructure failures; recipe6 stays frozen."""
from pathlib import Path
import dnn1_baselines as _base

VERSION = 'tvm-infrastructure-reuse-control-v2'
ORIGINAL = Path(__file__).with_name('dnn1_baselines.py')
ORIGINAL_SHA256 = 'bca21e6001155d4636ef509b22edd2a606f1a0d0410347dcfc77df9eb215ada2'
if _base.sha(ORIGINAL) != ORIGINAL_SHA256:
    raise RuntimeError('the infrastructure wrapper requires its pinned original CPU driver')
_queue, _command, _fallback = _base.queue, _base.arm_command, _base.fallback_arm


def infrastructure_failure(arm, result):
    if arm.get('system') != 'tvm':
        return None
    final = result.get('final', {})
    failure = final.get('failure', {})
    error = failure.get('error', '')
    if final.get('returncode') == 75:
        return 'resource guard rejection'
    if isinstance(final.get('returncode'), int) and final['returncode'] < 0:
        return 'measurement process interrupted by signal'
    if 'incomplete or failed TVM cache attempt; explicit diagnosis required before retry' in error:
        return 'incomplete cache requires a separately recorded recovery attempt'
    return None


def fallback_arm(arm, result):
    reason = infrastructure_failure(arm, result)
    if reason:
        final = result['final']
        previous = dict(final.get('failure', {}))
        final['failure'] = dict(failure_kind='infrastructure', error=reason,
                                original_failure=previous, control_version=VERSION)
        raise _base.HardStop('TVM infrastructure rejection; no FP16 alternative: ' + reason)
    return _fallback(arm, result)


def arm_command(cell, arm, role, index, folder):
    command = _command(cell, arm, role, index, folder)
    if arm['system'] == 'tvm' and cell.get('infrastructure_cache_work_dir'):
        command[command.index('--tvm-work-dir') + 1] = str(
            Path(cell['infrastructure_cache_work_dir']) / arm['precision'])
    return command


def validate_reuse(key, value):
    """An explicit finished cache may be reused; interrupted attempts stay intact."""
    work = Path(value['work_dir']).resolve()
    if not work.is_relative_to(_base.ROOT/'runs/dnn1') or not work.is_dir():
        raise ValueError('completed cache root must exist under runs/dnn1')
    session = Path(value['source_session'])
    if _base.sha(session) != value['source_session_sha256']:
        raise ValueError('completed cache source session hash differs')
    result = _base.read(session)
    if result.get('status') != 'complete' or not result.get('ended'):
        raise ValueError('cache reuse requires a finished session; not performance acceptance')
    if f"{result['model']}_b{result['batch']}" != key:
        raise ValueError('completed cache source session belongs to another cell')
    entries = value.get('compiled_receipts', {})
    if not entries:
        raise ValueError('completed cache requires pinned compiled receipts')
    for filename, digest in entries.items():
        path = Path(filename).resolve()
        if not path.is_relative_to(work) or _base.sha(path) != digest:
            raise ValueError('compiled cache receipt path/hash differs')
        receipt = _base.read(path)
        identity = receipt['identity']
        if (identity['recipe']['schema'] != 6 or identity['batch'] != result['batch'] or
                identity['precision'] != 'bf16' or path.parent.parent.name != key+'_bf16' or
                _base.sha(path.with_name('model.so')) != receipt['model_sha256']):
            raise ValueError('completed VM identity/hash differs')
    return str(work)


def queue(args):
    config = _base.read(args.spec)
    recovery = config.get('tvm_infrastructure_recovery', {})
    selected = {f"{row['model']}_b{row['batch']}" for row in config.get('cells', [])}
    reusable = config.get('tvm_completed_cache', {})
    for key, value in reusable.items():
        if not selected or key in selected:
            validate_reuse(key, value)
            if key in recovery:
                raise ValueError('a cell cannot request fresh recovery and completed reuse together')
    # A recovery names a fresh root; never erase or overwrite the rejected cache.
    for key, value in recovery.items():
        if selected and key not in selected:
            continue
        path = Path(value['work_dir']).resolve()
        if not path.is_relative_to(_base.ROOT/'runs/dnn1') or path.exists():
            raise ValueError('recovery work_dir must be fresh and under runs/dnn1: ' + key)
        if not value.get('rejected_session'):
            raise ValueError('recovery must identify the retained rejected session')
    _queue(args)
    prepared = _base.read(args.out/'queue-spec.json')
    for path in map(Path, prepared['cells']):
        cell = _base.read(path)
        key = f"{cell['model']}_b{cell['batch']}"
        cell.update(infrastructure_control_version=VERSION,
                    infrastructure_control_sha256=_base.sha(__file__),
                    wrapped_cpu_driver_sha256=ORIGINAL_SHA256)
        cell['recipe_sources'][str(ORIGINAL.resolve())] = ORIGINAL_SHA256
        if key in recovery:
            cell['infrastructure_cache_work_dir'] = str(Path(recovery[key]['work_dir']).resolve())
            cell['infrastructure_recovery'] = recovery[key]
        if key in reusable:
            cell['infrastructure_cache_work_dir'] = validate_reuse(key, reusable[key])
            cell['completed_cache_reuse'] = reusable[key]
        _base.write(path, cell)


# Existing run/preflight/queue source checks now identify this versioned wrapper.
# The numerical harness, TVM importer/compiler and guard files remain untouched.
_base.__file__ = str(Path(__file__).resolve())
_base.queue, _base.arm_command, _base.fallback_arm = queue, arm_command, fallback_arm


def __getattr__(name):
    return getattr(_base, name)


if __name__ == '__main__':
    _base.main()
