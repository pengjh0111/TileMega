"""Shared-GPU permission is restricted to explicit correctness diagnostics."""
import json
from .measure import _exclusive, _gpu_owners


def validation_guard(path, label, wait, *, allow_shared=False):
    if not allow_shared:
        return _exclusive(path, label, wait)
    record = dict(label=label, guard_mode='shared_correctness', timing_eligible=False)
    try:
        pids, visible, used = _gpu_owners()
        record.update(pids=sorted(pids), visible_mib=visible, used_mib=used)
    except Exception as error:
        record['occupancy_error'] = str(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return True
