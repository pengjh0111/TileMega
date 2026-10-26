"""AS-2 entry point for the shared, no-lock single-device job runner."""
import json
from pathlib import Path
import sys
from tilemega.bench.as1_run import main, process_identity


def unfinished_owned_jobs():
    """Observe our receipts only; this is not a device/PID lock or mutex.

    The caller must still await each submission. This read-only preflight
    catches an accidentally unawaited previous tool session without inspecting
    or restricting unrelated processes, including jobs on the same account.
    """
    root=Path('/home/d50064638/junhuip/TileMega/runs/as2/jobs')
    terminal={'passed','failed','timeout','interrupted','launch_error','blocked-resource','telemetry_unavailable'}
    active=[]
    for path in root.glob('*/job.json'):
        receipt=json.loads(path.read_text())
        if receipt.get('project')!='as2' or receipt.get('status') in terminal or not receipt.get('pid'):
            continue
        try:actual=process_identity(int(receipt['pid']))
        except (FileNotFoundError,ProcessLookupError):continue
        if all(actual[k]==receipt.get(k) for k in ('pid','start_ticks','process_group','session')):
            active.append(dict(job=str(path.parent),**actual))
    return active

if __name__ == '__main__':
    active=unfinished_owned_jobs()
    if active:
        print(json.dumps(dict(status='own_submission_still_running',jobs=active,
            device_locked=False,reason='Await our previous submission; external device users are not checked')),file=sys.stderr)
        raise SystemExit(75)
    raise SystemExit(main(project='as2'))
