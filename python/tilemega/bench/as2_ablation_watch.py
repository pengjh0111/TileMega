"""Finite CPU-only follow-up for the sole no-lock NPU submitter."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import traceback

from tilemega.bench.as1_run import process_identity
from tilemega.bench.as2_ablation import summarize
from tilemega.bench.as2_queue import controlled,write


def watch(directory,hours):
    directory=controlled(directory)
    if not 0<hours<=2:raise ValueError('Require a finite CPU watch budget of at most two hours')
    import os
    # Identity is an audit receipt, never an exclusion mechanism or device lock.
    identity=process_identity(os.getpid());deadline=time.monotonic()+3600*hours
    write(directory/'followup.json',dict(status='watching',identity=identity,finite_hours=hours))
    while time.monotonic()<deadline:
        state=json.loads((directory/'state.json').read_text())
        if state['status'] not in ('prepared','running'):break
        time.sleep(5)
    try:
        out=directory/'ablation.json'
        if out.exists():
            result=json.loads(out.read_text())
            if result['protocol_sha256']!=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest():
                raise ValueError('Existing summary belongs to a different protocol')
        else:result=summarize(directory,out)
        value=dict(status='completed',identity=identity,finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            queue_status=state['status'],summary=str(out),measurement_status=result['status'],
            no_automatic_default_or_git_change=True,no_device_operations=True)
    except Exception:
        value=dict(status='failed',identity=identity,error=traceback.format_exc(),no_device_operations=True)
    write(directory/'followup.json',value)
    print(json.dumps(value),flush=True)
    return 0 if value['status']=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--hours',type=float,default=2)
    a=p.parse_args();raise SystemExit(watch(a.directory,a.hours))
