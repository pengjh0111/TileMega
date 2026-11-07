#!/usr/bin/env python3
"""Use the frozen guarded scheduler with shorter X1 polling and retry waits."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sm90d1_common import ROOT, read, sha, write


class PollingClock:
    def __init__(self,clock,poll):
        self.clock,self.poll=clock,poll

    def __getattr__(self,name):
        return getattr(self.clock,name)

    def sleep(self,seconds):
        return self.clock.sleep(min(seconds,self.poll))


def shorten_retry_waits(state,seconds):
    for record in state.values():
        if record['status']=='pending' and record.get('exit_code')==75 and 'ended' in record:
            record['retry_at']=min(record.get('retry_at',0),record['ended']+seconds)


def children(pid):
    found=set()
    for task in Path('/proc',str(pid),'task').glob('*'):
        try:found.update(map(int,(task/'children').read_text().split()))
        except OSError:pass
    return found


def process_start(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0] not in ('Z','X') else None
    except OSError:return None


def stop_for_inspection(pid):
    os.kill(pid,signal.SIGSTOP)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('T','t','Z','X'):return
        time.sleep(.001)
    raise RuntimeError('process did not stop for preflight inspection')


def abandon_preflight(scheduler,state_path):
    """Freeze the submitter and guards before checking that no child has started."""
    stopped={};terminating=False
    try:
        token=process_start(scheduler.pid)
        if token is None:return False
        stopped[scheduler.pid]=token;stop_for_inspection(scheduler.pid)
        for pid in children(scheduler.pid):
            token=process_start(pid)
            if token is None:continue
            stopped[pid]=token;stop_for_inspection(pid)
            command=Path('/proc',str(pid),'cmdline').read_bytes()
            if b'tilemega.bench.gpu_guard' not in command or children(pid):return False
        state=read(state_path)
        if any((Path(r['out'])/'pgid').exists() for r in state.values() if r['status']=='running'):
            return False
        write(Path(state_path).parent/'preflight-handoff.json',dict(at=time.time(),pids=list(stopped),
              reason='another allowed GPU is idle; no guard measurement child exists'))
        terminating=True
        for pid in stopped:
            if process_start(pid)!=stopped[pid]:continue
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        scheduler.wait(timeout=10)
        return True
    except (OSError,RuntimeError,subprocess.TimeoutExpired):
        return False
    finally:
        for pid,token in reversed(list(stopped.items())):
            if process_start(pid)==token:
                try:os.kill(pid,signal.SIGKILL if terminating else signal.SIGCONT)
                except ProcessLookupError:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--poll',type=float,default=5)
    parser.add_argument('--retry-wait',type=float,default=5)
    args,remainder=parser.parse_known_args()
    if args.poll<=0 or args.retry_wait<=0:parser.error('poll and retry wait must be positive')
    sys.path.insert(0,str(ROOT/'python'))
    from tilemega.bench import scheduler
    original=scheduler.ready_steps

    def ready_steps(steps,state,now):
        shorten_retry_waits(state,args.retry_wait)
        return original(steps,state,now)

    scheduler.ready_steps=ready_steps
    scheduler.time=PollingClock(time,args.poll)
    if '--out' in remainder:
        out=Path(remainder[remainder.index('--out')+1])
        write(out/'x1-adapter.json',dict(adapter_sha256=sha(__file__),
              frozen_scheduler_sha256=sha(scheduler.__file__),poll_s=args.poll,retry_wait_s=args.retry_wait,
              guard_unchanged=True,arguments=remainder))
    scheduler.main(remainder)


if __name__=='__main__':
    main()
