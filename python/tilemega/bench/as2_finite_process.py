"""Finite own-child execution, including descendants that create new sessions."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import time
from tilemega.bench.as1_run import process_identity


class Deadline(BaseException):
    pass


def remember_tree(pid, identities):
    try:
        identity=process_identity(pid)
        old=identities.get(pid)
        if old is not None and old!=identity:return
        identities[pid]=identity
        for child in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():
            remember_tree(int(child),identities)
    except (FileNotFoundError,ProcessLookupError):pass


def alive(identity):
    try:return process_identity(identity['pid'])==identity
    except (FileNotFoundError,ProcessLookupError):return False


def terminate_owned(identities):
    # Membership comes exclusively from this launch's ancestry and start ticks.
    # UID, HOME, device process lists and unrelated groups are never consulted.
    for pid in list(identities):
        if alive(identities[pid]):remember_tree(pid,identities)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for identity in reversed(list(identities.values())):
            if alive(identity):
                try:os.kill(identity['pid'],sig)
                except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(.5)
    return [identity for identity in identities.values() if alive(identity)]


def run(argv,log,deadline,timeout_s,*,env=None,on_identity=None):
    """The total deadline bounds CPU helpers and wrapper/device descendants."""
    remaining=deadline-time.monotonic()
    if remaining<=1:raise Deadline('Total watch deadline')
    end=min(deadline,time.monotonic()+timeout_s);identities={};proc=None
    with Path(log).open('x') as stream:
        try:
            proc=subprocess.Popen(list(map(str,argv)),stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,env=env,start_new_session=True)
            remember_tree(proc.pid,identities)
            if on_identity:on_identity(identities[proc.pid])
            while proc.poll() is None:
                remember_tree(proc.pid,identities)
                if time.monotonic()>=end:
                    raise Deadline('Total watch deadline') if end==deadline else subprocess.TimeoutExpired(argv,timeout_s)
                time.sleep(min(.2,max(0,end-time.monotonic())))
            return proc.returncode
        finally:
            if proc is not None:
                remember_tree(proc.pid,identities)
                if proc.poll() is None or any(alive(v) for k,v in identities.items() if k!=proc.pid):
                    terminate_owned(identities)
                try:proc.wait(timeout=.5)
                except subprocess.TimeoutExpired:pass
