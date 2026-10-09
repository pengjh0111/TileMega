"""Bound a compile/measurement and all process groups in its own session."""
import os
import signal
import subprocess
import time


def run_until(argv,*,deadline=None,**kwargs):
    if deadline is None:return subprocess.run(argv,**kwargs).returncode
    remaining=deadline-time.monotonic()
    if remaining<=0:return 124
    with subprocess.Popen(argv,start_new_session=True,**kwargs) as process:
        try:return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            # GNU timeout starts extra process groups in this same session.
            # Kill the session rather than leaving its CUDA child holding locks.
            subprocess.run(['pkill','-TERM','-s',str(process.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
            subprocess.run(['pkill','-KILL','-s',str(process.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process.wait()
            return 124
