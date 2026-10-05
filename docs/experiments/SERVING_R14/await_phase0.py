#!/usr/bin/env python3
"""Block on process completion events, then gate timing on correctness evidence."""
import argparse,json,os,select,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def wait_process(pid_file,done,deadline):
    if done.exists():return
    pid=int(pid_file.read_text());fd=os.pidfd_open(pid)
    try:
        poll=select.poll();poll.register(fd,select.POLLIN)
        if not poll.poll(max(0,int((deadline-time.monotonic())*1000))):
            raise TimeoutError(f'correctness process {pid} did not finish')
    finally:os.close(fd)
    if not done.exists():raise RuntimeError(f'correctness process ended without {done}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--timeout-s',type=int,default=5400);a=p.parse_args()
    end=time.monotonic()+a.timeout_s
    wait_process(Path('/root/r14_work/flow/pid'),HERE/'raw/FX24/run.done',end)
    result=json.loads((HERE/'raw/FX24/after/results.json').read_text())
    if {r['name'] for r in result}!={'fixed','joint'} or any(r['exit_code'] for r in result):
        raise RuntimeError('FX-24 search-only reproduction failed')
    wait_process(Path('/root/r14_work/phase0/pid'),HERE/'raw/phase0_checks/run.done',end)
    checks={line.split('\t')[0]:int(line.split('\t')[1]) for line in (HERE/'raw/phase0_checks/progress.tsv').read_text().splitlines()}
    required={'identity','selection','build','host','attention_layout','paged_gemm',*(f'compile_sm{arch}' for arch in (80,90,100,120))}
    if not required<=checks.keys() or any(checks[name] for name in required):raise RuntimeError('Phase-0 checks incomplete')
    print('Phase-0 correctness prerequisites passed; trace diagnosis may run')
if __name__=='__main__':main()
