#!/usr/bin/env python3
"""Increase parallelism only for a validated, not-yet-admitted matrix arm.

Preserves old launch metadata. Refuses to interrupt a solver or replace any
generated artifact. All geometry candidates, passes and compiler bytes stay
unchanged; only worker count and admission slots are explicit CLI changes.
"""
import argparse
import hashlib
import json
import pathlib
import signal
import subprocess
import time
import os
import fcntl

E=pathlib.Path(__file__).resolve().parent
R=E.parents[2]


def promote(args):
    if not 2<=args.jobs<=64 or not 1<=args.slots<=16:raise ValueError('invalid concurrency')
    directory=E/'matrix'/f'{args.model}_s{args.seq}'/f'skeleton-k{args.k_base}'
    if (directory/'solver_admission.json').exists() or list(directory.glob('selected*')):
        raise RuntimeError('this arm already began solving; refusing to interrupt it')
    metadata=directory/'launch.command.json'
    original=json.loads(metadata.read_text());pid=original['pid'];command=original['command']
    if command[:2]!=['python3',str(E/'run_matrix.py')]:raise ValueError('unexpected queued executable')
    if command[command.index('--out')+1]!=str(directory):raise ValueError('wrong output directory')
    actual=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if actual!=command:raise RuntimeError('PID no longer identifies the recorded queue runner')
    # Stop the queue-only parent before the last check so admission cannot
    # race cancellation. A runner that acquired a slot is resumed, not killed.
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(50):
            if 'State:\tT' in pathlib.Path(f'/proc/{pid}/status').read_text():break
            time.sleep(.1)
        else:raise RuntimeError('could not freeze queue runner for the admission check')
        children=pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        if (directory/'solver_admission.json').exists() or children:
            raise RuntimeError('runner entered admission; leaving it alive')
    except Exception:
        os.kill(pid,signal.SIGCONT)
        raise
    archive=directory/'queued_launch_history'/str(time.time_ns())
    try:
        archive.mkdir(parents=True)
        metadata.rename(archive/metadata.name)
        os.kill(pid,signal.SIGTERM)
    finally:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    for _ in range(50):
        status=pathlib.Path(f'/proc/{pid}/status')
        if not status.exists() or 'State:\tZ' in status.read_text():break
        time.sleep(.1)
    else:raise RuntimeError('queued runner did not exit; refusing a duplicate launch')
    if (directory/'solver_admission.json').exists():raise RuntimeError('admission raced cancellation; inspect before proceeding')
    if (directory/'runner.log').exists():(directory/'runner.log').rename(archive/'runner.log')
    for option,value in [('--search-jobs',args.jobs),('--solver-slots',args.slots),('--solver-slot',args.slots-1)]:
        if option in command:command[command.index(option)+1]=str(value)
        else:command += [option,str(value)]
    compiler=pathlib.Path(command[command.index('--compiler')+1])
    with (directory/'runner.log').open('w') as log:
        process=subprocess.Popen(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(command=command,pid=process.pid,started_ns=time.time_ns(),
               compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
               previous_launch=str(archive/'launch.command.json'),
               head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
               compiler_snapshot_head=original.get('compiler_snapshot_head',original.get('head')))
    metadata.write_text(json.dumps(value,indent=2)+'\n')
    print('PROMOTED',directory,'pid',process.pid,'jobs',args.jobs,'slots',args.slots,flush=True)
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['llama','qwen3'])
    parser.add_argument('--seq',type=int,choices=[1,4,16,64])
    parser.add_argument('--k-base',choices=['4','8','16','W'])
    parser.add_argument('--jobs',type=int,required=True)
    parser.add_argument('--slots',type=int,required=True)
    parser.add_argument('--drain-queue',action='store_true',help='reuse the last admission slot for queued arms, one at a time')
    args=parser.parse_args()
    if not 2<=args.jobs<=64 or not 1<=args.slots<=16:raise ValueError('invalid concurrency')
    if not args.drain_queue:
        if args.model is None or args.seq is None or args.k_base is None:parser.error('single promotion requires model, seq and k-base')
        promote(args);return
    supervisor=open('/tmp/tilemega-r9-promotion-supervisor.lock','a')
    fcntl.flock(supervisor,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state=E/'matrix_progress.json'
        if state.exists() and not json.loads(state.read_text())['pending']:return
        with open(f'/tmp/tilemega-r9-solver-{args.slots-1}.lock','a') as probe:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(15);continue
        promoted=None
        for directory in sorted((E/'matrix').glob('*/skeleton-*')):
            if not (directory/'launch.command.json').exists() or (directory/'solver_admission.json').exists():continue
            model,seq=directory.parent.name.rsplit('_s',1)
            try:
                promoted=promote(argparse.Namespace(model=model,seq=int(seq),k_base=directory.name.removeprefix('skeleton-k'),jobs=args.jobs,slots=args.slots))
            except (RuntimeError,ProcessLookupError,FileNotFoundError) as error:
                print('SKIP',directory,str(error),flush=True);continue
            break
        if promoted:
            # Do not launch another arm before the new process has obtained
            # the dedicated slot, even if process startup is temporarily slow.
            while not (promoted/'solver_admission.json').exists():
                pid=json.loads((promoted/'launch.command.json').read_text())['pid']
                if not pathlib.Path(f'/proc/{pid}').exists():raise RuntimeError('promoted runner exited before admission')
                time.sleep(1)
        time.sleep(15)


if __name__=='__main__':main()
