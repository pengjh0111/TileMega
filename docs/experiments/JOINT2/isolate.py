#!/usr/bin/env python3
"""Pause this round's known occupancy-query parents during a GPU measurement.

Usage: isolate.py RECORD.json PID... -- COMMAND...
Only explicitly provided processes and their current descendants are paused;
all are resumed in finally, including when a measured command fails.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path
args=sys.argv[1:];i=args.index('--');record=Path(args[0]);parents=list(map(int,args[1:i]));command=args[i+1:]
paused=[]
try:
 for pid in parents:
  try:os.kill(pid,signal.SIGSTOP);paused.append(pid)
  except ProcessLookupError:pass
 # Parents cannot create a new child after SIGSTOP. Recursively stop existing
 # compiler/query children before measurement begins.
 while True:
  children=[]
  for row in subprocess.check_output(['ps','-eo','pid,ppid'],text=True).splitlines()[1:]:
   pid,ppid=map(int,row.split())
   if ppid in paused and pid not in paused:children.append(pid)
  if not children:break
  for pid in children:
   try:os.kill(pid,signal.SIGSTOP);paused.append(pid)
   except ProcessLookupError:pass
 info=dict(start_ns=time.time_ns(),paused=paused,command=command)
 record.write_text(json.dumps(info,indent=2)+'\n')
 result=subprocess.run(command)
 info['exit_code']=result.returncode
finally:
 for pid in reversed(paused):
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 if 'info' in locals():
  info['end_ns']=time.time_ns();record.write_text(json.dumps(info,indent=2)+'\n')
sys.exit(result.returncode)
