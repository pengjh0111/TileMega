#!/usr/bin/env python3
"""R6 fresh-process collection with per-process hardware context.

The timing protocol, fixture and correctness tolerances match JOINT/measure.py.
Telemetry includes pre/post snapshots and a 100 ms external hardware sampler.
The sampler adds no instructions to the measured kernels.
No sample is filtered based on telemetry or timing.
"""
import fcntl,json,os,subprocess,time
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure as r5

def context():
 command=['nvidia-smi','--query-gpu=timestamp,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits']
 r=subprocess.run(command,capture_output=True,text=True,timeout=10)
 return dict(command=command,exit_code=r.returncode,output=r.stdout.strip(),error=r.stderr.strip(),loadavg=list(os.getloadavg()))

def run(cell,m,seq,arm,folder,round_,order,session,dump=False,past=3,phase=False):
 folder.mkdir(parents=True,exist_ok=True);log=folder/f'r{round_}.log'
 if log.exists():raise RuntimeError('refusing overwrite '+str(log))
 binary=cell/'bin'/arm;env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
 if 'correctness' in folder.parts or 'seqscan' in folder.parts:env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
 if dump:env.update(TILEMEGA_TRACE_V2='1',TILEMEGA_TRACE_V2_OUT=str((folder/'dump').resolve()),TILEMEGA_MODEL_NAME=m,TILEMEGA_PLACEMENT_BASE_DUMP='1',TILEMEGA_GLOBALTIMER_NS=os.getenv('PHASE_TICK_NS','1024'))
 if phase:env.update(TILEMEGA_TRACE_PHASE='1',TILEMEGA_TRACE_PHASE_OUT=str((folder/'dump').resolve()),TILEMEGA_MODEL_NAME=m,TILEMEGA_GLOBALTIMER_NS=os.getenv('PHASE_TICK_NS','1024'))
 command=[str(binary.resolve()),str(r5.fixture(m,seq,past))]
 with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);before=context();start=time.time_ns()
  sampler_command=before['command']+['-lms','100']
  with log.with_suffix('.gpu.csv').open('w') as telemetry:
   sampler=subprocess.Popen(sampler_command,stdout=telemetry,stderr=subprocess.DEVNULL,text=True)
   try:
    r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300);status=r.returncode;text=r.stdout+r.stderr
   except subprocess.TimeoutExpired as e:status=124;text=str(e)
   finally:
    sampler.terminate()
    try:sampler.wait(timeout=5)
    except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
  elapsed=time.time_ns()-start;after=context()
 log.write_text(text)
 log.with_suffix('.json').write_text(json.dumps(dict(command=command,environment={k:v for k,v in env.items() if k.startswith('TILEMEGA_')},exit_code=status,session=session,round=round_,order=order,started_ns=start,elapsed_ns=elapsed,binary_sha256=r5.sha(binary),telemetry_before=before,telemetry_after=after,sampler_command=sampler_command))+'\n')
 if status or 'RESULT status=PASS' not in text:raise RuntimeError('correctness: '+str(log))
