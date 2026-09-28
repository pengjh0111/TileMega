#!/usr/bin/env python3
"""Run the frozen R12b GPU queue once; retry only external occupancy (75)."""
import argparse,csv,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    steps=json.loads(a.queue.read_text());states={}
    progress=a.out/'progress.tsv'
    if not progress.exists():progress.write_text('name\tstatus\tstart\tend\tseconds\tlast_line\n')
    for step in steps:
        name=step['name'];done=a.out/(name+'.done');failed=a.out/(name+'.failed')
        if done.exists():states[name]='done';continue
        if failed.exists():states[name]='failed';continue
        deps=step.get('after',[])
        any_deps=step.get('after_any',[])
        blocked=any(states.get(dep)!='done' for dep in deps) or any(dep not in states for dep in any_deps)
        start=stamp();begin=time.monotonic();status='skipped' if blocked else 'failed'
        log=a.out/(name+'.log');last='dependency failed' if blocked else ''
        if not blocked:
            for attempt in range(13):
                with log.open('a') as output:
                    output.write(f'ATTEMPT {attempt+1} {stamp()}\n');output.flush()
                    try:
                        code=subprocess.run(step['command'],cwd=Path(__file__).resolve().parents[3],
                            stdout=output,stderr=subprocess.STDOUT,
                            timeout=step['timeout'],env=os.environ.copy()).returncode
                    except subprocess.TimeoutExpired:code=124
                if code==75 and attempt<12:
                    time.sleep(20*60)
                    if name=='Q9' and '--resume' not in step['command'][-1]:
                        step['command'][-1]+=' --resume'
                    continue
                status='done' if code==0 else 'failed'
                lines=log.read_text(errors='replace').splitlines()
                last=lines[-1] if lines else f'exit={code}'
                break
        (done if status=='done' else failed).write_text(f'{status}\t{last}\n')
        end=stamp();seconds=round(time.monotonic()-begin,2)
        with progress.open('a',newline='') as stream:
            csv.writer(stream,delimiter='\t').writerow([name,status,start,end,seconds,last[:300]])
        print(f'{name}\t{status}\t{seconds}s',flush=True)
        states[name]=status
    print('QUEUE_COMPLETE',json.dumps(states,sort_keys=True),flush=True)

if __name__=='__main__':main()
