"""Measure three independent resident-grid TP constant processes per scope."""
import argparse,hashlib,json,os,pathlib,statistics,subprocess,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--processes',type=int,choices=(2,3),default=3);p.add_argument('--round-start',type=int,default=0);a=p.parse_args()
if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':raise RuntimeError('group guard is required')
build=json.loads((a.build/'BUILD.json').read_text());source=json.loads((a.build.parent/'SOURCE.json').read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if sha(build['binary'])!=build['binary_sha256']:raise ValueError('constant binary changed')
for name,digest in source['files'].items():
 if sha(a.build.parent/'frozen'/name)!=digest:raise ValueError('frozen constant source changed')
if len(os.environ['TILEMEGA_DEVICE_GROUP'].split(','))!=build['tp']:raise ValueError('constant world differs from group')
a.out.mkdir(parents=True,exist_ok=False);all_rows=[]
(a.out/'measure_driver.py').write_bytes(pathlib.Path(__file__).read_bytes())
for fresh in range(a.round_start,a.round_start+a.processes):
 out=a.out/f'process{fresh}';out.mkdir();started=time.monotonic()
 with(out/'stdout.jsonl').open('w')as stdout,(out/'stderr.txt').open('w')as stderr:
  child=subprocess.Popen([str(pathlib.Path(build['binary']).resolve())],stdout=stdout,stderr=stderr);code=child.wait()
 rows=[json.loads(x)for x in(out/'stdout.jsonl').read_text().splitlines()if x.startswith('{')]
 record=dict(PID=child.pid,fresh_process=fresh,returncode=code,elapsed_s=time.monotonic()-started,build=build,source_digest=source['source_digest'],gpu_indices=os.environ['TILEMEGA_DEVICE_GROUP'],records=rows,driver_sha256=sha(__file__))
 (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
 if code or len(rows)!=9 or not all(x['pass']for x in rows):raise RuntimeError('constant comparison failed; preserve process evidence')
 all_rows.append(rows);print('constant process',fresh,'passed',flush=True)
summary=[]
for i,first in enumerate(all_rows[0]):
 values=[rows[i]['median_ns']for rows in all_rows];m=statistics.median(values)
 summary.append(dict(key=dict(tp=first['tp'],b=first['b'],stage=first['stage'],tn=first['tn']),median_ns=m,p90_ns=statistics.median(rows[i]['p90_ns']for rows in all_rows),spread=(max(values)-min(values))/abs(m)if m else float('inf'),processes=a.processes,rounds=values,source_digest=source['source_digest']))
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
