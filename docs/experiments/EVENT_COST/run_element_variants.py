#!/usr/bin/env python3
"""A2 independent element-ownership and runtime-variant projection audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(text, prefix):
    rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in text.splitlines() if line.startswith(prefix+' ')]
    if len(rows) != 1:
        raise RuntimeError(f'missing/ambiguous {prefix}')
    return rows[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runs', type=int, default=50)
    p.add_argument('--phase', choices=['build','run','project','verify','all'], default='all')
    a = p.parse_args()
    if a.runs < 50:
        raise ValueError('at least 50 fresh processes per cell are required')
    repo = Path(__file__).resolve().parents[3]
    out = a.out.resolve()
    compiler = repo/'build-portable/tools/tilemega-compile'
    projection = repo/'build-portable/tools/tilemega-runtime-projection'
    builds = [(m,k) for m in ('gqa2','mha4') for k in (1,2,4,8,16)]
    if any(k.startswith('TILEMEGA_') for k in os.environ):
        raise RuntimeError('remove inherited TILEMEGA overrides')
    if a.phase in ('build','all'):
        out.mkdir(parents=True,exist_ok=False)
        for part in ('bin','src','plan','ptxas','logs','projection'):
            (out/part).mkdir()
        manifest = dict(compiler_sha256=sha(compiler),projection_sha256=sha(projection),runs=a.runs,
                        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
        (out/'source.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=repo))
        def build(config):
            model,split = config
            tag = f'{model}_k{split}'
            plan = dict(schema='tilemega.runtime_variants.v1',variants=[])
            for begin,end,tile_m in ((1,128,128),(129,2048,64)):
                plan['variants'].append(dict(seq_begin=begin,seq_end=end,
                    rope_tile_per_block=False,kv_tile_per_block=False,
                    activation_tile_per_block=False,combiner_tile_per_block=False,
                    uniform=dict(tile_m=tile_m,tile_n=128,tile_k=16,stages=3,split_k=split)))
            plan_path = out/'plan'/f'{tag}.json'
            source = out/'src'/f'{tag}.cu'
            binary = out/'bin'/tag
            plan_path.write_text(json.dumps(plan,indent=2)+'\n')
            export = repo/f'docs/experiments/SEQSCAN/raw/export/{model}.json'
            with (out/'ptxas'/f'{tag}_codegen.txt').open('w') as log:
                subprocess.run([str(compiler),str(export),str(source),'--variants',str(plan_path)],
                               stdout=log,stderr=log,check=True)
            command = ['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=native','-lineinfo',
                       '-Xptxas=-v,--warn-on-spills','--expt-relaxed-constexpr',
                       '-DTILEMEGA_FP32_PARTIALS=1','-DTILEMEGA_CG_SPLIT_TASK_ORDER=1',
                       *[f'-I{repo/d}' for d in ('include','third_party/cutlass/include',
                         'third_party/cutlass/tools/util/include','third_party/cutlass/test')],
                       str(source),str(repo/'build-portable/libtilemega.a'),
                       '-L/usr/local/cuda/lib64','-lcudart','-o',str(binary)]
            with (out/'ptxas'/f'{tag}.txt').open('w') as log:
                subprocess.run(command,stdout=log,stderr=log,check=True)
            print('BUILT',tag,flush=True)
            return dict(tag=tag,command=command,binary_sha256=sha(binary),source_sha256=sha(source),
                        plan_sha256=sha(plan_path),export_sha256=sha(export))
        with ThreadPoolExecutor(max_workers=4) as pool:
            manifest['builds'] = list(pool.map(build,builds))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (out/'status.txt').write_text('BUILT\n')
    if a.phase == 'build':
        return
    manifest = json.loads((out/'manifest.json').read_text())
    for item in manifest['builds']:
        if sha(out/'bin'/item['tag']) != item['binary_sha256']:
            raise RuntimeError('frozen binary changed')
    if a.phase in ('run','all'):
        if (out/'correctness.tsv').exists():
            raise RuntimeError('refusing to overwrite correctness evidence')
        cases = [(m,k,s,past) for s in (1,4,128,512,2048) for past in (0,3,512) for m,k in builds]
        try:
            with (out/'correctness.tsv').open('w') as stream:
                fields = ['round','execution_index','model','split','seq','past','grid','threads',
                          'task_refs','waits','max_worker_tasks','hash','status','log_sha256']
                writer = csv.DictWriter(stream,fields,delimiter='\t'); writer.writeheader()
                for r in range(a.runs):
                    shift=r%len(cases)
                    for index,(model,split,seq,past) in enumerate(cases[shift:]+cases[:shift]):
                        fixture=repo/f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p{past}'
                        run=subprocess.run([str(out/'bin'/f'{model}_k{split}'),str(fixture)],
                            capture_output=True,text=True,timeout=120)
                        text=run.stdout+run.stderr
                        log=out/'logs'/f'{model}_k{split}_s{seq}_p{past}_r{r}.txt'
                        log.write_text(text)
                        if run.returncode or 'RESULT status=PASS' not in text:
                            raise RuntimeError(f'correctness gate failed: {log}')
                        resource=record(text,'E2E_RESOURCE'); schedule=record(text,'E2E_SCHEDULE')
                        hashes=record(text,'E2E_HASH')
                        if len(set(hashes.values()))!=1:
                            raise RuntimeError(f'cross-level hash mismatch: {log}')
                        writer.writerow(dict(round=r,execution_index=index,model=model,split=split,
                            seq=seq,past=past,grid=resource['grid'],threads=resource['block'],
                            task_refs=schedule['task_refs'],waits=schedule['waits'],
                            max_worker_tasks=schedule.get('max_worker_tasks',''),hash=hashes['l2'],
                            status='PASS',log_sha256=sha(log)))
                        stream.flush()
                    (out/'status.txt').write_text(f'RUNNING rounds={r+1}/{a.runs}\n')
                    print('ROUND',r+1,flush=True)
            (out/'status.txt').write_text(f'CORRECTNESS PASS {a.runs*len(cases)} processes; projection pending\n')
        except BaseException:
            (out/'status.txt').write_text('STOP: inspect the last preserved log\n')
            raise
    if a.phase not in ('project','verify','all'):
        return
    if sha(projection)!=manifest['projection_sha256']:
        raise RuntimeError('projection binary changed; freeze and record a new verification manifest')
    rows=list(csv.DictReader((out/'correctness.tsv').open(),delimiter='\t'))
    if (a.phase!='project' and len(rows)!=150*a.runs) or not rows:
        raise RuntimeError('incomplete correctness matrix')
    cases=[(m,k,s,past) for s in (1,4,128,512,2048) for past in (0,3,512) for m,k in builds]
    seen=set()
    for row in rows:
        r,index=int(row['round']),int(row['execution_index'])
        model,split,seq,past=row['model'],int(row['split']),int(row['seq']),int(row['past'])
        if not (0<=r<a.runs and 0<=index<len(cases)) or (r,index) in seen:
            raise RuntimeError('missing or repeated fresh-process cell')
        seen.add((r,index))
        if cases[(index+r%len(cases))%len(cases)]!=(model,split,seq,past):
            raise RuntimeError('state/fixture rotation does not match the declared order')
        log=out/'logs'/f'{model}_k{split}_s{seq}_p{past}_r{r}.txt'
        if row['status']!='PASS' or sha(log)!=row['log_sha256']:
            raise RuntimeError('correctness evidence changed')
        text=log.read_text()
        schedule=record(text,'E2E_SCHEDULE'); resource=record(text,'E2E_RESOURCE')
        hashes=record(text,'E2E_HASH')
        if 'RESULT status=PASS' not in text or len(set(hashes.values()))!=1 or hashes['l2']!=row['hash']:
            raise RuntimeError('cross-level correctness log disagrees with its summary')
        if any(schedule[field]!=row[field] for field in ('task_refs','waits')) or resource['grid']!=row['grid'] or resource['block']!=row['threads']:
            raise RuntimeError('raw schedule/resource log disagrees with its summary')
    queries={}
    comparisons=0
    for row in rows:
        model,split,seq,past=row['model'],int(row['split']),int(row['seq']),int(row['past'])
        tile_m=128 if seq<=128 else 64
        key=(model,split,tile_m,row['grid'],row['threads'])
        if key not in queries:
            command=[str(projection),str(repo/f'docs/experiments/SEQSCAN/raw/export/{model}.json'),
                row['grid'],row['threads'],'1','symbolic',str(tile_m),'128','16','3',str(split),
                'element','1','1','1']
            tag=f'{model}_k{split}_m{tile_m}_g{row["grid"]}_b{row["threads"]}_audit1'
            stdout=out/'projection'/f'{tag}.tsv'
            stderr=out/'projection'/f'{tag}.txt'
            receipt=out/'projection'/f'{tag}.json'
            if receipt.exists():
                previous=json.loads(receipt.read_text())
                if previous['command']!=command or previous.get('environment')!={'TILEMEGA_ISL_AUDIT':'1'} or previous['binary_sha256']!=sha(projection) or previous['stdout_sha256']!=sha(stdout) or previous['stderr_sha256']!=sha(stderr):
                    raise RuntimeError('frozen symbolic projection receipt changed')
                output,error=stdout.read_text(),stderr.read_text()
            else:
                run=subprocess.run(command,capture_output=True,text=True,timeout=600,
                                   env=dict(os.environ,TILEMEGA_ISL_AUDIT='1'))
                with stdout.open('x') as f:
                    f.write(run.stdout)
                with stderr.open('x') as f:
                    f.write(run.stderr)
                run.check_returncode()
                output,error=run.stdout,run.stderr
                with receipt.open('x') as f:
                    json.dump(dict(command=command,environment={'TILEMEGA_ISL_AUDIT':'1'},binary_sha256=sha(projection),
                        stdout_sha256=sha(stdout),stderr_sha256=sha(stderr)),f,indent=2)
            if 'ISL_CONTEXT remaining=0' not in error:
                raise RuntimeError('missing explicit zero-reference evidence')
            queries[key]={(int(r['seq']),int(r['past'])):r
                          for r in csv.DictReader(io.StringIO(output),delimiter='\t')}
        q=queries[key][seq,past]
        for field in ('task_refs','waits'):
            if int(q[field])!=int(row[field]):
                raise RuntimeError(f'projection mismatch: {key}, S={seq}, past={past}, {field}')
            comparisons+=1
    verification='prefix_verification.json' if a.phase=='project' else 'verification.json'
    with (out/verification).open('x') as f:
        f.write(json.dumps(dict(processes=len(rows),
        projection_equalities=comparisons,queries=len(queries),status='PASS'),indent=2)+'\n')
    if a.phase!='project':
        (out/'status.txt').write_text(f'PASS processes={len(rows)} exact_counters={comparisons}\n')


if __name__ == '__main__':
    main()
