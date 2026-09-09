#!/usr/bin/env python3
"""A2 coordinate-order repair: frozen CUDA/fixtures, full-state round rotation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['gqa2','mha4'])
    p.add_argument('--splits', nargs='+', type=int, default=[1,2,4,8,16])
    p.add_argument('--seqs', nargs='+', type=int, default=[1,4,128,512,2048])
    p.add_argument('--pasts', nargs='+', type=int, default=[0,3,512])
    p.add_argument('--states', nargs='+', type=int, choices=[0,1], default=[0,1])
    p.add_argument('--runs', type=int, default=50)
    p.add_argument('--phase', choices=['build','run','all'], default='all')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    for part in ('bin','ptxas','logs'): (a.out/part).mkdir(parents=True,exist_ok=True)
    builds = [(m,k,s) for m in a.models for k in a.splits for s in a.states]
    def tag(config): return '%s_k%d_order%d' % config
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    if a.phase in ('build','all'):
        if (a.out/'build_manifest.json').exists() or (a.out/'correctness.tsv').exists():
            raise RuntimeError('refusing to overwrite an existing build/evidence snapshot')
        manifest = dict(arguments={**vars(a),'out':str(a.out)},sources={})
        inputs = list((repo/'include/tilemega/Codegen/tasks').glob('*'))
        inputs += [repo/'include/tilemega/Codegen/RuntimeOwnership.h']
        for path in inputs:
            if path.is_file(): manifest['sources'][str(path.relative_to(repo))] = sha(path)
        def compile_one(config):
            m,k,s = config
            source = repo/f'docs/experiments/BF16/raw_splitk/src/{m}_k{k}.cu'
            command = ['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=native',
                       '-lineinfo','-Xptxas=-v,--warn-on-spills','--expt-relaxed-constexpr',
                       '-DTILEMEGA_FP32_PARTIALS=1',f'-DTILEMEGA_CG_SPLIT_TASK_ORDER={s}',
                       *[f'-I{repo/d}' for d in ('include','third_party/cutlass/include',
                         'third_party/cutlass/tools/util/include','third_party/cutlass/test')],
                       str(source),str(repo/'build-portable/libtilemega.a'),
                       '-L/usr/local/cuda/lib64','-lcudart','-o',str(a.out/'bin'/tag(config))]
            with (a.out/'ptxas'/f'{tag(config)}.txt').open('w') as log:
                subprocess.run(command,stdout=log,stderr=log,check=True)
            print('BUILT',tag(config),flush=True)
            return dict(command=command,source_sha256=sha(source),binary_sha256=sha(a.out/'bin'/tag(config)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            manifest['builds'] = list(pool.map(compile_one,builds))
        (a.out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.phase == 'build': return
    manifest = json.loads((a.out/'build_manifest.json').read_text())
    expected = {Path(item['command'][-1]).name:item['binary_sha256'] for item in manifest['builds']}
    for config in builds:
        if sha(a.out/'bin'/tag(config)) != expected.get(tag(config)):
            raise RuntimeError('binary does not match its build manifest: '+tag(config))
    if (a.out/'correctness.tsv').exists(): raise RuntimeError('refusing to overwrite run evidence')
    if any(k.startswith('TILEMEGA_') for k in os.environ):
        raise RuntimeError('remove inherited TILEMEGA environment overrides')
    cases = [(m,k,s,seq,past) for seq in a.seqs for past in a.pasts for m,k,s in builds]
    (a.out/'status.txt').write_text('RUNNING\n')
    try:
        with (a.out/'correctness.tsv').open('w') as f:
            fields = ['round','execution_index','model','split','order','seq','past','pass',
                      'l2_mismatch','max_abs','task_refs','waits','hash_l05','hash_l1','hash_l2']
            writer = csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n'); writer.writeheader()
            for r in range(a.runs):
                shift = r%len(cases)
                for index,(m,k,s,seq,past) in enumerate(cases[shift:]+cases[:shift]):
                    binary = a.out/'bin'/tag((m,k,s))
                    fixture = repo/f'docs/experiments/SEQSCAN/raw/fixture/{m}_s{seq}_p{past}'
                    run = subprocess.run([str(binary),str(fixture)],stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,timeout=120)
                    log = a.out/'logs'/f'{tag((m,k,s))}_s{seq}_p{past}_r{r}.txt'
                    log.write_text(run.stdout)
                    def record(prefix):
                        line = next(x for x in run.stdout.splitlines() if x.startswith(prefix+' '))
                        return dict(re.findall(r'(\w+)=([^\s]+)',line))
                    schedule = record('E2E_SCHEDULE'); hashes = record('E2E_HASH')
                    error = re.search(r'l2_vs_l1_mismatch=(\d+) max_abs=(\S+)',run.stdout)
                    passed = run.returncode==0 and 'RESULT status=PASS' in run.stdout
                    writer.writerow(dict(round=r,execution_index=index,model=m,split=k,order=s,seq=seq,past=past,
                        **{'pass':int(passed)},l2_mismatch=error[1],max_abs=error[2],
                        task_refs=schedule['task_refs'],waits=schedule['waits'],
                        hash_l05=hashes['l05'],hash_l1=hashes['l1'],hash_l2=hashes['l2']))
                    f.flush()
                    # OFF is the diagnosed negative control, never a passing
                    # correctness claim. ON stops on numerical or runtime failure.
                    if s and (not passed or len(set(hashes.values()))!=1):
                        raise RuntimeError(f'repaired-path correctness gate failed: {log}')
                    if not s and run.returncode not in (0,1): run.check_returncode()
                print('ROUND',r+1,flush=True)
        (a.out/'status.txt').write_text('COMPLETE; inspect counts per state (OFF is negative control)\n')
    except BaseException:
        (a.out/'status.txt').write_text('STOPPED; inspect last row and log\n')
        raise


if __name__ == '__main__': main()
