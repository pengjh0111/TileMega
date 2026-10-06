#!/usr/bin/env python3
"""Three fixed-geometry LA smoke cases, including partial-tile combine."""
import argparse,json,subprocess
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','smoke'));a=p.parse_args()
    jobs=json.loads((HERE/'la_builds.json').read_text());out=HERE/'raw/LA_models';out.mkdir(parents=True,exist_ok=True)
    if a.action=='build':
        subprocess.run([PYTHON,str(HERE/'builds_r14.py'),'--jobs',str(HERE/'la_builds.json'),'--out',str(out/'builds.json')],check=True)
        return
    rows=[]
    for job in jobs:
        model=job['cell'].split('_')[0];config=json.loads((ROOT/f'configs/e2e/{model}_r13_final.json').read_text());dest=out/job['cell']/'smoke'
        dest.mkdir(parents=True,exist_ok=True);so=Path(job['out'])/'plan.so'
        cmd=[PYTHON,'-m','tilemega.serving.smoke','--so',str(so),'--model',config['model']['path'],'--batch',str(job['batch']),'--steps','64','--out',str(dest)]
        write(dest/'command.json',cmd)
        with (dest/'run.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        if r.returncode:raise SystemExit(r.returncode)
        report=json.loads((dest/'smoke.json').read_text());rows.append(dict(cell=job['cell'],**report))
        if not report['pass']:raise ValueError('LA smoke failed')
    write(out/'results.json',rows);print('Nonpaged merge/combine model smoke PASS; not 50-process acceptance')
if __name__=='__main__':main()
