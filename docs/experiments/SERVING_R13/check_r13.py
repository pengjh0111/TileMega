#!/usr/bin/env python3
"""R13 B5: real-weight executor checks and teacher-forced geometry checks."""
import argparse,json,subprocess,sys
from pathlib import Path
from phase_b_r13 import arms,CELLS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',choices=CELLS,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checks',choices=('all','C-1','C-2'),default='all')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows={r['label']:r for r in arms('B4')[a.cell]};rows.update({r['label']:r for r in arms('B3')[a.cell]})
    result=[];model,b=a.cell.split('_B');prompt=ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
    for label in (('B0','PR_L2','PS_L2','PSA_L2') if a.checks in ('all','C-2') else ()):
        arm=rows.get(label)
        if not arm or not arm['available']:
            result.append(dict(label=label,check='C-2',status='missing'));continue
        folder=a.out/label;folder.mkdir(exist_ok=True)
        cmd=[sys.executable,'-m','tilemega.serving.check_modes','--model',arm['model_path'],
            '--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',str(prompt),
            '--batch',b,'--steps','1024','--out',str(folder)]
        (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (folder/'stdout.log').open('w') as f:code=subprocess.run(['timeout','1200']+cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        if code==75:return 75
        result.append(dict(label=label,check='C-2',exit_code=code))
    for label in (('B0h','PSA_L2') if a.checks in ('all','C-1') else ()):
        arm=rows.get(label)
        if not arm or not arm['available']:
            result.append(dict(label=label,check='C-1',status='missing'));continue
        files=list((HERE/'raw').glob(f'B4_{a.cell}_*/{a.cell}/{label}/round*/tokens_N1024_run1.json'))
        if not files:
            result.append(dict(label=label,check='C-1',status='no measured token file'));continue
        folder=a.out/(label+'-HF');folder.mkdir(exist_ok=True)
        cmd=[sys.executable,'-m','tilemega.serving.hf_check','--model',arm['model_path'],
             '--prompt-ids',str(prompt),'--generated',str(files[0]),'--skip-free-greedy','--out',str(folder/'report.json')]
        (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (folder/'stdout.log').open('w') as f:code=subprocess.run(['timeout','1200']+cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        if code==75:return 75
        result.append(dict(label=label,check='C-1',exit_code=code))
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    passed=sum(r.get('exit_code')==0 for r in result);print(f'B5 correctness {a.cell}: {passed}/{len(result)}')
    return int(passed!=len(result))
if __name__=='__main__':raise SystemExit(main())
