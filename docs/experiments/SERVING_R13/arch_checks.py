#!/usr/bin/env python3
"""Compile the same serving sources and loadbench for all required architectures."""
import argparse,hashlib,json,re,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def command(binary,arch,output,kind,extra=()):
    original=shlex.split(Path(str(binary)+'.build_command.txt').read_text())
    source=next(x for x in original if x.endswith('.cu'))
    flags=[x for x in original if x.startswith(('-I','-D')) or x=='--expt-relaxed-constexpr']
    return [original[0],'-std=c++17','-O3',f'-arch=sm_{arch}',*flags,*extra,
            '-Xptxas=-v','--'+kind,source,'-o',str(output)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--nonpaged',type=Path,required=True)
    p.add_argument('--paged',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for arch in (80,89,90,100,120):
        for label,binary in [('nonpaged',a.nonpaged),('paged',a.paged),('loadbench',None)]:
            for kind in ('cubin','ptx') if label=='nonpaged' and arch in (90,120) else ('cubin',):
                for trigger in (0,1) if kind=='ptx' else (0,):
                    dest=a.out/f'{label}_sm{arch}_trigger{trigger}.{kind}'
                    if binary:
                        cmd=command(binary,arch,dest,kind,[f'-DTILEMEGA_PDL_TRIGGER={trigger}'])
                    else:
                        cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3',f'-arch=sm_{arch}',
                             '--expt-relaxed-constexpr','-I'+str(ROOT/'include'),
                             '-I'+str(ROOT/'third_party/cutlass/include'),'-Xptxas=-v',
                             '--'+kind,str(ROOT/'tools/experimental/loadbench/main.cu'),'-o',str(dest)]
                    (Path(str(dest)+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
                    with Path(str(dest)+'.log').open('w') as f:
                        code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
                    row=dict(arch=arch,label=label,kind=kind,trigger=trigger,exit_code=code,path=str(dest))
                    if code==0:
                        row['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
                        if kind=='cubin':
                            sass=subprocess.check_output(['cuobjdump','-sass',str(dest)],text=True)
                            row['fp64_instructions']=len(re.findall(r'\bD(?:ADD|MUL|FMA|SETP)\b',sass))
                            (Path(str(dest)+'.sass')).write_text(sass)
                            if label=='nonpaged':
                                # Audit list; classification must be written into the report,
                                # never automatically assumed immutable by the tool.
                                (Path(str(dest)+'.nc_loads.txt')).write_text('\n'.join(x for x in sass.splitlines() if 'LDG.E.CONSTANT' in x)+'\n')
                        else:
                            row['position_exit_code']=subprocess.run(['python3',str(HERE/'ptx_pdl_check.py'),
                                str(dest),'--trigger',str(trigger),'--out',str(dest)+'.positions.json']).returncode
                    rows.append(row)
                    print(label,arch,kind,trigger,code,flush=True)
    (a.out/'arch_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('arch checks: '+str(sum(r['exit_code']==0 for r in rows))+'/'+str(len(rows)))
if __name__=='__main__':main()
