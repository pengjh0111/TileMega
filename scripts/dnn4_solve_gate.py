#!/usr/bin/env python3
"""Check full20 F4 measured-key coverage and emit each predicted top-4 on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--bridges',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=ROOT/'runs/dnn4/build/main/tools/tilemega')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.target=args.target.resolve(strict=True);args.binary=args.binary.resolve(strict=True)
    args.bridges=args.bridges.resolve(strict=True);args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    inputs={str(path):sha(path) for path in (args.target,args.binary,Path(__file__))}
    rows=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TILEMEGA_DEVICE_INDEX='1',
             TILEMEGA_FORWARD_CONSTANTS_FILE=str(args.target))
    def invoke(command,log):
        with log.open('w') as f:
            return subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
    for model in ('resnet18','resnet50','vgg16','bert'):
        for batch in (1,2,4,8,16):
            name=f'{model}_b{batch}';folder=args.out/name;folder.mkdir()
            bridge=args.bridges/(name+'.bridge.json');inputs[str(bridge)]=sha(bridge)
            source=folder/'winner.cu'
            base=[str(args.binary),'compile',str(bridge),str(source),'--emit','serving',
                  '--serving','forward','--frontend','dnn','--batch',str(batch),'--abi','f3']
            exit_code=invoke(base+['--solve',str(args.target),'--selection','predicted','--measure-top','4'],folder/'solve.log')
            row=dict(model=model,batch=batch,solve_exit=exit_code,status='failed',candidates=[])
            if exit_code==0:
                receipt=json.loads(Path(str(source)+'.search/receipt.json').read_text())
                if not 1<=len(receipt['ranked'])<=4 or receipt['target_sha256']!=inputs[str(args.target)]:
                    raise ValueError('search did not bind four candidates to the measured target')
                for rank in range(len(receipt['ranked'])):
                    output=folder/f'candidate-{rank}.cu'
                    coords=Path(str(source)+f'.search/candidate-{rank}.tmexec.json')
                    command=[*base];command[3]=str(output)
                    code=invoke(command+['--tmexec-config',str(coords),'--runtime-target',str(args.target)],folder/f'candidate-{rank}.log')
                    item=dict(rank=rank,exit_code=code,coordinates=str(coords),coordinates_sha256=sha(coords))
                    if code==0:item.update(source=str(output),sha256=sha(output))
                    row['candidates'].append(item)
                row['selfcheck']=receipt['ranked'][0]['prediction']['search_receipt']['f4_candidate_selfcheck']
                row['sensitivity']=receipt['ranked'][0]['prediction']['search_receipt']['f4_coordinate_sensitivity']
                row.update(status='passed' if all(x['exit_code']==0 for x in row['candidates']) else 'failed',
                           predicted_ns=receipt['ranked'][0]['prediction']['forward_ns'],receipt=str(Path(str(source)+'.search/receipt.json')))
            rows.append(row)
            result=dict(evidence='verified',protocol='CPU solve/codegen only; native compile and P-SEL remain separate',inputs=inputs,rows=rows,
                        failures=sum(r['status']!='passed' for r in rows),status='partial')
            (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
    result['coordinate_sensitivity']={field:any(row.get('sensitivity',{}).get('fields',{}).get(field,False) for row in rows) for field in ['tile_m','tile_n','tk','split_k','slots','store','mc']}
    result['candidate_selfcheck_failed_cells']=[row['model']+'_b'+str(row['batch']) for row in rows if not row.get('selfcheck',{}).get('passed',False)]
    if any(sha(Path(path))!=digest for path,digest in inputs.items()):
        raise ValueError('CPU solve inputs changed')
    result['status']='passed' if result['failures']==0 else 'failed'
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return int(result['failures']!=0)


if __name__=='__main__':
    raise SystemExit(main())
