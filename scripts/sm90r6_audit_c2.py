#!/usr/bin/env python3
"""Audit accepted R6 C2 processes against guards, identities and full token arrays."""
import argparse
from pathlib import Path
import hashlib,json,time


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    raw=root/'runs/sm90r6'
    patterns=['a0/c2-v1/*/result.json','final-reference-v1/*/c2/result.json','ablation/correctness-v1/*/*/c2/result.json','ablation/correctness-retry-v1/*/*/c2/result.json','ablation/correctness-retry-v1/*/AB4b/AB4c/c2/result.json','ab5/pair-v1/*/*-c2/result.json','i3/qwen-moeb16-v1/TN*/c2/result.json','i3/qwen-denseb1-fallback-v1/TN*/c2/result.json']
    rows=[];artifacts={}
    for pattern in patterns:
     for path in sorted(raw.glob(pattern)):
      data=read(path);processes=data['processes'];assert data['steps']==1024
      if not processes:continue
      assert data['passed']==(len(processes)>=5)
      for value in data['artifacts'].values():
       binary=value['path']
       if binary not in artifacts:artifacts[binary]=sha(binary)
       assert artifacts[binary]==value['sha256']==read(binary+'.identity.json')['binary_sha256']
      dirs=set();pgids=set();checkpoints=set();tokens=set();gpu_set=set();hashes={str(path):sha(path)}
      prompt='olmo' if data['model'].startswith('olmo') else 'llama3' if data['model'].startswith('llama') else 'qwen'
      expected_prompt=sha(root/f'configs/prompts/{prompt}_isl64.json')
      for item in processes:
       directory=Path(item['directory']);assert str(directory) not in dirs;dirs.add(str(directory))
       pgid=(directory/'guard/pgid').read_text().strip();assert pgid.isdigit() and pgid not in pgids;pgids.add(pgid)
       assert item['gpu'] in (0,1,2,3,4,5,7);gpu_set.add(item['gpu'])
       assert read(directory/'result.json')['returncode']==0 and read(directory/'guard/guard_result.json')['code']==0
       checks=read(directory/'checks/result.json')
       assert checks['passed'] and checks['batch']==data['batch'] and checks['steps']==1024
       assert int(checks['gpu'])==item['gpu'] and checks['prompt_sha256']==expected_prompt
       assert checks['checkpoint']==item['checkpoint']==read(directory/'checkpoint/random_checkpoint.json')['checkpoint_sha256']
       checkpoints.add(item['checkpoint']);tokens.add(item['tokens_sha256'])
       expected={('candidate',False),('candidate',True)}
       if data['artifacts']['reference']['path']!=data['artifacts']['candidate']['path']:expected|={('reference',False),('reference',True)}
       assert {(v['implementation'],v['decode_loop_used']) for v in checks['checks']}==expected
       reference=None
       for check in checks['checks']:
        assert check['passed'] and check['binary_sha256']==data['artifacts'][check['implementation']]['sha256']
        token_path=directory/'checks'/f"{check['implementation']}-loop{int(check['decode_loop_used'])}.tokens.json"
        assert sha(token_path)==check['tokens_sha256']==item['tokens_sha256']
        value=read(token_path);assert len(value)==data['batch'] and all(len(v)==1024 for v in value)
        if reference is not None:assert value==reference
        reference=value
       for suffix in ('result.json','checks/result.json','guard/guard_result.json','guard/pgid','command.json'):
        p=directory/suffix;hashes[str(p)]=sha(p)
      assert len(checkpoints)==len(tokens)==1
      rows.append(dict(source=str(path),passed=data['passed'],processes=len(processes),model=data['model'],batch=data['batch'],gpus=sorted(gpu_set),pgids=sorted(pgids),hashes=hashes))
    value=dict(verified=True,scope='accepted C2 raw guards, process groups, token arrays, prompt/checkpoint and binary identity; partial cells are not complete C2 gates',at=time.time(),complete=sum(x['passed'] for x in rows),accepted_processes=sum(x['processes'] for x in rows),rows=rows)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();value=audit(args.root)
    value['audit_source_sha256']=sha(__file__)
    args.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({key:item for key,item in value.items() if key!='rows'}))
