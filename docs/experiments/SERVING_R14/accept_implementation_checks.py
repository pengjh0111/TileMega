#!/usr/bin/env python3
"""Archive bounded numerical evidence without accepting synchronization yet."""
import hashlib,json,re,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def main():
    sources=[];models=[]
    for cell in ('llama_B1','qwen3_B1','llama_B16'):
        directory=HERE/'raw/LA_models'/cell
        smoke=json.loads((directory/'smoke/smoke.json').read_text())
        manifest=json.loads((directory/'plan/plan.so.plan.json').read_text())
        assert smoke['pass'] and manifest['nonpaged_la']
        log=(directory/'smoke/run.log').read_text()
        stage=re.findall(r'E2E_STAGES[^\n]*',log)
        fields={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',stage[0])}
        expected={'llama_B1':16,'qwen3_B1':28,'llama_B16':48}[cell]
        assert fields['elided']==expected
        models.append(dict(cell=cell,pass_=True,stages=fields,smoke=smoke,
            split_gemms=sum(g['split_k']>1 for g in manifest['gemms'])))
        for p in directory.rglob('*'):
            if p.is_file() and (p.suffix in ('.json','.log','.tsv') or p.name.endswith('.build_command.txt')):
                sources.append((p,str(p.relative_to(HERE))))
    for root,label in [(HERE/'raw/nonpaged_la_checks','la_checks'),
        (Path('/root/r14_work/development/docs/experiments/SERVING_R14/raw/gemv_checks'),'gemv_checks')]:
        for p in root.iterdir():
            if p.is_file() and p.suffix in ('.json','.log'):sources.append((p,label+'/'+p.name))
    raw=HERE/'raw';records=[]
    with tarfile.open(raw/'la_gemv_completed.tar.xz','w:xz') as tar:
        for p,name in sources:
            tar.add(p,arcname=name,recursive=False)
            records.append((str(p),name,hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size))
    (raw/'la_gemv_completed_manifest.tsv').write_text('source\tarchive_member\tsha256\tbytes\n'+''.join('\t'.join(map(str,row))+'\n' for row in records))
    result=dict(models=models,gemv='five architecture compile and position-coded standalone numerical checks passed',
        reliability='not accepted: 50 fresh processes still required',integration='production dispatch validation queued')
    (HERE/'results/implementation_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Archived three LA model smoke cases and standalone GEMV numeric evidence; 50-process validation pending')
if __name__=='__main__':main()
