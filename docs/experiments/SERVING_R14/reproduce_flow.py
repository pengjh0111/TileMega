#!/usr/bin/env python3
"""Replay the archived sm120 fixed rejection and the paged Llama search on CPU."""
import argparse,copy,gzip,json,os,subprocess,sys
from pathlib import Path
from pin_case import pin
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
TARGET=ROOT/'docs/experiments/SERVING_R13_SM120/target_sm120.json'

def commands(compiler,out):
    jobs=json.loads((ROOT/'docs/experiments/SERVING_R13/builds_a.json').read_text())
    donor=next(j for j in jobs if j['cell']=='llama_B16' and j['label']=='N-R12b-noWD')
    original=json.loads((HERE/'raw/FX24/original/fixed_record.json').read_text())
    data=json.loads(Path(donor['manifest']).read_text());data.update(pg='pages',residency=1,kappa=1)
    for g in data['gemms']:
        g.update(stages=2,split_k=1);g['tile_n']=min(g['tile_n'],16384//(2*g['tile_k']))
    folder=out/'fixed';folder.mkdir(parents=True,exist_ok=True)
    source=folder/'source.json';source.write_text(json.dumps(data,indent=2)+'\n')
    fixed=pin(source,donor['classes'],donor['classes'],folder,original['overrides'])
    common=[str(compiler),'compile',donor['export'],'OUTPUT','--serving','decode',
            '--batch','BATCH','--past-range','64:1086','--capacity','1088','--solver','skeleton',
            '--solve',str(TARGET),'--runtime-target',str(TARGET),'--emit','serving',
            '--flow-search-only','1','--search-jobs','3','--handoff','off']
    fixed_cmd=[x.replace('OUTPUT',str(folder/'plan.so')).replace('BATCH','16') for x in common]
    fixed_cmd+=['--search-passes','1','--top-m','1','--search-budget-ms','60000']+fixed['options']
    config=json.loads((ROOT/'docs/experiments/SERVING_R13_SM120/llama_r13_sm120_r3.json').read_text())
    folder=out/'joint';folder.mkdir(parents=True,exist_ok=True)
    joint=[x.replace('OUTPUT',str(folder/'plan.so')).replace('BATCH','1') for x in common]
    joint+=['--search-passes','2','--top-m','8','--search-budget-ms','1600000']
    for key,value in config['features'].items():
        if key in ('decode_executor','decode_loop','prefill_executor','handoff'):continue
        joint+=['--'+key.replace('_','-'),'pages' if key=='pg' else str(value)]
    return [('fixed',fixed_cmd),('joint',joint)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,cmd in commands(a.compiler,a.out):
        folder=a.out/name
        (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        # Audit is intentionally a streamed archive: a full search emits many
        # per-task partitions and never sends them through the conversation.
        with gzip.open(folder/'audit.log.gz','wb') as log:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                env=dict(os.environ,TILEMEGA_FLOW_AUDIT='1'))
            for line in proc.stdout:log.write(line)
            code=proc.wait()
        rows.append({'name':name,'exit_code':code,'command':str(folder/'command.json')})
        (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'run.done').touch()
if __name__=='__main__':main()
