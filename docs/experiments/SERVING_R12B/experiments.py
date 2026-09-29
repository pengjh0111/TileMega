#!/usr/bin/env python3
"""Fixed R12b S-1/S-1b/S-3 controls using the existing compiler and measure tool."""
import argparse, contextlib, fcntl, json, os, subprocess, sys
from pathlib import Path
from tilemega.cache import export_key
from tilemega.fingerprint import ROOT

HERE=Path(__file__).resolve().parent
WORK=HERE/'controls'
BIN=Path(os.getenv('TILEMEGA_BIN',str(ROOT/'build-phase12/tools/tilemega')))
CONFIG=json.loads((ROOT/'configs/e2e/llama_r12.json').read_text())
MODEL=Path(CONFIG['model']['path']);CACHE=Path(CONFIG['device']['cache_dir']).expanduser()
TARGET=next((CACHE/'targets').glob('*.json'))
PROMPTS=ROOT/CONFIG['workload']['prompts']
POLICY=ROOT/CONFIG['test']['policy_file']

def exported(phase):
    digest=export_key(MODEL/'config.json',ROOT/'python/tilemega/serving/export.py',
                      phase,1 if phase=='decode' else 64,1088)
    result=CACHE/'exports'/digest/'bridge.json'
    if not result.exists():raise FileNotFoundError(result)
    return result

def fixed_plan(batch,phase,pg,label,*,lookahead=-1,poll=0):
    folder=WORK/label;folder.mkdir(parents=True,exist_ok=True)
    so=folder/f'{phase}_B{batch}.so'
    if so.exists():return so
    # The prefill lm_head still has M=batch, so a uniform 64-row domain is
    # empty at B=1 under serving R-2. Use the common legal 16-row geometry.
    shape=dict(tile_m=16,tile_n=128,
               tile_k=64,stages=2,split_k=1)
    domain={'geometries':[dict(shape,split_k=split) for split in (1,2,4,8)]}
    domain_file=folder/f'{phase}_B{batch}_domain.json';domain_file.write_text(json.dumps(domain))
    interval='64:1086' if phase=='decode' else '0:0'
    args=[str(BIN),'compile',str(exported(phase)),str(so),'--serving',phase,
          '--batch',str(batch),'--past-range',interval,'--capacity','1088',
          '--solver','skeleton','--solve',str(TARGET),'--runtime-target',str(TARGET),
          '--emit','serving','--search-domain',str(domain_file),
          '--search-passes','1','--top-m','1','--search-jobs','3',
          '--search-budget-ms','60000','--measure-cmd',
          f'{sys.executable} -m tilemega.serving.measure_candidate --model {MODEL}',
          '--pg',pg,'--weight-layout','tiled' if pg=='pages' else 'row',
          '--page-bytes','16384','--sync','calibrated','--handoff','off',
          '--lookahead-bytes',str(lookahead),'--v3-poll-ns',str(poll),
          '--kphase-mask','31']
    subprocess.run(args,check=True)
    return so

@contextlib.contextmanager
def gpu_lock():
    path=Path(os.getenv('TILEMEGA_GPU_LOCK',str(CACHE/'gpu.lock')))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield

def timing(prefill,decode,batch,label,*,loop=True,mode='L2',env=None,
           warmup=1,repeats=3):
    out=WORK/label;out.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-m','tilemega.serving.measure','--model',str(MODEL),
          '--prefill-so',str(prefill),'--decode-so',str(decode),
          '--prompt-ids',str(PROMPTS),'--batch',str(batch),'--out',str(out),
          '--mode',mode,'--decode-loop',str(int(loop)),'--max-new-tokens','256',
          '--warmup',str(warmup),'--repeats',str(repeats),'--policy',str(POLICY)]
    with gpu_lock():subprocess.run(args,check=True,env=dict(os.environ,**(env or {})))
    result=json.loads((out/'measurements.json').read_text())
    return 1000*result['tpot_p50_seconds']

def s1():
    rows=[]
    for batch in (1,16):
        prefill=fixed_plan(batch,'prefill','l2',f's1_B{batch}')
        for pg in ('pages','l2'):
            decode=fixed_plan(batch,'decode',pg,f's1_B{batch}_{pg}')
            ms=timing(prefill,decode,batch,f's1_B{batch}_{pg}_measure')
            rows.append(dict(batch=batch,pg=pg,p50_ms=ms,prefill=str(prefill),decode=str(decode)))
    (HERE/'s1.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows))

def s1b():
    try:
        s1rows=json.loads((HERE/'s1.json').read_text())
        results={'complete':False,'class_off_ms':{},'lookahead_ms':{},'poll_ms':{}}
        for batch in (1,16):
            row=next(r for r in s1rows if r['batch']==batch and r['pg']=='pages')
            prefill=Path(row['prefill']);decode=Path(row['decode'])
            for mask in [31,0]+[31 & ~(1<<bit) for bit in range(5)]:
                ms=timing(prefill,decode,batch,f's1b_B{batch}_mask{mask}',
                          env={'TILEMEGA_KPHASE_MASK':str(mask)})
                results.setdefault('masks_by_batch',{}).setdefault(str(batch),{})[str(mask)]=ms
                if batch==16:
                    if mask==31:results['baseline_ms']=ms
                    elif mask==0:results['all_off_ms']=ms
                    else:
                        for bit in range(5):
                            if mask==31 & ~(1<<bit):results['class_off_ms'][str(bit)]=ms
        row=next(r for r in s1rows if r['batch']==16 and r['pg']=='pages')
        for depth in (0,131072):
            decode=fixed_plan(16,'decode','pages',f's1b_D{depth}',lookahead=depth)
            results['lookahead_ms'][str(depth)]=timing(row['prefill'],decode,16,f's1b_D{depth}_measure')
        for poll in (0,200):
            decode=fixed_plan(16,'decode','pages',f's1b_poll{poll}',poll=poll)
            results['poll_ms'][str(poll)]=timing(row['prefill'],decode,16,f's1b_poll{poll}_measure')
        results['complete']=True
    except BaseException as error:
        results.setdefault('error',repr(error))
        (HERE/'s1b.json').write_text(json.dumps(results,indent=2)+'\n')
        raise
    (HERE/'s1b.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results))

def s3():
    plans=json.loads((ROOT/'runs/r12b-llama/plans.json').read_text())
    arms=[('eft_loop',True,'L2',{}),('rotate_loop',True,'L2',{'TILEMEGA_PLACEMENT_ABLATION':'rotate'}),
          ('eft_separate',False,'L2',{}),('l1_separate',False,'L1',{}),
          ('no_phase',True,'L2',{'TILEMEGA_KPHASE_MASK':'0'})]
    rows=[]
    for batch in (1,16):
        pair=plans[str(batch)]
        for round in range(3):
            for label,loop,mode,env in arms[round:]+arms[:round]:
                ms=timing(pair['prefill'],pair['decode'],batch,
                          f's3_B{batch}_r{round}_{label}',loop=loop,mode=mode,env=env,
                          warmup=0,repeats=1)
                rows.append(dict(batch=batch,round=round,arm=label,p50_ms=ms))
    (HERE/'s3.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=('s1','s1b','s3'))
    a=p.parse_args();globals()[a.experiment]()
