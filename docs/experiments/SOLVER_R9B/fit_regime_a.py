#!/usr/bin/env python3
"""R9b backend calibration from the 490 R6 observations, never E2E times."""
import csv,functools,hashlib,json,math,pathlib,re,statistics,sys
E=pathlib.Path(__file__).resolve().parent;C=E.parent/'COSTMODEL'
sys.path.insert(0,str(C))
from fit_body import nnls

def summary(errors):
    return dict(p50=statistics.median(errors),mean=statistics.mean(errors),p90=sorted(errors)[math.ceil(.9*len(errors))-1],max=max(errors))
def main():
    out=E/'fit';out.mkdir(exist_ok=True)
    target=json.loads((C/'event_fit/target.json').read_text());section=target['calibration_by_dtype']['bf16'];cal=section['pipelines'];fit=section['task_body'];sms=target['resources']['num_sms']
    rows=list(csv.DictReader((C/'body_fit/observations.tsv').open(),delimiter='\t'));assert len(rows)==490
    specs={};sources={};meta={};slots={};assumptions=[]
    for r in rows:
        for k in ('tile_m','tile_n','tile_k','stages','stage','samples'):r[k]=int(r[k])
        for k in ('iterations','operand_elements','fixed_ns','loop_body_ns','loop_wait_ns','loop_fixed_ns'):r[k]=float(r[k])
        key=(r['cohort'],r['cell'])
        if key not in specs:specs[key]=json.loads((C/key[0]/key[1]/'specs.json').read_text())
        spec=specs[key][r['arm']];dump=pathlib.Path(r['dump'])
        if r['dump'] not in meta:
            meta[r['dump']]={x['key']:x['value'] for x in csv.DictReader((dump/'meta.tsv').open(),delimiter='\t')}
            slots[r['dump']]=list(csv.DictReader((dump/'slots.tsv').open(),delimiter='\t'))
        m=int(meta[r['dump']]['seq']);tm=r['tile_m'];tn=r['tile_n'];tk=r['tile_k'];source=spec['source']
        if source not in sources:
            text=pathlib.Path(source).read_text();table=text.split('constexpr GemmDesc kGemms[] = {',1)[1].split('};',1)[0]
            sources[source]=[(int(a),int(b)) for a,b in re.findall(r'\{(\d+), (\d+),',table)]
        assert all(n%tn==0 for n,k in sources[source]),'N tails require per-stage coordinate mapping'
        assert m<tm or m%tm==0,'M tails require per-coordinate features'
        actual=[int(x['logical_task']) for x in slots[r['dump']] if int(x['stage'])==r['stage']]
        assert len(actual)==r['samples'] and len(set(actual))==len(actual)
        r['physical_writes']=min(m,tm)*tn;r['nominal_writes']=tm*tn
        r['physical_stage_bytes']=2*tk*(min(m,tm)+tn)
        if 'residency' not in spec:assumptions.append(dict(cell=r['cell'],arm=r['arm'],occupancy=1))
        r['occupancy']=float(spec.get('residency',1))
        # References fit in L2. For real-width, the external weight stream is
        # DRAM and activations are produced; these shapes have only A/B reads.
        external=tn/(min(m,tm)+tn) if r['cell'].startswith('real_') else 0
        r['latency_ns']=(1-external)*cal['l2_latency_ns']+external*cal['dram_latency_ns']
        body=fit['loop_body'][0]+fit['loop_body'][1]*tm*tn*tk
        lanes=max(r['occupancy']*r['physical_stage_bytes']/(cal['l2_gbps']/sms),r['occupancy']*2*tm*tn*tk/(cal['tc_bf16_gflops']/sms))
        r['base_ns']=max(body,lanes)
        r['old_loop_ns']=max(body,lanes)+fit['loop_wait'][0]+fit['loop_wait'][1]*r['operand_elements']
        r['observed_loop_ns']=r['loop_body_ns']+r['loop_wait_ns']
        r['old_fixed_ns']=sum(a*b for a,b in zip(fit['fixed'],[1,tm*tn,r['operand_elements']*r['stages']]))
    # Nonnegative coordinate minimization of squared error through max(base,
    # lambda*latency/(S-1)+inverse_rate*o*bytes/(S-1)). Same unweighted stages
    # as the repository's original NNLS (not sample-count weighted).
    terms=[(r['latency_ns']/(r['stages']-1),r['occupancy']*r['physical_stage_bytes']/(r['stages']-1),r['base_ns'],r['observed_loop_ns']) for r in rows]
    def loss(x):return sum((max(b,x[0]*a+x[1]*t)-y)**2 for a,t,b,y in terms)
    x=[1.,sms/cal['l2_gbps']]
    for _ in range(100):
        previous=loss(x)
        for axis,hi in [(0,20.),(1,10.)]:
            lo=0
            for j in range(70):
                a=lo+(hi-lo)/3;b=hi-(hi-lo)/3;xa=x.copy();xb=x.copy();xa[axis]=a;xb[axis]=b
                if loss(xa)<=loss(xb):hi=b
                else:lo=a
            x[axis]=(hi+lo)/2
        if abs(previous-loss(x))<1e-8:break
    physical=nnls([[1,r['physical_writes'],r['operand_elements']*r['stages']] for r in rows],[r['fixed_ns'] for r in rows])
    for r in rows:
        r['new_loop_ns']=max(r['base_ns'],(x[0]*r['latency_ns']+x[1]*r['occupancy']*r['physical_stage_bytes'])/(r['stages']-1))
        r['new_fixed_ns']=sum(a*b for a,b in zip(physical,[1,r['physical_writes'],r['operand_elements']*r['stages']]))
    errors={}
    for label,rs in [('all',rows)]+[(f'S{s}',[r for r in rows if r['stages']==s]) for s in sorted(set(r['stages'] for r in rows))]:
        errors[label]={}
        for term,actual in [('loop','observed_loop_ns'),('fixed','fixed_ns')]:
            for version in ('old','new'):errors[label][version+'_'+term]=summary([abs(r[version+'_'+term+'_ns']/r[actual]-1) for r in rs])
    fit['latency_scale']=x[0];fit['stage_rate_bytes_per_ns']=1/x[1];fit['fixed_physical']=physical
    fit['regime_a_source']=str(out/'observations.tsv')
    (out/'target.json').write_text(json.dumps(target,indent=2)+'\n')
    with (out/'observations.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    result=dict(rows=len(rows),latency_scale=x[0],stage_rate_bytes_per_ns=1/x[1],fixed_physical=physical,errors=errors,occupancy_assumptions=assumptions,
                provenance=dict(observations_sha256=hashlib.sha256((C/'body_fit/observations.tsv').read_bytes()).hexdigest(),target_sha256=hashlib.sha256((C/'event_fit/target.json').read_bytes()).hexdigest()),
                limitations=['Loaded latency uses operand-size-weighted L2/DRAM source classes; real-width observations use streamed B and produced A.','These phase observations are historical backend calibration, not fresh performance conclusions.'])
    (out/'fit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
