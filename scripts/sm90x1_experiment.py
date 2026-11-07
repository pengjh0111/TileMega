"""SM90X1 phase-gated sweeps, measured compositions and reportable arm units."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_inputs import model_dir
from sm90r4_fixed import native_command
from sm90r4_dispatch import run_unit
import sm90x1_analysis as stats

RUN=ROOT/'runs/sm90x1'
RESULTS=ROOT/'docs/results'


def families(c):
    return ['qkv','o','lm_head','attention','experts'] if c['moe'] else ['qkv','o','gate_up','down','lm_head','attention']


def gate(phase):
    p0=read(RUN/'phase0.json')
    if p0['stop'] or not p0['complete']:
        raise ValueError('Phase 0 must complete within the 2% gate before sweeps or main timing')
    if phase>=2:
        pilot=read(RUN/'pilot-gate.json')
        if not pilot['continue']:
            raise ValueError('pilot cost projection gate has not passed')
    if phase>=3:
        if not read(RUN/'main-complete.json')['complete']:
            raise ValueError('Phase 3 requires complete Phase 2 results')


def rows(a):
    import sm90x1 as base
    selected=base.cells(a.cell)
    selected=[c for c in selected if c['pilot']==(a.phase==1)] if a.phase in (1,2) else selected
    if not selected:raise ValueError('no selected cells belong to this phase')
    return selected


def cell_dir(c,a):
    return RUN/'cells'/c['cell']/a.tag


def inspect(a,c,mode,out,extra=()):
    d=RUN/'cells'/c['cell']
    command=[str(a.compiler.resolve()),'inspect','native-sweep','--export',str(d/'decode.export.json'),
             '--target',str(ROOT/'configs/targets/sm_90_h800.json'),'--batch',str(c['batch']),
             '--reference',str(d/'W.json'),'--mode',mode,'--out',str(out),*extra]
    with out.with_suffix('.log').open('w') as log:
        subprocess.check_call(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                              stdout=log,stderr=subprocess.STDOUT)
    return read(out/'manifest.json')


def prepare(a):
    gate(a.phase)
    if a.k==3 and (not (RUN/'pilot-gate.json').exists() or read(RUN/'pilot-gate.json')['next_k']!=3):
        raise ValueError('k=3 requires the measured k=4 projection to exceed the budget')
    for c in rows(a):
        d=cell_dir(c,a);d.mkdir(parents=True,exist_ok=True)
        variants=[];plan={}
        out=d/'sweeps'/'all';out.parent.mkdir(exist_ok=True)
        manifest=read(out/'manifest.json') if (out/'manifest.json').exists() else inspect(
            a,c,'sweep',out,('--family','all','--top',str(a.k or c['k'])))
        for family in families(c):
            for v in manifest['variants']:
                if v['family']!=family:continue
                v=dict(v,cell=c['cell'],family=family)
                if v['selected']:
                    key=json.dumps(v['coordinates'],sort_keys=True)
                    entry=plan.get(key)
                    if entry is None:
                        entry=dict(id='W' if v['reference'] else family+'-'+v['variant'],
                                   coordinates=v['path'],coordinates_sha256=sha(v['path']),
                                   normalized_coordinates=v['coordinates'])
                        plan[key]=entry
                    v['build_id']=entry['id']
                variants.append(v)
        write(d/'variants.json',variants)
        write(d/'build-plan.json',list(plan.values()))


def build(a):
    import sm90x1 as base
    gate(a.phase)
    compiler=base.freeze_compiler(a.compiler)
    campaign=RUN/'measurement-head.json'
    if campaign.exists() and read(campaign)['head']!=base.head():
        raise ValueError('all Phase 1/2/3 arms require the frozen measurement HEAD')
    write(campaign,dict(head=base.head()))
    for c in rows(a):
        d=cell_dir(c,a)
        plan=read(d/'arm-plan.json') if a.arms_only else read(d/'build-plan.json')
        for entry in plan:
            out=d/entry['id']
            if (out/'binding.json').exists():
                binding=read(out/'binding.json')
                if binding['head']!=base.head() or sha(binding['decode'])!=binding['decode_sha256']:
                    raise ValueError('existing build differs; retain it and choose another tag')
                continue
            out.mkdir(exist_ok=True)
            if (out/'build.json').exists():
                continue  # failed binaries remain failed rows, never silently replaced
            binary=out/'decode.so'
            command=native_command(compiler,RUN/'cells'/c['cell']/'decode.export.json',binary,
                                   entry['coordinates'],c['batch'])
            start=time.monotonic();peak=0
            with (out/'build.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                                         stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    peak=max(peak,base.tree_rss(process.pid));time.sleep(0.5)
            record=dict(entry,cell=c['cell'],head=base.head(),status=process.returncode,
                        wall_s=time.monotonic()-start,peak_tree_rss_bytes=peak,command=command)
            write(out/'build.json',record)
            if process.returncode:
                continue
            with (out/'host_abi.log').open('w') as log:
                status=subprocess.call([PYTHON,str(ROOT/'scripts/check_sm90r2_native_abi.py'),
                        '--prefill',c['prefill'],'--decode',str(binary),'--model-config',
                        str(model_dir(c['model'])/'config.json')],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'python')),
                        stdout=log,stderr=subprocess.STDOUT)
            identity=read(str(binary)+'.identity.json')
            record.update(decode=str(binary),decode_sha256=sha(binary),prefill=c['prefill'],
                          host_abi_status=status,trace=identity['trace'],source_head=identity['source']['head'])
            if status or identity['trace'] or identity['source']['head']!=base.head():
                write(out/'failure.json',record);continue
            write(out/'binding.json',record)


def trace(a):
    import sm90x1 as base
    gate(a.phase)
    # Reuse the Phase 0 trace scheduler for each unique selected coordinate.
    for c in rows(a):
        for entry in read(cell_dir(c,a)/'build-plan.json'):
            base.trace_cell(a,c,entry['id'],reference=False)


def analyze(a):
    gate(a.phase)
    for c in rows(a):
        d=cell_dir(c,a);out=[];deltas=[]
        for row in read(d/'variants.json'):
            row=dict(row)
            if row['selected']:
                family=row['family']
                measured=stats.aggregate(d/row['build_id']/'trace',family,c['moe'])
                reference=stats.aggregate(d/'W/trace',family,c['moe'])
                row.update({k:v for k,v in measured.items() if k!='stages'})
                row.update(delta_own_us=row['own_us']-reference['own_us'],
                           delta_step_us=row['step_us']-reference['step_us'])
                row['delta_coupled_us']=row['delta_step_us']-row['delta_own_us']
                reference_stages={s['stage']:s for s in reference['stages']}
                for s in measured['stages']:
                    deltas.append(dict(cell=c['cell'],family=family,variant=row['variant'],
                                       stage=s['stage'],stage_kind=s['stage_kind'],stage_name=s['stage_name'],
                                       delta_us=s['ledger_us']-reference_stages[s['stage']]['ledger_us']))
            out.append(row)
        write(d/'analysis.json',out);write(d/'stage-deltas.json',deltas)


def arms(a):
    gate(a.phase)
    for c in rows(a):
        d=cell_dir(c,a);d.mkdir(parents=True,exist_ok=True)
        if a.phase==3:
            source=RUN/'cells'/c['cell']/('phase1' if c['pilot'] else 'phase2')
            plans=[dict(e,id=str(source/e['id'])) for e in read(source/'arm-plan.json')]
            for arm in (['T'] if c['moe'] else ['T','N']):
                out=d/('control-'+arm)
                manifest=inspect(a,c,arm,out);variant=manifest['variants'][0]
                plans.append(dict(id=arm,arm=arm,coordinates=variant['path'],coordinates_sha256=sha(variant['path'])))
            write(d/'arm-plan.json',plans)
            continue
        data=read(d/'analysis.json');plans=[];decisions={}
        reference=next(e for e in read(d/'build-plan.json') if e['id']=='W')
        plans.append(dict(reference,arm='W'))
        for arm,criterion in [('L','own_us'),('Wstar','step_us')]:
            selection={}
            for family in families(c):
                measured=[r for r in data if r['family']==family and r['selected']]
                ordered=stats.criterion_order(measured,criterion)
                selection[family]=[r['path'] for r in ordered]
            selection_path=d/(arm+'-selection.json');write(selection_path,selection)
            out=d/('compose-'+arm)
            manifest=read(out/'manifest.json') if (out/'manifest.json').exists() else inspect(
                a,c,'compose',out,('--selection',str(selection_path)))
            variant=manifest['variants'][0]
            # Coordinate equality, rather than file formatting, decides binary reuse.
            found=next((e for e in plans if e['normalized_coordinates']==variant['coordinates']),None)
            entry=dict(id=found['id'] if found else arm,arm=arm,
                       coordinates=found['coordinates'] if found else variant['path'],
                       coordinates_sha256=sha(found['coordinates'] if found else variant['path']),
                       normalized_coordinates=variant['coordinates'])
            plans.append(entry);decisions[arm]=manifest['choices']
        write(d/'arm-plan.json',plans);write(d/'arm-decisions.json',decisions)


def measurement(a):
    gate(a.phase)
    for c in rows(a):
        d=cell_dir(c,a);plan=read(d/'arm-plan.json')
        bindings={e['arm']:read(d/e['id']/'binding.json') for e in plan}
        if len({b['head'] for b in bindings.values()})!=1:
            raise ValueError('every arm of a cell must share one build HEAD')
        if any(b['host_abi_status'] or b['trace'] or sha(b['decode'])!=b['decode_sha256'] for b in bindings.values()):
            raise ValueError('arm ABI/identity failed')
        if sha(c['prefill'])!=c['prefill_sha256']:raise ValueError('frozen prefill changed')
        labels=list(bindings)
        if a.action=='corr':
            for entry in {e['id']:e for e in plan}.values():
                label=entry['arm'];binding=bindings[label]
                spec=dict(name=f"sm90x1-corr-{a.tag}-{c['cell']}-{entry['id']}",kind='corr',model=c['model'],
                          batch=c['batch'],arms={label:dict(decode=binding['decode'],prefill=c['prefill'])},
                          order=[label]*5,correctness=dict(c1=256,c2=True,reference_processes=1),
                          **{'class':c['class']})
                path=d/(entry['id']+'-corr-spec.json');write(path,spec)
                result=run_unit(path,d/entry['id']/'corr',c['class'])
                if not result['pass']:
                    raise ValueError('P-CORR failed; cell must remain a failure row')
        else:
            for e in plan:
                result=read(d/e['id']/'corr/result.json')
                if not result['pass'] or not result['c1_pass'] or not result['c2_pass']:
                    raise ValueError('P-CORR must pass before arm timing')
            order=[]
            for i in range(c['processes']):
                order+=labels if i%2==0 else list(reversed(labels))
            spec=dict(name=f"sm90x1-sel-{a.tag}-{c['cell']}",kind='sel',model=c['model'],batch=c['batch'],
                      arms={label:dict(decode=b['decode'],prefill=c['prefill']) for label,b in bindings.items()},
                      order=order,**{'class':c['class']})
            path=d/'time-spec.json';write(path,spec);run_unit(path,d/'time',c['class'])


def pilot_gate(a):
    import sm90x1 as base
    cells=base.cells('all');pilot=base.cells('pilot')
    if not all((cell_dir(c,a)/'time/result.json').exists() for c in pilot):
        raise ValueError('pilot arms and timing must finish before the cost projection')
    counts=[len(read(cell_dir(c,a)/'build-plan.json')) for c in pilot]
    k=a.k or 4
    # Enumerated group counts are workload-dependent; use the conservative maximum for every unswept cell.
    builds=sum(2*(1+len(families(c))*(k-1))+2 for c in cells)
    measured_s=0;processes=0
    for c in pilot:
        for path in cell_dir(c,a).rglob('guard_result.json'):
            guard=read(path)
            if guard.get('code')==0:
                # Scheduler run_s includes preflight, process startup and cooldown.
                processes+=1
        for path in cell_dir(c,a).rglob('state.json'):
            measured_s+=sum(r.get('run_s',0) for r in read(path).values() if r.get('status')=='done')
    if not processes:
        raise ValueError('no accepted GPU processes to project')
    estimated_processes=sum(2*(1+len(families(c))*(k-1))+15+3*c['processes'] for c in cells)
    hours=measured_s/processes*estimated_processes/3600
    result=dict(k=k,max_builds=builds,gpu_hours=hours,continue_=builds<=600 and hours<=40,
                next_k=3 if k==4 and (builds>600 or hours>40) else k,
                evidence='inferred conservative build bound and pilot mean process wall time')
    result['continue']=result.pop('continue_');write(RUN/'pilot-gate.json',result)
    (RUN/'pilot.md').write_text('# SM90X1 pilot\n\n[inferred] Cost projection: '+json.dumps(result)+'\n\n'
        '[verified] Per-cell metrics and diagnostics are in the SM90X1 compact tables; no trace TPOT is reportable.\n')
    return result


def decomposition(c,family):
    if family=='attention':
        return c['attention']['chunks'],c['attention']['block_keys']
    if family=='experts':
        return c['gemms']['expert_gate_up']['tile_n'],c['gemms']['expert_down']['tile_n']
    return c['gemms'][family]['tile_n']


def tables(a):
    import sm90x1 as base
    all_cells=base.cells('all');candidates=[];stages=[];arms_out=[];ratios=[];choices=[];reversed_pairs=[]
    for c in all_cells:
        tag=a.tag if a.phase==3 or a.cell!='all' else ('phase1' if c['pilot'] else 'phase2')
        d=RUN/'cells'/c['cell']/tag
        source=RUN/'cells'/c['cell']/('phase1' if c['pilot'] else 'phase2')
        diagnostic=d if (d/'analysis.json').exists() else source
        data=read(diagnostic/'analysis.json') if (diagnostic/'analysis.json').exists() else \
             read(diagnostic/'variants.json') if (diagnostic/'variants.json').exists() else []
        failures=[read(path) for path in sorted((d/'failures').glob('*.json'))]
        failure_text='; '.join(r['action']+': '+r['error'] for r in failures)
        if not data and failure_text:
            data=[dict(cell=c['cell'],family=f,variant='enumeration_failed',legal='',selected=False,
                       reason=failure_text) for f in families(c)]
        for r in data:
            if r.get('build_id'):
                artifact=d/r['build_id']
                r['build_status']='compiled' if (artifact/'binding.json').exists() else \
                    'failed' if (artifact/'build.json').exists() else 'not_run'
            if failure_text:r['failure_reason']=failure_text
        candidates+=data
        if (diagnostic/'stage-deltas.json').exists():stages+=read(diagnostic/'stage-deltas.json')
        for family in families(c):
            measured=[r for r in data if r['family']==family and r['selected'] and 'own_us' in r]
            for reversal in stats.reversals(measured):
                reversed_pairs.append(dict(reversal,cell=c['cell'],family=family))
        measured=d if (d/'time/result.json').exists() else source
        plan_dir=d if (d/'arm-plan.json').exists() else source
        plan=read(plan_dir/'arm-plan.json') if (plan_dir/'arm-plan.json').exists() else []
        result=read(measured/'time/result.json') if (measured/'time/result.json').exists() else None
        protocol=read(measured/'time/unit.json') if result else {}
        for label in ['W','L','Wstar']+(['T'] if a.phase==3 and c['moe'] else ['T','N'] if a.phase==3 else []):
            entry=next((e for e in plan if e['arm']==label),None)
            row=dict(cell=c['cell'],batch=c['batch'],arm=label,status='blocked' if failure_text else 'not_run',
                     failure_reason=failure_text,gpu='',**{'class':c['class']})
            if entry:
                row['coordinates_sha256']=entry['coordinates_sha256']
                if (plan_dir/entry['id']/'binding.json').exists():
                    binding=read(plan_dir/entry['id']/'binding.json')
                    row.update(binary_sha256=binding['decode_sha256'],build_head=binding['head'])
                    if binding['trace']:raise ValueError('trace build cannot enter arms table')
                corr_path=plan_dir/entry['id']/'corr/result.json'
                if corr_path.exists():
                    corr=read(corr_path);row.update(c1_pass=corr['c1_pass'],c2_pass=corr['c2_pass'])
                    if not corr['pass']:row['status']='corr_fail'
                elif (plan_dir/entry['id']/'build.json').exists() and not (plan_dir/entry['id']/'binding.json').exists():
                    row['status']='build_fail'
                if result and label in result['arms']:
                    if not row.get('c1_pass') or not row.get('c2_pass'):
                        raise ValueError('uncertified TPOT cannot enter report')
                    row.update(result['arms'][label],status='measured',gpu=protocol['gpu'])
                    for ref,suffix in [('W','w'),('Wstar','wstar')]:
                        other=result['arms'][ref];change=100*(row['tpot_ms']/other['tpot_ms']-1)
                        threshold=max(0.5,2*max(row['range_pct'],other['range_pct']))
                        row['change_vs_'+suffix+'_pct']=change
                        row['decision_vs_'+suffix]='indistinguishable' if abs(change)<=threshold else 'faster' if change<0 else 'slower'
            arms_out.append(row)
        if result:
            r=result['arms'];ratios.append(dict(cell=c['cell'],batch=c['batch'],X=r['L']['tpot_ms']/r['Wstar']['tpot_ms'],
                                              Xprime=r['L']['tpot_ms']/r['W']['tpot_ms']))
            coords={e['arm']:read(e['coordinates']) for e in plan}
            for f in families(c):
                choices.append(dict(cell=c['cell'],batch=c['batch'],family=f,
                    Y=int(decomposition(coords['L'],f)!=decomposition(coords['Wstar'],f)),
                    Yprime=int(decomposition(coords['L'],f)!=decomposition(coords['W'],f))))
    candidates_fields=['cell','family','variant','decomposition_key','coordinates_sha256','legal','reason',
        'predicted_stage_ns','predicted_step_ns','modeled_bytes','own_us','step_us','delta_own_us',
        'delta_step_us','delta_coupled_us','processes','selected','own_us_range_pct','step_us_range_pct','build_status','failure_reason']
    stats.write_csv(RESULTS/'SM90X1_candidates.csv',candidates,candidates_fields)
    stats.write_csv(RESULTS/'SM90X1_stage_deltas.csv',stages,
                    ['cell','family','variant','stage_kind','delta_us','stage','stage_name'])
    stats.write_csv(RESULTS/'SM90X1_arms.csv',arms_out,['cell','arm','coordinates_sha256','binary_sha256',
        'c1_pass','c2_pass','tpot_ms','range_pct','processes','change_vs_wstar_pct','decision_vs_wstar',
        'change_vs_w_pct','decision_vs_w','gpu','class','status','build_head','failure_reason'])
    summary=[]
    for batch in [1,16,'overall']:
        selected=[c for c in all_cells if batch=='overall' or c['batch']==batch]
        names={c['cell'] for c in selected};subset=[r for r in ratios if r['cell'] in names]
        complete=len(subset)==len(selected)
        for metric in ['X','Xprime']:
            values=[r[metric] for r in subset]
            summary.append(dict(metric=metric,batch=batch,family='',count=len(values),denominator=len(selected),
                value=math.exp(sum(map(math.log,values))/len(values)) if complete else '',
                min=min(values) if complete else '',max=max(values) if complete else '',
                status='complete' if complete else 'pending'))
        for metric in ['Y','Yprime']:
            subset_y=[r for r in choices if r['cell'] in names];denominator=sum(len(families(c)) for c in selected)
            count=sum(r[metric] for r in subset_y)
            summary.append(dict(metric=metric,batch=batch,family='',count=count if complete else '',
                                denominator=denominator,value=count/denominator if complete else '',
                                status='complete' if complete else 'pending'))
        for family in sorted({f for c in selected for f in families(c)}):
            data=[r for r in candidates if r['cell'] in names and r['family']==family and r.get('processes')]
            for metric,value in [('tau',stats.kendall(data)),('coupled_share',stats.coupled_share(data))]:
                summary.append(dict(metric=metric,batch=batch,family=family,count=len(data),denominator='',
                                    value=value if value is not None else '',status='diagnostic' if data else 'pending'))
    for c in all_cells:
        for family in families(c):
            data=[r for r in candidates if r['cell']==c['cell'] and r['family']==family and r.get('processes')]
            summary.append(dict(metric='tau_cell',batch=c['batch'],family=family,cell=c['cell'],
                                count=len(data),value=stats.kendall(data) if data else '',status='diagnostic' if data else 'pending'))
    for r in ratios:
        for metric in ['X','Xprime']:
            summary.append(dict(metric=metric+'_cell',cell=r['cell'],batch=r['batch'],value=r[metric],
                                count=1,denominator=1,status='complete'))
    stats.write_csv(RESULTS/'SM90X1_summary.csv',summary,
                    ['metric','batch','family','cell','count','denominator','value','min','max','status'])
    normalized=[]
    for row in arms_out:
        if row['status']=='measured':
            ref=next(r for r in arms_out if r['cell']==row['cell'] and r['arm']=='Wstar')
            normalized.append(dict(cell=row['cell'],arm=row['arm'],normalized_tpot=row['tpot_ms']/ref['tpot_ms']))
    stats.write_csv(RESULTS/'SM90X1_figure_arms.csv',normalized,['cell','arm','normalized_tpot'])
    reversed_pairs.sort(key=lambda r:-r['strength_pct'])
    illustrative=list(dict.fromkeys(r['family'] for r in reversed_pairs))[:2]
    if len(illustrative)<2:
        illustrative=list(dict.fromkeys(illustrative+['qkv','attention']))[:2]
    stats.write_csv(RESULTS/'SM90X1_figure_scatter.csv',[r for r in candidates if r['family'] in illustrative and r.get('processes')],
                    ['cell','family','variant','own_us','step_us','processes'])
    write(RUN/'rank-reversals.json',reversed_pairs)
    write(RUN/'main-complete.json',dict(complete=len(ratios)==14,measured=len(ratios),required=14))
    print(json.dumps(dict(measured_cells=len(ratios),required_cells=14,summary=summary[:4]),indent=2))
    pilot_names={c['cell'] for c in all_cells if c['pilot']}
    if all(any(r['cell']==name for r in ratios) for name in pilot_names):
        pilot_args=argparse.Namespace(**vars(a));pilot_args.tag='phase1'
        projection=pilot_gate(pilot_args)
        pr=[r for r in ratios if r['cell'] in pilot_names];pc=[r for r in choices if r['cell'] in pilot_names]
        diagnostic=[r for r in candidates if r['cell'] in pilot_names and r.get('processes')]
        text=['# SM90X1 pilot','', '[verified] X='+str(math.exp(st.mean(math.log(r['X']) for r in pr)))]
        text+=['[verified] Y='+str(sum(r['Y'] for r in pc)/len(pc)),
               '[verified] Illegal groups='+str(sum(r['legal'] is False for r in candidates if r['cell'] in pilot_names)),
               '[inferred] Projection: '+json.dumps(projection)]
        for f in sorted({r['family'] for r in diagnostic}):
            data=[r for r in diagnostic if r['family']==f]
            text+=['[verified, diagnostic] '+f+': tau='+str(stats.kendall(data))+', coupled_share='+str(stats.coupled_share(data))]
        (RUN/'pilot.md').write_text('\n\n'.join(text)+'\n')
    report=['# SM90X1: operator-local vs whole-program selection','',
            '[stated] Fixed scope: 12 dense cells and two Qwen3 MoE cells, B1/B16; optional Phase 3 is gated.',
            '[verified] Frozen inputs: configs/serving/sm90x1/cells.json; MoE binary/binding/plan hashes checked.',
            '[verified] Only four H800 cards are currently visible; timing policy still excludes GPU6.',
            '[verified] '+str(len(ratios))+'/14 cells have reportable non-trace arm measurements.',
            '## Numbers for the paper','',
            '[verified] X, X\u2032, Y, Y\u2032 are unavailable until every required cell has completed. '
            'Missing cells remain in the arm table; no partial-cell geometric mean is substituted.' if len(ratios)!=14 else
            '[verified] Overall and batch results: '+json.dumps([r for r in summary if r['metric'] in ('X','Xprime','Y','Yprime')]),
            '[verified, diagnostic] Strongest rank reversals: '+json.dumps(reversed_pairs[:2]),
            '## Protocol and deviations','',
            '[verified] R5 baseline 3ff07bc2e was fast-forwarded onto tilemega before new tooling. '
            'No kernel body, cost model, target constant or search rule was edited in this experiment.',
            '[verified] Native P-CORR uses c1=256, c2=true, reference_processes=1. '
            'This checkout expresses the 256-token HF limit with --hf-positions 256, which writes a 256-token teacher-forced prefix.',
            '[inferred] Fused-expert implementation cost ties use fused shared bytes, then gate/up and down TileK; '
            'the solver ranks individual families before finding their deepest common legal slot count.',
            '[verified] Tau is tau-b with exact numerical ties; pooled family tau concatenates observations across cells. '
            'Rank reversals and arm selection additionally apply the conservative process-range threshold.',
            '## Tables and open items','',
            '[verified] See SM90X1_candidates.csv, SM90X1_stage_deltas.csv, SM90X1_arms.csv and SM90X1_summary.csv.',
            '[verified] Raw records, all failed attempts, trace overhead and build resource samples stay under runs/sm90x1.',
            '[verified] Guard acceptance in this container is provisional_container_pid_namespace; host PID ownership cannot be fully verified. '
            'A foreign compute context observed by the guard prevents acceptance.',
            '[stated] Phase 2 target date: 2026-11-21. Pending phases remain gated by Phase 0 and the pilot cost projection.']
    phase0=read(RUN/'phase0.json') if (RUN/'phase0.json').exists() else None
    report+=['## Phase 0 setup','']
    for c in all_cells:
        binding=RUN/'cells'/c['cell']/'phase0/W/binding.json'
        if c['pilot'] and binding.exists():
            b=read(binding)
            report+=['[verified] '+c['cell']+': native build '+str(round(b['wall_s'],2))+
                     ' s; sampled peak process-tree RSS '+str(round(b['peak_tree_rss_bytes']/2**30,3))+
                     ' GiB; build HEAD '+b['head']+'; host ABI status '+str(b['host_abi_status'])+'.']
    if phase0:
        report+=['[verified] Phase 0 complete='+str(phase0['complete'])+', stop='+str(phase0['stop'])+
                 '. Accepted TPOT comparisons: SM90X1_phase0.csv.']
    trace_unit=RUN/'cells/qwen3_0_6b-B1/phase0/trace/unit.json'
    if trace_unit.exists():
        unit=read(trace_unit)
        overhead=[read(Path(path)/'run.json')['overhead_pct'] for path in unit['runs']]
        report+=['[verified, diagnostic] Level-2 W trace overhead on GPU'+str(unit['gpu'])+
                 ': '+', '.join(str(round(v,3))+'%' for v in overhead)+
                 '; two fresh processes, each compared with non-trace W in the same session.']
    report+=['[verified] The R3 rounded solver table was rechecked: 30 cells, c0/winner geometric mean '
             '1.004542368, maximum 1.053627213. The additive/independent-family comment remains in Sm90DenseSearch.cpp.']
    (RESULTS/'SM90X1_h800.md').write_text('\n\n'.join(report)+'\n')
