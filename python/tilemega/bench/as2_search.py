"""Bounded outer implementation search through the native CTG/ISL compiler."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time

from tilemega.bench.as2_models import REPO, RUNS, WORK
from tilemega.bench.as2_native import digest
from tilemega.bench.as2_queue import controlled, write
from tilemega.bench.as1_run import process_identity
from tilemega.bench.as2_precision import REPLAY, STREAMING, STREAMING_FP32
from tilemega.bench.as2_native_prices import context_weights


def select(rows):
    legal=[r for r in rows if r['status']=='solved']
    if not legal:raise ValueError('No successfully solved candidate')
    for r in legal:
        n=r['predicted_forward_ns']
        if not isinstance(n,(int,float)) or not math.isfinite(n) or n<=0:
            raise ValueError('Unknown/nonpositive outer objective')
    return min(legal,key=lambda r:(r['predicted_forward_ns'],r['id']))


def check_compiler_identity(compiler,expected):
    if digest(compiler)!=expected:
        raise ValueError('Native compiler changed during the bounded candidate search')


def evidence(prices):
    p=json.loads(prices.read_text())
    if p['schema']!='tilemega.as2.native_task_prices.v1' or p['status']!='sourced_empirical_context':
        raise ValueError('Only positive measured native prices may enter search')
    for key in ('source_profile','source_costs'):
        source=controlled(p[key])
        if digest(source)!=p[key+'_sha256']:raise ValueError('Changed calibration provenance')
    source=json.loads(Path(p['source_profile']).read_text())
    cost=json.loads(Path(p['source_costs']).read_text())
    profiles=[source]
    if source.get('schema')=='tilemega.as2.native_body_contexts.v1':
        buckets=source['buckets'];weights=context_weights(buckets)
        if (source['status']!='measured_body_contexts' or source['phase']!='decode' or
                (source['past_min'],source['past_max'],source['forward_count'])!=(64,1086,1023) or
                (p['past_min'],p['past_max'])!=(64,1086) or
                p['context_aggregation']['bucket_counts']!=weights):
            raise ValueError('Unmatched decode context weights')
        profiles=[]
        for row in buckets:
            path=controlled(row['source_profile'])
            if digest(path)!=row['source_profile_sha256']:raise ValueError('Changed context profile provenance')
            profile=json.loads(path.read_text())
            if profile['phase']!='decode' or profile['past']!=row['past']:
                raise ValueError('Unmatched observed context')
            profiles.append(profile)
        if len({r['binary_sha256'] for r in profiles})!=1:
            raise ValueError('Mixed context implementation binaries')
    for profile in profiles:
        job=json.loads((controlled(profile['job'])/'job.json').read_text())
        if (profile['status']!='measured_body_only' or cost['status']!='measured_empirical_context' or
                profile['physical_device_id']!=cost['physical_device_id'] or
                profile['groups']!=cost['groups'] or p['groups']!=cost['groups'] or
                job['status']!='passed' or job['executable_sha256']!=profile['binary_sha256'] or
                job['physical_device_id']!=profile['physical_device_id']):
            raise ValueError('Unmatched measured Body/control context')
        for key in ('model_id','config_sha256','batch','phase','runtime_abi','precision_contract_id'):
            if key in profile and profile[key]!=p.get(key,REPLAY if key=='precision_contract_id' else None):
                raise ValueError('Context profile identity differs from native prices')
    return p,cost['physical_device_id']


def validate(domain):
    candidates=domain['candidates']
    if (domain['schema']!='tilemega.as2.outer_search_domain.v1' or not 1<=len(candidates)<=4 or
            not 0<domain.get('candidate_timeout_s',600)<=1800 or domain['batch'] not in (1,4,16)):
        raise ValueError('Require one to four candidates and a finite CPU solve budget')
    export=controlled(domain['export']);identity=json.loads((export/'as2_export.json').read_text())
    expected={k:identity[k] for k in ('model_id','config_sha256','phase')}
    expected['batch']=domain['batch']
    seen=set();signatures=set();costs={};devices=set();contexts=set()
    for c in candidates:
        if not re.fullmatch('[a-zA-Z0-9_-]+',c['id']) or c['id'] in seen:
            raise ValueError('Candidate ids must be unique safe path components')
        seen.add(c['id'])
        if c.get('body_binding') and c.get('decomposition'):
            raise ValueError('Body and CTG decomposition require separate native interfaces')
        if c['groups'] not in (2,4,8,14) or c['gemm_tile_n'] not in (64,128) or c['kv_block'] not in (128,1088):
            raise ValueError('Unadmitted topology or geometry')
        if c['groups']==14:
            path=Path(domain.get('target_config','')).absolute()
            if path.resolve()!=path or path.parent!=REPO/'configs/targets':
                raise ValueError('Unadmitted topology: G14 requires an explicit project target and local protocol evidence')
            target=json.loads(path.read_text());admission=target.get('topology_experiment',{})
            source=Path(admission.get('source_summary',''))
            source=controlled(source if source.is_absolute() else REPO/source)
            if (target.get('max_verified_mix_groups')!=14 or admission.get('default_enabled') is not False or
                    digest(source)!=admission.get('source_summary_sha256')):
                raise ValueError('Changed/unadmitted G14 topology provenance')
            proof=json.loads(source.read_text())
            if proof.get('status')!='pass' or proof.get('processes_passed_per_group',0)<5 or 14 not in proof.get('groups',[]):
                raise ValueError('G14 requires five-process payload evidence')
        signature=tuple(c.get(k) for k in ('groups','gemm_tile_n','kv_block','attention',
            'body_binding','decomposition','fifo_wait_elision'))
        if signature in signatures:raise ValueError('Duplicate implementation/plan candidate')
        signatures.add(signature)
        p,device=evidence(controlled(c['prices']));devices.add(device)
        if p.get('precision_contract_id',REPLAY)!=(STREAMING if c['attention']=='aiv-bf16-streaming' else STREAMING_FP32 if c['attention']=='aiv-fp32-streaming' else REPLAY):
            raise ValueError('Unmatched precision contract in native search prices')
        if c['groups']==14 and device!=proof['physical_device_id']:
            raise ValueError('G14 pricing and protocol evidence use different physical cards')
        if len(devices)>1:raise ValueError('Mixed device pricing context')
        if any(p[k]!=v for k,v in expected.items()) or p['groups']!=c['groups']:
            raise ValueError('Cost identity differs from the imported model/phase/B/G')
        contexts.add((p['runtime_abi'],p['past_min'],p['past_max']))
        # Separate G measurements are allowed. Same-G alternatives share one
        # dispatcher/GM calibration instead of being ranked by different noise.
        old=costs.setdefault(c['groups'],p['source_costs_sha256'])
        if old!=p['source_costs_sha256']:raise ValueError('Same-G candidates use different control calibration')
        for name in ('body_binding','decomposition'):
            if c.get(name):
                path=Path(c[name]).absolute()
                if path.resolve()!=path or not path.is_relative_to(REPO/'configs/ascend') or not path.is_file():
                    raise ValueError('Require a checked-in native implementation binding')
        binding=json.loads(Path(c['decomposition']).read_text()) if c.get('decomposition') else {}
        if binding.get('precision_contract_id',p.get('precision_contract_id'))!=p.get('precision_contract_id'):
            raise ValueError('Implementation binding and measured precision contracts differ')
        if binding.get('vector_argmax') and not p.get('precision_contract_id'):
            raise ValueError('New argmax requires explicitly measured precision identity')
        l1=binding.get('swiglu_l1_k',128)
        if (l1 not in (128,256) or l1==256 and not binding.get('swiglu_tile') or
                p.get('swiglu_l1_k',128)!=l1):
            raise ValueError('SwiGLU L1 implementation differs from measured native prices')
    if len(devices)!=1 or len(contexts)!=1:raise ValueError('Mixed device or runtime/past pricing context')
    return export,expected


def solve(domain_path,out,compiler):
    domain_path,out=controlled(domain_path),controlled(out)
    if out.exists():raise FileExistsError(out)
    compiler=Path(compiler).absolute()
    if compiler.resolve()!=compiler or not compiler.is_relative_to(WORK/'build') or compiler.name!='as2_compile_llm':
        raise ValueError('Use the existing private native compiler')
    domain=json.loads(domain_path.read_text());export,identity=validate(domain)
    out.mkdir();rows=[]
    summary=dict(schema='tilemega.as2.outer_search_result.v1',status='in_progress',identity=identity,
        domain=str(domain_path),domain_sha256=digest(domain_path),compiler_sha256=digest(compiler),
        scope='bounded whole-phase implementation/decomposition/topology outside native heterogeneous placement/order search',
        objective='minimum sourced predicted forward makespan at the declared pricing context',
        end_to_end_performance_admitted=False,automatic_runtime_default_changed=False,rows=rows)
    if domain.get('target_config'):
        target=Path(domain['target_config']).absolute()
        if target.resolve()!=target or target.parent!=REPO/'configs/targets':
            raise ValueError('Explicit target must remain in project configs/targets')
        summary.update(target_config=str(target),target_config_sha256=digest(target))
    for c in domain['candidates']:
        check_compiler_identity(compiler,summary['compiler_sha256'])
        disk=subprocess.run([sys.executable,str(REPO/'scripts/as2_disk_status.py')],capture_output=True,text=True)
        (out/(c['id']+'_disk.json')).write_text(disk.stdout)
        if disk.returncode:
            summary['status']='blocked-resource';write(out/'search.json',summary);return 75
        command='emit-priced-body' if c.get('body_binding') else 'emit-priced-decomposed' if c.get('decomposition') else 'emit-priced'
        if c.get('fifo_wait_elision',False):command+='-fifo'
        path=out/c['id']
        resource_log=out/(c['id']+'_resources.txt')
        argv=['/usr/bin/time','-v','-o',str(resource_log),'nice','-n','10',str(compiler),command,str(export),str(path),str(domain['batch']),
            str(c['groups']),str(c['kv_block']),str(c['gemm_tile_n']),c['attention'],c['prices']]
        if domain.get('target_config'):
            at=argv.index(str(compiler))+1
            argv[at:at]=['--target',str(target)]
        argv.extend([c[k] for k in ('body_binding','decomposition') if c.get(k)])
        row=dict(id=c['id'],argv=argv,status='running',prices_sha256=digest(Path(c['prices'])),
            candidate=c,output=str(path));rows.append(row);write(out/'search.json',summary)
        start=time.monotonic()
        with (out/(c['id']+'.log')).open('x') as log:
            with subprocess.Popen(argv,stdout=log,stderr=log,start_new_session=True,
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')) as child:
                owned=process_identity(child.pid)
                try:
                    row['cpu_process']=owned;write(out/'search.json',summary)
                    code=child.wait(timeout=domain.get('candidate_timeout_s',600))
                    row.update(returncode=code,status='rejected' if code else 'solved')
                except BaseException as exc:
                    # time/nice/native compiler share only this newly created
                    # session. Never terminate by UID or executable name.
                    if child.poll() is None and process_identity(child.pid)==owned:
                        os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if process_identity(child.pid)==owned:os.killpg(child.pid,signal.SIGKILL)
                        child.wait(timeout=5)
                    if not isinstance(exc,subprocess.TimeoutExpired):raise
                    row.update(returncode=124,status='timeout')
        row['offline_wall_s']=time.monotonic()-start
        check_compiler_identity(compiler,summary['compiler_sha256'])
        row['peak_rss_kib']=None
        if resource_log.exists():
            match=re.search(r'Maximum resident set size \(kbytes\): (\d+)',resource_log.read_text())
            if match:row['peak_rss_kib']=int(match.group(1))
        if row['status']=='solved':
            imported=json.loads((path/'import.json').read_text());inner=json.loads((path/'search.json').read_text())
            if not imported['priced_solver_executed'] or not imported['native_plan_materialized']:
                raise ValueError('Native compiler did not execute the priced semantic pipeline')
            row.update(predicted_forward_ns=min(inner['predicted_eft_ns'],inner['predicted_structural_ns']),
                inner_selection=inner['selected'],compared_inner_plans=inner['compared_legal_plans'],
                source_sha256=digest(path/'persistent_llm.cpp'),
                tables_sha256={p.name:digest(p) for p in sorted((path/'plan').glob('*.bin'))})
        write(out/'search.json',summary)
    if not any(r['status']=='solved' for r in rows):
        summary['status']='fail';write(out/'search.json',summary);return 1
    winner=select(rows)
    summary.update(status='selected_prediction_not_device_admitted',selected=winner['id'],
        selected_plan=str(Path(winner['output'])/'plan'),selected_source=str(Path(winner['output'])/'persistent_llm.cpp'),
        note='Each successful candidate ran Import/CTG/ISL/native scheduling/materialization. No device launch, benchmark, weight packing or default promotion occurs here.')
    write(out/'search.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}));return 0


def select_request(rows,output_count):
    if output_count!=1024:raise ValueError('Request search requires the complete 64/1024 objective')
    legal=[r for r in rows if r['status']=='solved']
    for r in legal:
        phases=[r[k] for k in ('predicted_prefill_ns','predicted_decode_ns')]
        if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in phases):
            raise ValueError('Unknown/nonpositive request phase price')
        r['predicted_request_ns']=phases[0]+(output_count-1)*phases[1]
    if not legal:raise ValueError('No successfully solved compatible request recipe')
    return min(legal,key=lambda r:(r['predicted_request_ns'],r['id']))


def phase_pair_contract(prefill,decode):
    from tilemega.bench.as2_request import cached_kv_state,request_source
    _,source=request_source(prefill,decode)
    states=[]
    for path in (prefill,decode):
        m=struct.unpack('<12I3Q',(path/'metadata.bin').read_bytes())
        stages=list(struct.iter_unpack('<17Q16I',(path/'stages.bin').read_bytes()))
        buffers=json.loads((path/'buffers.json').read_text())['buffers']
        states.append(cached_kv_state(stages,buffers,m))
    if states[0]!=states[1]:
        raise ValueError('Request recipe requires matching prefill/decode derived KV state')
    return dict(derived_kv_layers=len(states[0]),generated_source_contract=source)


def request_phase_key(candidate):
    # Prices are part of this identity: identical math with two independently
    # measured prices must not silently reuse whichever measurement came first.
    return json.dumps({k:v for k,v in candidate.items() if k!='id'},sort_keys=True)


def request_phase_inputs(candidate):
    return {k:digest(Path(candidate[k])) for k in ('prices','body_binding','decomposition') if candidate.get(k)}


def check_reused_phase(original,inputs,expected):
    if inputs!=expected or digest(Path(original['search']))!=original['search_sha256']:
        raise ValueError('Shared request phase inputs or solved receipt changed')
    result=json.loads(Path(original['search']).read_text())
    selected=next(r for r in result['rows'] if r['id']==result['selected'])
    for name,sha in selected['tables_sha256'].items():
        if digest(Path(original['plan'])/name)!=sha:
            raise ValueError('Shared request phase Plan changed')


def request_domains(domain):
    recipes=domain['recipes']
    if (domain['schema']!='tilemega.as2.outer_request_search_domain.v1' or
            not 1<=len(recipes)<=2 or domain['output_count']!=1024):
        raise ValueError('At most two request recipes/four compiled phase plans per batch')
    if len({r['id'] for r in recipes})!=len(recipes) or any(not re.fullmatch('[a-zA-Z0-9_-]+',r['id']) for r in recipes):
        raise ValueError('Request recipe ids must be unique safe path components')
    domains={};identities=[];devices=set();costs={}
    for phase in ('prefill','decode'):
        unique={}
        for r in recipes:unique.setdefault(request_phase_key(r[phase]),dict(r[phase],id=r['id']))
        child=dict(schema='tilemega.as2.outer_search_domain.v1',export=domain[phase+'_export'],
            batch=domain['batch'],candidate_timeout_s=domain.get('candidate_timeout_s',600),
            candidates=list(unique.values()))
        if domain.get('target_config'):child['target_config']=domain['target_config']
        _,identity=validate(child)
        if identity['phase']!=phase:raise ValueError('Request phase export mismatch')
        identities.append({k:v for k,v in identity.items() if k!='phase'})
        for c in child['candidates']:
            p,device=evidence(controlled(c['prices']));devices.add(device)
            if phase=='prefill' and (p['past_min'],p['past_max'])!=(0,0) or phase=='decode' and not p['past_min']<=64<=1086<=p['past_max']:
                raise ValueError('Request prices must cover all actual forward contexts')
            old=costs.setdefault(c['groups'],p['source_costs_sha256'])
            if old!=p['source_costs_sha256']:raise ValueError('Request recipes use different same-G control calibration')
        domains[phase]=child
    if identities[0]!=identities[1] or len(devices)!=1:
        raise ValueError('Request recipes have different models/devices')
    return domains,identities[0]


def solve_request(domain_path,out,compiler):
    domain_path,out=controlled(domain_path),controlled(out)
    if out.exists():raise FileExistsError(out)
    domain=json.loads(domain_path.read_text());domains,identity=request_domains(domain)
    out.mkdir();rows=[];phase_cache={}
    summary=dict(schema='tilemega.as2.outer_request_search_result.v1',status='in_progress',identity=identity,
        domain=str(domain_path),domain_sha256=digest(domain_path),compiler_sha256=digest(compiler),
        objective='predicted_prefill_ns + 1023 * context_weighted_predicted_decode_ns',
        output_count=1024,rows=rows,end_to_end_performance_admitted=False,automatic_runtime_default_changed=False)
    for recipe in domain['recipes']:
        row=dict(id=recipe['id'],status='running',phases={});rows.append(row)
        recipe_out=out/recipe['id'];recipe_out.mkdir()
        for phase in ('prefill','decode'):
            check_compiler_identity(compiler,summary['compiler_sha256'])
            key=(phase,request_phase_key(recipe[phase]))
            inputs=request_phase_inputs(recipe[phase])
            if key in phase_cache:
                original,price,expected=phase_cache[key]
                check_reused_phase(original,inputs,expected)
                row['phases'][phase]=dict(original,reused_solved_phase=True)
                row['predicted_'+phase+'_ns']=price
                continue
            child=dict(domains[phase],candidates=[dict(recipe[phase],id=recipe['id'])])
            path=recipe_out/(phase+'_domain.json');write(path,child)
            code=solve(path,recipe_out/phase,compiler)
            result=json.loads((recipe_out/phase/'search.json').read_text())
            row['phases'][phase]=dict(search=str(recipe_out/phase/'search.json'),search_sha256=digest(recipe_out/phase/'search.json'))
            if code:
                row.update(status='rejected',reason='native '+phase+' search did not produce a legal plan')
                if code==75:
                    summary['status']='blocked-resource';write(out/'search.json',summary);return 75
                break
            winner=next(r for r in result['rows'] if r['id']==result['selected'])
            row['phases'][phase]['plan']=result['selected_plan']
            row['predicted_'+phase+'_ns']=winner['predicted_forward_ns']
            phase_cache[key]=(dict(row['phases'][phase]),winner['predicted_forward_ns'],inputs)
        if row['status']=='running':
            try:
                row['pair_contract']=phase_pair_contract(*(Path(row['phases'][p]['plan']) for p in ('prefill','decode')))
                row['status']='solved'
            except ValueError as exc:
                row.update(status='rejected',reason=str(exc))
        write(out/'search.json',summary)
    if not any(r['status']=='solved' for r in rows):
        summary['status']='fail';write(out/'search.json',summary);return 1
    selected=select_request(rows,domain['output_count'])
    summary.update(status='selected_prediction_not_device_admitted',selected=selected['id'],
        unique_solved_phases=len(phase_cache),
        selected_phases={p:r['plan'] for p,r in selected['phases'].items()},
        note='Both phases used native Import/CTG/ISL/priced schedule/materialization; state compatibility checked before joint recipe selection. No device/default admission.')
    write(out/'search.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('domain','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compiler',type=Path,default=WORK/'build/host/as2_compile_llm')
    a=p.parse_args()
    request=json.loads(controlled(a.domain).read_text())['schema']=='tilemega.as2.outer_request_search_domain.v1'
    raise SystemExit((solve_request if request else solve)(a.domain,a.out,a.compiler))
