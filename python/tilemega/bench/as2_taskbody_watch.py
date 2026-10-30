"""One finite, idle-admitted TaskBody acceptance submitter; no device reservation."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import sys
import time
import traceback

from tilemega.bench.as1_run import process_identity
from tilemega.bench.as2_ablation import paired_interval
from tilemega.bench.as2_finite_process import Deadline,run as finite_run
from tilemega.bench.as2_queue import REPO,RUNS,WORK,controlled,write,idle_devices,collect_samples,serious_failure
from tilemega.bench.as2_request import source_parts,SOURCE_CAPABILITIES,SOURCE_GEOMETRY,SOURCE_CONTRACT
from tilemega.bench.as2_run import unfinished_owned_jobs

PRECISION='as2_llm_fp32_replay_bf16_casts_v1'
MODULE='tilemega.bench.as2_taskbody_watch'


class DeviceUnsafe(BaseException):pass


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(directory):
    directory=controlled(directory);directory.mkdir(exist_ok=False)
    controls=json.loads((RUNS/'inventory/taskbody_r0_v1/controls.json').read_text())['controls']
    cells=[]
    for control in controls:
        binding=Path(control['binding']);identity=json.loads((binding/'request.json').read_text());phases={}
        for phase in ('prefill','decode'):
            plan=Path(identity['phases'][phase]['source_plan']);origin=json.loads((plan.parent/'import.json').read_text())
            body=origin.get('body_binding_candidate')
            recipe=dict(body or origin.get('decomposition_candidate') or dict(
                schema='tilemega.as2.implementation_binding.v1',model_id=identity['model_id'],
                batch=identity['batch'],phase=phase,families=[]))
            recipe.update(swiglu_tile=True,vector_argmax=True,precision_contract_id=PRECISION,
                default_enabled=False,acceptance_only=True)
            path=REPO/'configs/ascend'/f'as2_taskbody_{control["cell"]}_{phase}.json'
            if path.exists():
                if json.loads(path.read_text())!=recipe:raise ValueError('Existing recipe differs: '+str(path))
            else:path.write_text(json.dumps(recipe,indent=2)+'\n')
            attention=origin['attention_implementation']
            mode={'mix_fp32_cube_newton_norm_fold_softmax1088':'cube-fp32-fold-softmax',
                  'mix_fp32_cached_kv_valid_qk64_fold_softmax1088':'cube-fp32-cached-kv'}.get(attention)
            if not mode:raise ValueError('Unregistered control attention '+attention)
            phases[phase]=dict(source_plan=str(plan),config=str(path),config_sha256=digest(path),
                body=bool(body),attention=mode)
        cells.append(dict(cell=control['cell'],model=identity['model_id'],batch=identity['batch'],
            control=str(binding),control_sha256=digest(binding/'request.json'),inputs=identity['source_inputs'],
            weight_manifest_sha256=identity['weight_manifest_sha256'],phases=phases))
    write(directory/'protocol.json',dict(schema='tilemega.as2.taskbody_acceptance_watch.v1',
        prepared_at=dt.datetime.now(dt.timezone.utc).isoformat(),hours=8,pairs=5,warmup=1,repetitions=3,
        isl=64,osl=1024,confidence_method='paired_log_t_95',max_process_cv=.05,
        idle_observations=3,idle_interval_s=30,maximum_idle_utilization_percent=0,
        inherited_safety_holds=json.loads((RUNS/'resume.json').read_text()).get('safety_holds',[]),
        reentry='User explicitly requested bounded testing after healthy repeated idle observations; old device exit is not relabelled verified',
        frozen_code={str(p):digest(p) for p in sorted((REPO/'python/tilemega/bench').glob('as[12]_*.py'))},
        compiler_sha256=digest(WORK/'build/host/as2_compile_llm'),
        baseline_rerun=False,default_promotion=False,cells=cells))
    print(json.dumps(dict(directory=str(directory),cells=len(cells),new_NPU_jobs=0)))


def interval_summary(rows):
    if len(rows)!=10:return dict(status='incomplete',pairs_complete=sum(
        {v['variant'] for v in rows if v['pair']==i}=={'control','candidate'} for i in range(5)))
    identities={(r['receipt']['pid'],r['receipt']['start_ticks']) for r in rows}
    if len(identities)!=10 or len({r['receipt']['physical_device_id'] for r in rows})!=1:
        raise ValueError('Paired processes/device identities differ')
    ratios=[]
    for pair in range(5):
        p={r['variant']:r for r in rows if r['pair']==pair}
        ratios.append(p['control']['metrics']['tpot_ms']/p['candidate']['metrics']['tpot_ms'])
    values={v:[r['metrics']['tpot_ms'] for r in rows if r['variant']==v] for v in ('control','candidate')}
    cv={v:statistics.stdev(x)/statistics.mean(x) for v,x in values.items()};ci=paired_interval(ratios)
    return dict(status='inconclusive' if max(cv.values())>.05 else 'pass_local_gain' if ci['lower95']>1 else 'no_reliable_gain',
        interval=ci,process_cv=cv,process_pair_speedups=ratios,
        medians={v:{k:statistics.median(r['metrics'][k] for r in rows if r['variant']==v)
                    for k in ('tpot_ms','ttft_ms','e2e_ms','throughput_tokens_per_s')} for v in values},
        formal_AS2_eager_G_PERF='not_implied; frozen framework baselines are descriptive only')


class Watch:
    def __init__(self,directory,hours):
        self.root=controlled(directory);self.protocol=json.loads((self.root/'protocol.json').read_text())
        if not 0<hours<=8:raise ValueError('Maximum eight hours including active jobs')
        previous=self.root/'state.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            if old.get('steps') or old.get('jobs'):raise ValueError('Use a fresh watch directory; never overwrite attempts')
            try:active=process_identity(old['identity']['pid'])==old['identity']
            except (FileNotFoundError,ProcessLookupError):active=False
            if active:raise ValueError('This watch preparation process is still active')
        self.hard_end=time.monotonic()+hours*3600;self.end=self.hard_end-10
        self.state=dict(status='starting',identity=process_identity(os.getpid()),started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            deadline_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=hours)).isoformat(),
            finite_hours=hours,protocol_sha256=digest(self.root/'protocol.json'),steps={},jobs=[],cells={},new_NPU_jobs=0,
            device_locked=False,baseline_rerun=False,default_promotion=False)
        self.physical=None;self.counter=0;self.ready_cards=set();self.costs={}
        self.root.joinpath('logs').mkdir(exist_ok=True);self.root.joinpath('observations').mkdir(exist_ok=True)
        write(self.root/'state.json',self.state)

    def save(self):write(self.root/'state.json',self.state)

    def command(self,name,argv,timeout=600,job=None):
        if time.monotonic()>=self.end:raise Deadline('Eight-hour acceptance window ended')
        if (any(digest(p)!=sha for p,sha in self.protocol['frozen_code'].items()) or
                digest(WORK/'build/host/as2_compile_llm')!=self.protocol['compiler_sha256']):
            raise DeviceUnsafe('Frozen helper/compiler changed; refusing mixed-code measurements')
        self.counter+=1;step=dict(name=name,argv=list(map(str,argv)),status='running',kind='NPU' if job else 'CPU')
        key=f'{self.counter:04d}_{name}';self.state['steps'][key]=step;self.state['active_step']=key;self.save()
        def started(identity):step['identity']=identity;self.save()
        env=os.environ.copy();env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
        try:
            code=finite_run(argv,self.root/'logs'/f'{key}.log',self.end,timeout,env=env,on_identity=started)
            step.update(status='pass' if code==0 else 'fail',exit_code=code)
            if code:raise RuntimeError(name+' failed; see '+str(self.root/'logs'/f'{key}.log'))
        except (Deadline,subprocess.TimeoutExpired) as exc:
            step.update(status='deadline_terminated' if isinstance(exc,Deadline) else 'timeout',error=str(exc))
            self.save()
            if isinstance(exc,Deadline):raise
            if job:raise DeviceUnsafe('Own job timeout; device exit unknown')
            raise RuntimeError('Finite CPU helper timeout') from exc
        finally:self.state.pop('active_step',None);self.save()

    def cpu(self,name,module,*args):
        self.command(name,[sys.executable,'-m','tilemega.bench.'+module,*map(str,args)])

    def disk(self,reserve=64<<20):
        self.counter+=1;out=self.root/f'disk_{self.counter:04d}.json'
        self.command('disk', [sys.executable,'-m',MODULE,'disk','--out',out],timeout=30)
        r=json.loads(out.read_text())
        if r['warning'] or r['generated_bytes']+reserve>r['generated_soft_budget_bytes']:
            raise DeviceUnsafe('Project storage budget insufficient; no further jobs')

    def choose(self,required,physical=None):
        consecutive={};pinned=physical
        self.state['status']='waiting_healthy_idle';self.save()
        while time.monotonic()<self.end:
            if unfinished_owned_jobs():
                self.state['waiting_reason']='Existing owned NPU submission is still active';self.save()
            else:
                self.counter+=1;out=self.root/'observations'/f'{self.counter:05d}.json'
                self.command('observe',[sys.executable,'-m',MODULE,'observe','--out',out],timeout=15)
                observed=json.loads(out.read_text());eligible=idle_devices(observed,required,0)
                consecutive={i:consecutive.get(i,0)+1 for i in eligible if pinned is None or i==pinned}
                admitted=[i for i,n in consecutive.items() if n>=self.protocol['idle_observations']]
                if admitted:
                    chosen=max(admitted,key=lambda i:eligible[i]['free_mib'])
                    self.state['last_idle_admission']=dict(physical_device_id=chosen,evidence=str(out),observations=consecutive[chosen],
                        previous_device_exit='not_proven_by_idle; fresh bounded reentry authorized by user')
                    self.state['status']='running';self.save();return chosen
                self.state['waiting_reason']='Require HealthOK, util0% three times, sufficient HBM and no active owned job; resident external contexts are not killed'
                self.save()
            time.sleep(min(self.protocol['idle_interval_s'],max(0,self.end-time.monotonic())))
        raise Deadline('Eight-hour window expired while waiting for idle hardware')

    def npu(self,name,argv,out,required,physical=None,timeout=600):
        self.disk();card=self.choose(required,physical)
        jobdir=RUNS/'jobs'/f'taskbody8h_{self.root.name}_{name}_{time.time_ns()}'
        script='source "$1" || exit; shift; unset TORCH_DEVICE_BACKEND_AUTOLOAD; exec python -m tilemega.bench.as2_run "$@"'
        command=['bash','--noprofile','--norc','-c',script,'as2-acceptance',str(REPO/'scripts/as2_env.sh'),
            '--physical-device',str(card),'--required-free-mib',str(required),'--timeout-s',str(timeout),
            '--telemetry-timeout-s','5','--out',str(jobdir),'--',*map(str,argv)]
        record=dict(name=name,job_directory=str(jobdir),output=str(out),physical_device_id=card,argv=list(map(str,argv)))
        self.state['jobs'].append(record);self.state['new_NPU_jobs']+=1;self.save()
        try:self.command(name,command,timeout=timeout+20,job=record)
        except RuntimeError:
            receipt=json.loads((jobdir/'job.json').read_text()) if (jobdir/'job.json').exists() else {}
            record['receipt']=receipt;self.save()
            reason=serious_failure(jobdir,receipt)
            if reason:raise DeviceUnsafe(reason)
            raise
        receipt=json.loads((jobdir/'job.json').read_text());record['receipt']=receipt;self.save()
        reason=serious_failure(jobdir,receipt)
        if reason:raise DeviceUnsafe(reason)
        for path in (jobdir,Path(out)):
            if path.exists():self.command('register',[sys.executable,str(REPO/'scripts/as2_register.py'),
                '--path',str(path),'--stage','taskbody8h','--category','result','--completed'],timeout=30)
        if len(argv)==8 and argv[1]=='request':
            same=[r['job_directory'] for r in self.state['jobs'] if len(r['argv'])==8 and
                r['argv'][1]=='request' and r['argv'][2]==str(argv[2]) and r.get('receipt',{}).get('status')=='passed']
            for apply in (False,True):
                self.counter+=1;report=self.root/f'dedup_{self.counter:05d}.json'
                cmd=[sys.executable,str(REPO/'scripts/as2_dedup.py'),'--completed-jobs',*same[-8:],
                    '--min-bytes','65536','--out',str(report)]
                if apply:cmd.append('--apply')
                self.command('share_immutable_outputs',cmd,timeout=120)
        return record

    def card_ready(self,card):
        if card in self.ready_cards:return
        for i in range(5):
            out=self.root/f'argmax_card{card}_p{i}';self.npu(f'argmax_c{card}_p{i}',
                [WORK/'build/taskbody_argmax_v2/as2_argmax',out],out,1024,card,60)
            r=json.loads((out/'result.json').read_text())
            if r.get('status')!='passed' or r.get('cases')!=12:raise DeviceUnsafe('New argmax Body probe failed')
        jobs=[]
        for i in range(5):
            out=self.root/f'cost_card{card}_p{i}';r=self.npu(f'cost_c{card}_p{i}',
                [WORK/'build/probes/as2_native_cost'],out,2048,card,120);jobs.append(r['job_directory'])
        path=self.root/f'costs_card{card}.json'
        self.cpu('scheduling_costs','as2_native_prices','costs','--jobs',*jobs,'--out',path)
        self.costs[card]=path;self.ready_cards.add(card)

    def register(self,path,category='source'):
        self.command('register',[sys.executable,str(REPO/'scripts/as2_register.py'),'--path',str(path),
            '--stage','taskbody8h','--category',category,'--completed'],timeout=30)

    def cleanup(self):
        for mode in ('--dry-run','--apply'):
            self.counter+=1;path=self.root/f'cleanup_{self.counter:04d}.json'
            self.command('cleanup',[sys.executable,str(REPO/'scripts/as2_cleanup.py'),mode,
                '--report-json',str(path)],timeout=60)

    def compile_catalogue(self):
        compiler=WORK/'build/host/as2_compile_llm';sources=[]
        for cell in self.protocol['cells']:
            for phase,item in cell['phases'].items():
                out=self.root/f'catalogue_{cell["cell"]}_{phase}'
                self.command('reemit',[compiler,'reemit',item['source_plan'],out])
                sources.append(out/'persistent_llm.cpp')
        sources.append(RUNS/'models/qwen3_0_6b/taskbody_argmax_b1_decode_bootstrap_v2/persistent_llm.cpp')
        parts=[source_parts(p) for p in sources];fixed=SOURCE_GEOMETRY+SOURCE_CONTRACT
        if any(p[2]!=parts[0][2] or any(p[1][k]!=parts[0][1][k] for k in fixed) for p in parts):
            raise ValueError('Catalogue runtime/precision mismatch')
        fields={k:max(p[1].get(k,0) for p in parts) for k in SOURCE_CAPABILITIES}
        fields.update({k:parts[0][1][k] for k in fixed})
        generated=self.root/'catalogue.cpp';generated.write_text(''.join(f'#define {k} {v}\n' for k,v in fields.items())+parts[0][2])
        headers=list((REPO/'include/tilemega/Backend/Ascend').glob('*.h'))+list((REPO/'include/tilemega/Target/Ascend').glob('*.h'))
        before={str(p):digest(p) for p in headers}
        write(self.root/'build_inputs.json',before)
        self.build=WORK/'build'/self.root.name
        self.command('configure',['cmake','-S',REPO/'tools/ascend','-B',self.build,'-G','Ninja',
            '-DAS2_GENERATED_SOURCE='+str(generated),'-DAS2_COMPRESS_BODY_PROFILES=ON',
            '-DAS2_PROFILE_BOUNDARY_FORWARDS=ON',
            '-DAS2_CANN_NORM_PRIMITIVES=/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/norm_common'])
        self.command('compile',['cmake','--build',self.build,'--target','as2_llm','--parallel','1'])
        if before!={str(p):digest(p) for p in headers}:raise ValueError('Headers changed during compilation')
        self.image=self.build/'as2_llm';self.state['binary_sha256']=digest(self.image);self.save();self.register(self.build,'build')

    def reference(self,cell):
        identity=json.loads((Path(cell['control'])/'request.json').read_text())
        for p in (RUNS/'correctness').iterdir():
            meta=p/'result.json'
            if not meta.is_file():continue
            r=json.loads(meta.read_text())
            if (r.get('system')=='torch_npu_eager' and r.get('osl')==1024 and r.get('warmup')==0 and
                    r.get('repetitions')==1 and all(r.get(k)==identity[k] for k in
                        ('model_id','batch','config_sha256','weight_manifest_sha256','input_sha256'))):return p
        raise ValueError('No retained independent full1024 reference for '+cell['cell'])

    def cell(self,cell):
        key=cell['cell'];root=self.root/key;root.mkdir();self.disk(200<<20)
        state=self.state['cells'][key];control=Path(cell['control']);identity=json.loads((control/'request.json').read_text())
        if digest(control/'request.json')!=cell['control_sha256']:raise ValueError('Control changed')
        manifest=WORK/'weights'/cell['model']/'manifest.json'
        if digest(manifest)!=cell['weight_manifest_sha256']:raise ValueError('Canonical weights changed')
        cfg=json.loads(manifest.read_text())['config'];compiler=WORK/'build/host/as2_compile_llm'
        def solve(phase,priced=False):
            item=cell['phases'][phase];out=root/((phase+'_priced') if priced else (phase+'_bootstrap'))
            if digest(item['config'])!=item['config_sha256']:raise ValueError('Recipe changed')
            mode=('emit-priced-body' if priced else 'emit-body') if item['body'] else ('emit-priced-decomposed' if priced else 'emit-decomposed')
            cmd=[compiler,mode,RUNS/'models'/cell['model']/('export_'+phase),out,str(cell['batch']),'8','1088','128',item['attention']]
            if priced:cmd.append(root/(phase+'_prices.json'))
            cmd.append(item['config']);self.command('solve_'+phase,cmd)
            for name in ('ctg.mlir','symbolic_dependencies.isl'):
                if (out/name).read_bytes()!=(Path(item['source_plan']).parent/name).read_bytes():
                    raise ValueError('Implementation-only candidate changed CTG/ISL '+name)
            self.register(out);return out/'plan'
        plans={p:solve(p) for p in ('prefill','decode')}
        bootstrap=root/'bootstrap_request'
        self.cpu('bind_bootstrap','as2_request','--prefill',plans['prefill'],'--decode',plans['decode'],
            '--inputs',cell['inputs'],'--out',bootstrap)
        self.register(bootstrap)
        m=json.loads((bootstrap/'request.json').read_text());required=math.ceil(m['arena_bytes']/(1<<20))+6144
        card=self.choose(required);state['physical_device_id']=card;self.save();self.card_ready(card)
        ref=self.reference(cell)
        profile=root/'profile1024'
        record=self.npu(key+'_profile',[self.image,'request',bootstrap,profile,'1024','0','1','2'],profile,required,card,900)
        numeric=root/'bootstrap_numeric.json'
        self.cpu('bootstrap_C1','as2_compare_request','--binding',bootstrap,'--native',profile,'--reference',ref,'--out',numeric)
        for phase in ('prefill','decode'):
            output=root/(phase+'_profile.json')
            self.cpu('extract_'+phase,'as2_body_profile','--binding',bootstrap,'--job',record['job_directory'],'--phase',phase,'--out',output)
            self.cpu('price_'+phase,'as2_native_prices','prices','--plan',plans[phase],'--profile',output,
                '--costs',self.costs[card],'--out',root/(phase+'_prices.json'))
        plans={p:solve(p,True) for p in ('prefill','decode')};binding=root/'priced_request'
        self.cpu('bind_priced','as2_request','--prefill',plans['prefill'],'--decode',plans['decode'],
            '--inputs',cell['inputs'],'--out',binding);self.register(binding)
        state['binding']=str(binding);state['status']='correctness';self.save()
        actual=root/'full1024';self.npu(key+'_C1',[self.image,'request',binding,actual,'1024','0','1','0'],actual,required,card,900)
        self.cpu('priced_C1','as2_compare_request','--binding',binding,'--native',actual,'--reference',ref,'--out',root/'C1.json')
        jobs=[]
        for mode in (0,1):
            out=root/f'C2_mode{mode}';r=self.npu(key+f'_C2_{mode}',[self.image,'request',binding,out,'4','0','1',str(mode)],out,required,card)
            jobs.append(r['job_directory'])
        self.cpu('C2_audit','as2_native_audit','--compare-request',binding,'--jobs',*jobs,'--out',root/'C2.json')
        jobs=[]
        for i in range(5):
            out=root/f'C3_p{i}';r=self.npu(key+f'_C3_{i}',[self.image,'request',binding,out,'4','1','1','0'],out,required,card)
            jobs.append(r['job_directory'])
        self.cpu('C3_audit','as2_native_audit','--request',binding,'--jobs',*jobs,'--out',root/'C3.json')
        trace=root/'teacher_trace';self.cpu('teacher_trace','as2_teacher','prepare','--model',cell['model'],'--out',trace)
        refteacher=None;trace_hash=digest(trace/'trace.json')
        for p in (RUNS/'correctness').iterdir():
            file=p/'teacher.json'
            if file.exists():
                r=json.loads(file.read_text())
                if (r.get('system')=='torch_npu_eager' and r.get('batch')==cell['batch'] and
                        r.get('decoder_steps')==512 and r.get('trace_sha256')==trace_hash):refteacher=p;break
        if refteacher is None:
            refteacher=root/'teacher_reference'
            self.npu(key+'_teacher_reference',[sys.executable,'-m','tilemega.bench.as2_teacher','reference',
                '--trace',trace,'--batch',str(cell['batch']),'--steps','512','--out',refteacher],refteacher,required,card,900)
        teacher=root/'teacher512';self.npu(key+'_teacher512',[self.image,'teacher',binding,trace,teacher,'512'],teacher,required,card,900)
        self.cpu('teacher_compare','as2_teacher','compare','--trace',trace,'--binding',binding,'--native',teacher,
            '--reference',refteacher,'--out',root/'teacher512.json')
        state.update(status='timing',C1='pass',C2='pass',C3='5/5',teacher512='pass',samples=[]);self.save()
        # The pair count/order is fixed before any timed sample; no optional stopping.
        write(root/'timing_protocol.json',dict(pairs=5,order=[['control','candidate'] if i%2==0 else ['candidate','control'] for i in range(5)],
            primary_metric='tpot_ms',confidence_method='paired_log_t_95',max_process_cv=.05,
            binding_sha256=digest(binding/'request.json'),control_sha256=digest(control/'request.json'),binary_sha256=digest(self.image)))
        expected=json.loads((root/'C1.json').read_text())['rows'][0]['greedy']['native_sha256']
        for pair in range(5):
            for variant in (('control','candidate') if pair%2==0 else ('candidate','control')):
                out=root/f'timed_p{pair}_{variant}';plan=control if variant=='control' else binding
                r=self.npu(key+f'_p{pair}_{variant}',[self.image,'bench',plan,out,'1024','1','3'],out,required,card,900)
                for repetition in (-1,0,1,2):
                    if digest(out/f'request_{repetition}/tokens.bin')!=expected:raise ValueError('Timed natural greedy identity differs')
                metrics=collect_samples(dict(output=str(out),batch=cell['batch']))
                state['samples'].append(dict(pair=pair,variant=variant,metrics=metrics,receipt=r['receipt'],job=r['job_directory']))
                self.save();self.summary()
        state.update(status='completed',performance=interval_summary(state['samples']));self.register(root,'result');self.cleanup()

    def summary(self):
        rows=[]
        for cell in self.protocol['cells']:
            s=self.state['cells'].get(cell['cell'],{});performance=interval_summary(s.get('samples',[]))
            row=dict(model=cell['model'],batch=cell['batch'],status=s.get('status','not_started'),reason=s.get('reason'),
                control_tpot_ms=None,candidate_tpot_ms=None,candidate_ttft_ms=None,candidate_e2e_ms=None,
                candidate_tokens_per_s=None,speedup=None,lower95=None,upper95=None,performance_gate=performance['status'])
            if 'interval' in performance:
                row.update(control_tpot_ms=performance['medians']['control']['tpot_ms'],
                    candidate_tpot_ms=performance['medians']['candidate']['tpot_ms'],
                    candidate_ttft_ms=performance['medians']['candidate']['ttft_ms'],
                    candidate_e2e_ms=performance['medians']['candidate']['e2e_ms'],
                    candidate_tokens_per_s=performance['medians']['candidate']['throughput_tokens_per_s'],
                    speedup=performance['interval']['geometric_mean'],lower95=performance['interval']['lower95'],upper95=performance['interval']['upper95'])
            rows.append(row)
        write(self.root/'summary.json',dict(status=self.state['status'],rows=rows,baseline_rerun=False,
            all_samples_retained=True,default_promotion=False))
        with (self.root/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def run(self):
        def stop(signum,frame):raise Deadline('Hard wall-clock/signal stop '+str(signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
        signal.setitimer(signal.ITIMER_REAL,max(1,self.hard_end-time.monotonic()))
        self.state['cells']={c['cell']:dict(status='not_started') for c in self.protocol['cells']};self.summary()
        try:
            # Snapshot immediately, then reuse one compiled catalogue for all cells.
            self.command('initial_observation',[sys.executable,'-m',MODULE,'observe','--out',self.root/'initial_observation.json'],timeout=15)
            self.disk();self.compile_catalogue()
            for cell in self.protocol['cells']:
                state=self.state['cells'][cell['cell']];state['status']='preparing';self.save()
                try:self.cell(cell)
                except Exception as e:
                    state.update(status='failed',reason=str(e),traceback=traceback.format_exc());self.save()
            self.state['status']='completed_with_failures' if any(s['status']!='completed' for s in self.state['cells'].values()) else 'completed'
        except Deadline as e:self.state.update(status='eight_hour_deadline',reason=str(e),device_exit_after_host_stop='not_guaranteed')
        except DeviceUnsafe as e:self.state.update(status='stopped_safety_or_resource',reason=str(e))
        except Exception:self.state.update(status='failed',reason=traceback.format_exc())
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            self.state['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();self.save();self.summary()
        return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','observe','disk'))
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path);p.add_argument('--hours',type=float,default=8)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.directory)
    elif a.action=='run':return Watch(a.directory,a.hours).run()
    elif a.action=='observe':
        from tilemega.bench.as1_run import snapshot
        write(controlled(a.out),snapshot(5))
    else:
        from contextlib import redirect_stdout
        from io import StringIO
        sys.path.insert(0,str(REPO/'scripts'));from as1_disk_status import main as disk
        stream=StringIO()
        with redirect_stdout(stream):disk('as2')
        controlled(a.out).write_text(stream.getvalue())
    return 0


if __name__=='__main__':raise SystemExit(main())
