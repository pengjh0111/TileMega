#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run one bounded M0 trial for the current physical GPU4/7 pair."""
from pathlib import Path
import argparse,copy,ctypes,hashlib,json,os,select,shutil,sys,time

TERMINAL={'done','failed','skipped','not_run','cancelled'}
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def watch(directories):
    lib=ctypes.CDLL(None);fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
    for directory in directories:assert lib.inotify_add_watch(fd,os.fsencode(directory),0x8|0x80)>=0
    return fd
def wait(fd,seconds=55):
    if select.select([fd],[],[],seconds)[0]:
        try:
            while os.read(fd,1<<20):pass
        except BlockingIOError:pass

def validate_barriers(rows):
    """A completed peer process cannot certify a different round or nonce."""
    values={(r['gpu'],r['round']):read(r['absolute_measurement'])for r in rows if r['phase']=='concurrent'}
    for (gpu,rnd),value in values.items():
        peer=values[(7 if gpu==4 else 4,rnd)]
        for run in [1,2,3]:
            record=next(r for r in value['generation_runs']if r['N']==1024 and r['run']==run)
            other=next(r for r in peer['generation_runs']if r['N']==1024 and r['run']==run)
            a,b=record['m0_barrier'],other['m0_barrier']
            if (a['participants']!=['4','7']or a['round']!=rnd or a['run']!=run
                    or a['peer_pid']!=peer['pid']or b['peer_pid']!=value['pid']
                    or a['nonce']!=b['peer_nonce']or b['nonce']!=a['peer_nonce']):raise ValueError('M0 peer/nonce/round mismatch')
            directory=Path(a['directory']);own=read(directory/f'gpu{gpu}.json');remote=read(directory/f"gpu{7 if gpu==4 else 4}.json")
            if (own['pid']!=value['pid']or own['nonce']!=a['nonce']or remote['process_start']!=a['peer_process_start']
                    or own['ready_against'].get(remote['gpu'])!=remote['nonce']):raise ValueError('M0 archived live-process handshake differs')

class Trial:
    def __init__(self,root,out):
        self.root=root.resolve();self.out=out.resolve();self.anchor=self.root/'runs/r16/anchors/final-v1'
        self.states={4:self.root/'runs/r16/scheduler-rf2-final-v1/gpu4',7:self.root/'runs/r16/scheduler/gpu7'}
        self.queues={4:self.root/'runs/r16/queue-rf2-final-v1/gpu4/queue_m0_current47_v1.json',7:self.root/'runs/r16/queue/gpu7/queue_m0_current47_v1.json'}
    def prepare(self):
        if(self.out/'plan.json').exists():raise ValueError('one trial only; refuse overwrite or repeat')
        self.out.mkdir(parents=True,exist_ok=True);driver=self.out/'driver';driver.mkdir();anchor=read(self.anchor/'plan.json')
        for name,digest in anchor['files'].items():
            if sha(self.anchor/name)!=digest:raise ValueError('frozen official anchor changed')
            shutil.copy2(self.anchor/name,driver/name)
        shutil.copy2(Path(__file__),self.out/'controller_frozen.py');validator=self.out/'validator';validator.mkdir()
        for name in ['analyze_r16_m0.py','summarize_r16.py','r16_measurement_validation.py','r16_runtime_contract.py','r16_argument_contract.py','r16_kernel_resources.py']:
            shutil.copy2(self.root/'scripts'/name,validator/name)
        cards={'4':dict(uuid='1f2fad96-e938-be19-e775-94e8ae7501a4',pci='0000:86:00.0',numa=1),
               '7':dict(uuid='ac78f16f-955d-c08b-e6fe-4e4554c8954d',pci='0000:B7:00.0',numa=1)}
        templates=[j for j in read(self.anchor/'end-template-not-published.json')if '--batch'in j['command']and j['command'][j['command'].index('--batch')+1]=='16']
        if len(templates)!=3:raise ValueError('need original three B16 rounds')
        alljobs=[];gpujobs=[]
        for gpu in [4,7]:
            jobs=[];solo=[];gate=f'm0_current47_v1_gpu{gpu}_solo_phase_gate'
            for phase in ['solo','concurrent']:
                previous=None
                for rnd,template in enumerate(templates):
                    job=copy.deepcopy(template);name=f'm0_{phase}_current47_v1_llama3_2_1b_vllm_B16_r{rnd}_gpu{gpu}';job.update(name=name,priority=1,timeout_s=1200,after=([previous]if previous else[])+([gate]if phase=='concurrent'else[]))
                    env=job['env'];env.update(TILEMEGA_DEVICE_INDEX=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),TILEMEGA_NUMA_NODE='1',TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock')
                    for key in ['TILEMEGA_NUMA_CONCURRENCY_VERIFIED','TILEMEGA_NUMA_CONCURRENCY_TRIAL','TILEMEGA_M0_BARRIER_DIR','TILEMEGA_M0_PARTICIPANTS','TILEMEGA_M0_BARRIER_TIMEOUT_S']:env.pop(key,None)
                    if phase=='concurrent':env.update(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',TILEMEGA_M0_BARRIER_DIR=str(self.out/'barriers'),TILEMEGA_M0_PARTICIPANTS='4,7',TILEMEGA_M0_BARRIER_TIMEOUT_S='300')
                    cmd=job['command'];cmd[cmd.index('--cpunodebind=0')]='--cpunodebind=1'
                    for index,value in enumerate(cmd):
                        if value.endswith('/run_r16_with_device_identity.py')or value.endswith('/vllm_baseline.py'):cmd[index]=str(driver/Path(value).name)
                    for flag,value in [('--expected-uuid',cards[str(gpu)]['uuid']),('--expected-pci',cards[str(gpu)]['pci']),('--numa-node','1'),('--prompt-ids',str(driver/'llama3_isl64.json'))]:cmd[cmd.index(flag)+1]=value
                    jobs.append(job);gpujobs.append(dict(name=name,gpu=gpu,phase=phase,round=rnd));previous=name
                    if phase=='solo':solo.append(name)
                if phase=='solo':
                    jobs.append(dict(name=gate,gpu=False,timing=False,priority=1,after=solo,timeout_s=1900,cwd=str(self.root),env=copy.deepcopy(jobs[0]['env']),
                        command=['python3',str(self.out/'controller_frozen.py'),'--root',str(self.root),'--out',str(self.out),'--phase-gate','--gpu',str(gpu),'--receipt','{attempt_dir}/phase-gate.json']))
            write(self.out/f'proposed-gpu{gpu}.json',jobs);alljobs.extend(jobs)
        plan=dict(created=time.time(),run_id='current47_v1',deadline_seconds=3600,threshold=.01,cards=cards,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            model_config_sha256=anchor['config_sha256'],prompt_ids_sha256=anchor['files']['llama3_isl64.json'],versions=anchor['versions'],controller_sha256=sha(self.out/'controller_frozen.py'),jobs=gpujobs,phase_gate_names=[j['name']for j in alljobs if not j['gpu']],
            frozen_files={str(p.relative_to(self.out)):sha(p)for d in [driver,validator]for p in d.glob('*.py')},policy_changed=False,concurrency_scope='current physical pair4/7 only; root review required even after PASS')
        write(self.out/'plan.json',plan);sys.path.insert(0,str(self.root/'python'))
        from tilemega.bench.queue_check import check
        for gpu in [4,7]:
            q=self.queues[gpu];jobs=read(self.out/f'proposed-gpu{gpu}.json');existing=[j for p in q.parent.glob('queue_*.json')for j in read(p)];result=check(existing,jobs,read(self.states[gpu]/'state.json'));write(self.out/f'queue-check-gpu{gpu}.json',result)
            if not result['pass_check']:raise ValueError(result)
        for gpu in [4,7]:write(self.queues[gpu],read(self.out/f'proposed-gpu{gpu}.json'))
    def phase_gate(self,gpu,receipt):
        fd=watch([self.out]);deadline=time.monotonic()+1800
        try:
            while time.monotonic()<deadline:
                path=self.out/'phase.json'
                if path.exists():
                    value=read(path);passed=value.get('phase')=='concurrent'and value.get('solo_accepted')==6 and value.get('plan_sha256')==sha(self.out/'plan.json')
                    write(receipt,dict(at=time.time(),gpu=gpu,passed=passed,phase_sha256=sha(path)));return 0 if passed else 2
                wait(fd,min(55,max(0,deadline-time.monotonic())))
            write(receipt,dict(at=time.time(),gpu=gpu,passed=False,reason='bounded solo phase timeout'));return 2
        finally:os.close(fd)
    def collect(self,plan):
        sys.path.insert(0,str(self.out/'validator'))
        from r16_measurement_validation import completed_guard,validate_llm
        from summarize_r16 import dnn_device_identity
        states={g:read(d/'state.json')for g,d in self.states.items()};rows=[];rejected=[];pending=[]
        for item in plan['jobs']:
            record=states[item['gpu']].get(item['name'],{})
            if record.get('status')not in TERMINAL:pending.append(item);continue
            try:
                directory=Path(record['out']);guard=completed_guard(directory,record);paths=list((directory/'measurements').glob('B16/measurements.json'))
                if len(paths)!=1:raise ValueError('one B16 measurement required')
                value,samples,ownership=validate_llm(paths[0],guard,self.root);identity=dnn_device_identity(value,directory,paths[0]);expected=plan['cards'][str(item['gpu'])]
                if identity['physical_gpu_uuid']!=expected['uuid']or identity['hardware_boot_id']!=plan['boot']:raise ValueError('trial physical UUID/boot changed')
                if value['model_config_sha256']!=plan['model_config_sha256']or any(value['versions'].get(k)!=v for k,v in plan['versions'].items()):raise ValueError('official model/version changed')
                physical=value['physical_device_identity']
                if physical['cuda_pci_bus_id'].lower()!=expected['pci'].lower()or physical['numa_node']!=1:raise ValueError('trial PCI/NUMA binding differs')
                if value['prompt_ids_sha256']!=plan['prompt_ids_sha256']or value['round_index']!=item['round']or value['gpu_index']!=item['gpu']:raise ValueError('trial prompt/round/device differs')
                if value['load_format']!='dummy'or value['configuration']['seed']!=0:raise ValueError('official initialization changed')
                rows.append(dict(job=item['name'],gpu=item['gpu'],phase=item['phase'],round=item['round'],model='llama3_2_1b',pid=value['pid'],e2e_ms=value['e2e_seconds']*1000,tpot_ms=value['tpot_seconds']*1000,
                    ownership_verified=ownership['ownership_verified'],measurement=str(paths[0].relative_to(self.root)),absolute_measurement=str(paths[0]),measurement_sha256=sha(paths[0]),physical_identity=identity))
            except Exception as error:rejected.append(dict(job=item['name'],reason=str(error)))
        return states,rows,rejected,pending
    def abort(self,reason,states):
        write(self.out/'phase.json',dict(phase='abort',reason=reason,at=time.time(),plan_sha256=sha(self.out/'plan.json')))
        for gpu,path in self.queues.items():
            jobs=read(path);changed=False
            for job in jobs:
                if states[gpu].get(job['name'],{}).get('status','pending')=='pending':
                    hold='__m0_current47_v1_aborted_no_repeat__'
                    if hold not in job['after']:job['after'].append(hold);changed=True
            if changed:
                original=self.out/f'abort-original-gpu{gpu}.json'
                if not original.exists():shutil.copy2(path,original)
                write(path,jobs)
    def advance(self):
        plan=read(self.out/'plan.json')
        if sha(self.out/'controller_frozen.py')!=plan['controller_sha256']:raise ValueError('trial controller changed')
        for name,digest in plan['frozen_files'].items():
            if sha(self.out/name)!=digest:raise ValueError('frozen official/helper file changed: '+name)
        states,rows,rejected,pending=self.collect(plan);phase=read(self.out/'phase.json')if(self.out/'phase.json').exists()else{}
        gate_failed=any(states[g].get(f'm0_current47_v1_gpu{g}_solo_phase_gate',{}).get('status')in TERMINAL-{'done'}for g in [4,7])
        if phase.get('phase')!='abort'and(rejected or gate_failed or time.time()>plan['created']+plan['deadline_seconds']):
            self.abort('terminal invalid trial receipt/gate or bounded deadline; no repeat',states);phase=read(self.out/'phase.json')
        solo=[r for r in rows if r['phase']=='solo']
        if len(solo)==6 and not phase:
            write(self.out/'phase.json',dict(phase='concurrent',at=time.time(),solo_accepted=6,plan_sha256=sha(self.out/'plan.json'),measurements={r['absolute_measurement']:r['measurement_sha256']for r in solo}))
        running=[j for g,state in states.items()for j,r in state.items()if r.get('status')=='running'and j in {x['name']for x in plan['jobs']}|set(plan['phase_gate_names'])]
        if not pending or phase.get('phase')=='abort'and not running:
            from analyze_r16_m0 import analyze
            result=analyze(self.root,plan['run_id'],rows=rows,rejected=rejected)
            try:
                if len(rows)!=12 or len({r['pid']for r in rows})!=12:raise ValueError('need twelve distinct fresh runner processes')
                validate_barriers(rows)
            except Exception as error:result['errors'].append(str(error));result.update(passed=False,status='failed')
            for interval in result['interval_coverage']:interval['conservative_overlap_fraction']=interval['guaranteed_overlap_seconds']/interval['request_duration_seconds']
            result.update(can_enable_numa_concurrency=False,root_review_required=True,policy_changed=False,current_boot=plan['boot'],physical_cards=plan['cards'],phase_status=phase,
                proof_can_enable_current_pair_after_root_review=result['passed'],hardware_proofs=[dict(job=r['job'],identity=r['physical_identity'],measurement_sha256=r['measurement_sha256'])for r in rows])
            write(self.out/'result.json',result);return True
        write(self.out/'progress.json',dict(at=time.time(),accepted=len(rows),solo_accepted=len(solo),pending=len(pending),rejected=rejected,phase=phase.get('phase','solo'),root_review_required=True));return False

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--phase-gate',action='store_true');parser.add_argument('--gpu',type=int);parser.add_argument('--receipt',type=Path);args=parser.parse_args();trial=Trial(args.root,args.out)
    if args.prepare:trial.prepare();return
    if args.phase_gate:raise SystemExit(trial.phase_gate(args.gpu,args.receipt))
    fd=watch(trial.states.values())
    try:
        while True:
            try:
                if trial.advance():break
            except Exception as error:write(trial.out/'observer-error.json',dict(at=time.time(),error=str(error)))
            wait(fd)
    finally:os.close(fd)
if __name__=='__main__':main()
