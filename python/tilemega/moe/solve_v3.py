"""Build and measure four qualified MX-6 MoE combinations, with sealed receipts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tilemega.build.identity import verify
from tilemega.mx5_resources import inspect
from tilemega.moe.plan_occupancy import validate_inputs
from tilemega.moe.selection_v3 import stage_lists,k_best,pin,select_measured

ROOT=Path(__file__).resolve().parents[3]
LAYERS={'qwen3_30b_a3b_l24':24,'olmoe_1b_7b':16}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))


def save(path,value):
    with Path(path).open('x')as output:json.dump(value,output,indent=2,allow_nan=False);output.write('\n')


class Cell:
    def __init__(self,args):
        self.a=args;self.out=args.out.resolve();self.cell=args.cell_dir.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.stop=ROOT/'runs/mx6/v2/A_STOP.json';self.receipts=[]
        self.environment=dict(os.environ,PYTHONPATH=str(ROOT/'python'),NEEDS_MIB='40960',MX_MOE_HF_EXPERTS='eager',
            TILEMEGA_GPU_JOB_LOG=str(ROOT/'runs/mx6/gpu_jobs.log'))
        self.cpu_environment=dict(self.environment,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
        producer=self.out/'producer';producer.mkdir(exist_ok=True);digest=sha(args.compiler)
        self.compiler=producer/(digest+'_tilemega')
        if not self.compiler.exists():shutil.copy2(args.compiler,self.compiler)
        if sha(self.compiler)!=digest:raise ValueError('changed sealed producer')
        if not(producer/'PRODUCER.json').exists():save(producer/'PRODUCER.json',dict(binary=ref(self.compiler),original=ref(args.compiler)))

    def active(self):
        if self.stop.exists():raise RuntimeError('campaign A stop')

    def halt(self,reason,details):
        if not self.stop.exists():save(self.stop,dict(checkpoint='CP4',reason=reason,details=details))
        raise RuntimeError(reason)

    def job(self,out,cap,command):
        self.active();out.mkdir(parents=True);argv=[ROOT/'scripts/mx_gpu_job.sh',str(self.a.gpu),str(cap),out/'guard',*command];start=time.time()
        with(out/'stdout.log').open('x')as output:process=subprocess.run(list(map(str,argv)),cwd=ROOT,env=self.environment,stdout=output,stderr=subprocess.STDOUT)
        save(out/'PROCESS.json',dict(argv=list(map(str,argv)),card=self.a.gpu,start=start,end=time.time(),returncode=process.returncode))
        if process.returncode:raise RuntimeError('B job failure; retain output and diagnose: '+str(out))

    def compile(self,pin_path,directory,phase,watchdog=0):
        self.active();directory.mkdir(parents=True,exist_ok=True);binary=directory/'plan.so'
        if(directory/'COMPLETE.json').exists():
            verify(binary);return json.loads((directory/'COMPLETE.json').read_text())
        if binary.exists():raise ValueError('partial compile needs an explicit archived remedy directory')
        command=[self.compiler,'compile',self.a.bridge,binary,'--serving',phase,'--emit','serving','--frontend','decoder',
            '--batch',str(self.a.batch),'--prompt-len','64','--capacity','1088','--past-range','64:1086',
            '--runtime-target',self.a.target,'--evaluate-configs',pin_path,'--selection','predicted','--watchdog',str(watchdog)]
        if phase=='prefill':command+=['--mx-prompt','1']
        start=time.time()
        with(directory/'stdout.log').open('x')as output:process=subprocess.run(list(map(str,command)),cwd=ROOT,env=self.cpu_environment,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
        save(directory/'PROCESS.json',dict(argv=list(map(str,command)),start=start,end=time.time(),returncode=process.returncode))
        if process.returncode:return dict(status='compile_or_resource_rejected',directory=str(directory),process=ref(directory/'PROCESS.json'))
        identity=verify(binary);resources=inspect(binary,str(binary)+'.ptxas.log',512,kernel_name='mx_moe_kernel_v3')
        save(directory/'actual_resources.json',resources)
        if resources['violations']:raise ValueError('compiler accepted invalid resources')
        for name,command in [('MIR',[sys.executable,ROOT/'experiments/mx5/mir_check.py',str(binary)+'.ptxas.log','--out',directory/'MIR.json']),
            ('wait_report',[sys.executable,ROOT/'scripts/mx_mir_report.py',str(binary)+'.ptxas.log','--csv',directory/'wait_report.csv'])]:
            with(directory/(name+'.log')).open('x')as output:p=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            if p.returncode:return dict(status='MIR_rejected',directory=str(directory),log=ref(directory/(name+'.log')))
        spec=dict(kind='mx5_moe_plan_occupancy_inputs',schema_version=1,program_version=3,batch=self.a.batch,threads=512,
            context_kind='native_identity',binary=ref(binary),generated_source=ref(str(binary)+'.cu'),resources=ref(directory/'actual_resources.json'),
            log=ref(str(binary)+'.ptxas.log'),context=ref(str(binary)+'.identity.json'),source_snapshot=ref(str(binary)+'.source.json'),manifest=ref(str(binary)+'.plan.json'))
        if self.a.model_id=='olmoe_1b_7b':spec['model_geometry']=dict(layers=16,hidden=2048,intermediate=1024,experts=64,hkv=16,q=1)
        validate_inputs(spec);save(directory/'query_inputs.json',spec)
        result=dict(status='CPU_resource_fit',binary=ref(binary),identity=identity['artifact_id'],resources=resources['kernel'])
        save(directory/'COMPLETE.json',result);return result

    def query(self,folder,label):
        out=folder/label;out.mkdir(parents=True);commands=[]
        for i,phase in enumerate(('decode','prefill','decode_w')):
            commands.append([sys.executable,'-m','tilemega.moe.plan_occupancy','--inputs',str(folder/phase/'query_inputs.json'),
                '--out',str(out/f'query{i}.json'),'--query-only','--iterations','0'])
        driver=out/'query_cpu.py';driver.write_text('import subprocess\nfor a in '+repr(commands)+':subprocess.run(a,check=True)\n')
        self.job(out/'execution',300,[sys.executable,driver])
        queries=[json.loads((out/f'query{i}.json').read_text())for i in range(3)]
        return all(q['query_status']==0 and q['resident_ctas']==1 and q['attributes']['localSizeBytes']==q['attributes']['sharedSizeBytes']==0 for q in queries)

    def correctness(self,folder,label):
        out=folder/label
        try:self.job(out,1800,[ROOT/'scripts/mx_check_modes.sh',self.a.model,folder/'prefill/plan.so',folder/'decode_w/plan.so',
            str(self.a.batch),'512',out/'check',self.a.prompt_ids])
        except RuntimeError:
            for path in(out/'check/c1/hf_check.json',out/'check/c2/mode_check.json'):
                if path.exists()and json.loads(path.read_text()).get('pass')is False:self.halt('13.4.A2 C1/C2 failure',ref(path))
            raise
        c1=out/'check/c1/hf_check.json';c2=out/'check/c2/mode_check.json'
        if not json.loads(c1.read_text())['pass']or not json.loads(c2.read_text())['pass']:self.halt('13.4.A2 C1/C2 failure',dict(C1=ref(c1),C2=ref(c2)))
        return dict(C1=ref(c1),C2=ref(c2))

    def measure(self,folder,label):
        values=[];records=[]
        for number in(1,2):
            out=folder/f'{label}_round{number}'
            self.job(out,900,[sys.executable,'-m','tilemega.serving.measure','--model',self.a.model,
                '--prefill-so',folder/'prefill/plan.so','--decode-so',folder/'decode/plan.so','--prompt-ids',self.a.prompt_ids,
                '--batch',str(self.a.batch),'--mode','L1','--prefill-mode','L1','--decode-loop','1',
                '--max-new-tokens','1024','--warmup','1','--repeats','2','--step-events','0',
                '--policy',ROOT/'configs/bench/measurement_policy_c550.json','--out',out/'measure'])
            path=out/'measure/measurements.json';result=json.loads(path.read_text())
            if not result['timed_tokens_identical']:self.halt('13.4.A2 repeated greedy generation differs',ref(path))
            values.append(result['tpot_seconds']*1e9);records.append(ref(path))
        spread=(max(values)-min(values))/(sum(values)/2)
        if spread>.02:
            save(folder/(label+'_B_SPREAD.json'),dict(values=values,spread=spread,records=records))
            raise RuntimeError('B3 measurement spread above2percent; preserve and remedy once')
        return dict(fresh_process_ns=values,measurements=records)

    def build(self,pin_path,folder):
        attempts=[]
        for phase,watchdog in(('decode',0),('prefill',0),('decode',1)):
            label=phase+('_w'if watchdog else'');result=self.compile(pin_path,folder/label,phase,watchdog);attempts.append(result)
            if result['status']!='CPU_resource_fit' and phase=='prefill':
                original=json.loads(pin_path.read_text())
                if original['mx_moe_v3']['attn_version']==3:
                    rejected=folder/'prefill_attention3_rejected'
                    (folder/'prefill').rename(rejected)
                    control=ROOT/f'configs/pins/mx6/cp1/{self.a.model_id}_B{self.a.batch}_manual.json'
                    fallback=original['mx_moe_v3'];fallback.update(attn_version=2,split=1,
                        b_min=json.loads(control.read_text())['mx_moe_v3']['b_min'])
                    original['prefill_resource_fallback']=dict(reason='approved attention2 fallback after prefill resource rejection; GEMM coordinates unchanged',
                        rejected_directory=str(rejected),same_batch_control=ref(control),decode_pin=ref(pin_path))
                    if fallback['purpose']=='selected':fallback['selection']['prefill_resource_fallback']=original['prefill_resource_fallback']
                    fallback_pin=folder/'prefill_attention2_pin.json';save(fallback_pin,original)
                    result=self.compile(fallback_pin,folder/'prefill','prefill',watchdog);attempts.append(result)
            if result['status']!='CPU_resource_fit':return False,attempts
        return self.query(folder,'queries'),attempts

    def run(self):
        catalog=json.loads(self.a.costs.read_text())
        if catalog.get('complete_scope')is not True:raise ValueError('complete source-bound measured costs required')
        cost_hash=sha(self.a.costs);lists=stage_lists(catalog['rows'],self.a.model_id,self.a.batch);qualified=[];rejected=[]
        for choice in k_best(lists,LAYERS[self.a.model_id]):
            folder=self.out/f"rank{choice['candidate_rank']:07d}";folder.mkdir(exist_ok=True)
            choice_path=folder/'CHOICE.json'
            if not choice_path.exists():save(choice_path,choice)
            pin_path=folder/'resource_pin.json'
            if not pin_path.exists():save(pin_path,pin(choice,cost_hash))
            fit,attempts=self.build(pin_path,folder)
            result=dict(choice=choice,pin=ref(pin_path),folder=str(folder),attempts=attempts,status='qualified'if fit else'resource_rejected')
            save(folder/'RESOURCE_RESULT.json',result)
            if fit:qualified.append(result)
            else:rejected.append(result)
            if len(qualified)==4:break
        if len(qualified)!=4:raise RuntimeError('B2 no four resource-qualified combinations; inspect closure')
        save(self.out/'QUALIFIED.json',dict(qualified=qualified,rejected=rejected,cost_table=ref(self.a.costs)))
        predictions=[];measured=[]
        for candidate in qualified:
            folder=Path(candidate['folder']);candidate['correctness']=self.correctness(folder,'correctness')
            candidate['timing']=self.measure(folder,'timing');predictions.append(candidate['choice']['predicted_ns']);measured.append(candidate['timing']['fresh_process_ns'])
            save(folder/'CANDIDATE_RESULT.json',candidate)
        selection=select_measured(predictions,measured);selected=qualified[selection['selected']]
        save(self.out/'MEASURED_SELECTION.json',dict(selection=selection,qualified=qualified,cost_table=ref(self.a.costs)))
        final_pin=pin(selected['choice'],cost_hash,'selected');final_pin['mx_moe_v3']['selection'].update(
            measurement_decision=ref(self.out/'MEASURED_SELECTION.json'),eligible_rank=selection['selected']+1,
            fresh_processes_per_candidate=2,correctness_steps=512,default_enabled=False)
        pin_path=self.out/'selected_pin.json';save(pin_path,final_pin)
        fit,attempts=self.build(pin_path,self.cell)
        if not fit:raise RuntimeError('B2 selected-purpose rebuild rejected; retain candidate identities and inspect')
        correctness=self.correctness(self.cell,'selected_correctness');timing=self.measure(self.cell,'selected_timing')
        result=dict(model=self.a.model_id,batch=self.a.batch,selection=selection,cost_table=ref(self.a.costs),
            selected_pin=ref(pin_path),choice=selected['choice'],selected_correctness=correctness,selected_timing=timing,
            final_builds=attempts,qualified=ref(self.out/'QUALIFIED.json'),decision=ref(self.out/'MEASURED_SELECTION.json'),
            solver_selected=True,default_enabled=False,G6_acceptance=False)
        save(self.out/'results.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id',choices=tuple(LAYERS),required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True)
    parser.add_argument('--gpu',type=int,choices=(7,6,5,4),required=True)
    for name in('bridge','costs','prompt-ids','out','cell-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--compiler',type=Path,default=ROOT/'build/tools/tilemega')
    parser.add_argument('--target',type=Path,default=ROOT/'configs/targets/xcore1000.json')
    args=parser.parse_args();args.cell_dir.mkdir(parents=True,exist_ok=True)
    with(args.cell_dir/'.mx6_selection.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(args.out/'results.json').exists():
            result=json.loads((args.out/'results.json').read_text())
            if ref(args.costs)!=result['cost_table']:raise ValueError('completed selection cost table changed')
            for phase in('decode','prefill','decode_w'):verify(args.cell_dir/phase/'plan.so')
            print('MX6_MEASURED_SELECTION_ALREADY_COMPLETE',args.model_id,args.batch);return
        cell=Cell(args)
        try:cell.run()
        except BaseException as error:
            marker=cell.out/f'B_FAILURE_{time.time_ns()}.json'
            save(marker,dict(error=repr(error),A_stop=cell.stop.exists(),source='retained incomplete selection; no acceptance'))
            raise
        print('MX6_MEASURED_SELECTION_COMPLETE',args.model_id,args.batch,flush=True)


if __name__=='__main__':main()
