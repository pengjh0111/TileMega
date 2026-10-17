#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Release the frozen final anchor only after all final acceptance reports."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,select,statistics,sys,time

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def acceptance_ready(native,baselines,paired,dnn):
    for report,count,key in [(native,30,'selected'),(baselines,120,'cells')]:
        rows=report.get(key,[])
        if report.get('complete')is not True or report.get('completed_cells')!=count or report.get('expected_cells')!=count or len(rows)!=count:return False
        if len({(r.get('model'),r.get('system'),r.get('batch'))for r in rows})!=count:return False
        if any(r.get('protocol_complete')is not True or r.get('fresh_processes')!=3 for r in rows):return False
    cells=paired.get('cells',[])
    if paired.get('complete_cells')!=9 or len(cells)!=9:return False
    if len({(c.get('system'),c.get('batch'))for c in cells})!=9:return False
    if any(c.get('complete')is not True or c.get('fresh_processes')!=3 for c in cells):return False
    cells=dnn.get('cells',[])
    return (dnn.get('complete')is True and dnn.get('completed_models')==5 and dnn.get('expected_models')==5
            and len(cells)==5 and len({c.get('model')for c in cells})==5
            and all(c.get('status')=='complete'and c.get('selected')for c in cells))

class Observer:
    def __init__(self,root):
        self.root=root.resolve();self.out=self.root/'runs/r16/anchors/final-v1'
        self.scheduler=self.root/'runs/r16/scheduler-rf2-final-v1/gpu3'
        self.queue=self.root/'runs/r16/queue-rf2-final-v1/gpu3/queue_final_end_anchor_v1.json'
        self.paths=[self.root/p for p in [
            'runs/r16/final-acceptance-v1/selected-final.json',
            'runs/r16/final-acceptance-v1/final-baselines.json',
            'runs/r16/moe-paired-middle-2ce-v1/gpu7-migration-v1/validated-results.json',
            'runs/r16/dnn/final-selection-v1/selection.json']]
    def release(self):
        if self.queue.exists():
            if read(self.queue)!=read(self.out/'end-template-not-published.json'):raise ValueError('immutable anchor queue changed')
            if not(self.out/'end-release-proof.json').exists():raise ValueError('anchor queue lacks release proof')
            return True
        snapshots=[p.read_bytes()for p in self.paths];reports=[json.loads(v)for v in snapshots]
        if not acceptance_ready(*reports):return False
        for row in reports[0]['selected']:
            if sha(row['plan'])!=row['plan_sha256']:return False
        dnn_plan=reports[3]['combination_plan']
        if sha(dnn_plan['path'])!=dnn_plan['sha256']:return False
        start=read(self.out/'start-results.json');plan=read(self.out/'plan.json')
        if start.get('complete')is not True:raise ValueError('start anchor incomplete')
        for name,digest in plan['files'].items():
            if sha(self.out/name)!=digest:raise ValueError('frozen official anchor file changed: '+name)
        jobs=read(self.out/'end-template-not-published.json')
        if len(jobs)!=6:raise ValueError('end anchor must have exactly six frozen jobs')
        sys.path.insert(0,str(self.root/'python'))
        from tilemega.bench.queue_check import check
        existing=[j for p in self.queue.parent.glob('queue_*.json')for j in read(p)]
        result=check(existing,jobs,read(self.scheduler/'state.json'));write(self.out/'end-queue-check.json',result)
        if not result['pass_check']:raise ValueError(result)
        if any(p.read_bytes()!=v for p,v in zip(self.paths,snapshots)):return False
        write(self.out/'end-release-proof.json',dict(at=time.time(),reports={str(p):hashlib.sha256(v).hexdigest()for p,v in zip(self.paths,snapshots)},
            template_sha256=sha(self.out/'end-template-not-published.json'),start_results_sha256=sha(self.out/'start-results.json'),jobs=6))
        end_plan=dict(plan,jobs=jobs,release_proof_sha256=sha(self.out/'end-release-proof.json'))
        write(self.out/'end-plan.json',end_plan);write(self.queue,jobs);return True
    def collect(self):
        sys.path.insert(0,str(self.root/'scripts'))
        from r16_measurement_validation import completed_guard,validate_llm
        from summarize_r16 import dnn_device_identity
        plan=read(self.out/'end-plan.json');state=read(self.scheduler/'state.json');accepted=[];pending=[];rejected=[]
        for job in plan['jobs']:
            record=state.get(job['name'],{})
            if record.get('status')not in ['done','failed','skipped','not_run','cancelled']:pending.append(job['name']);continue
            try:
                directory=Path(record['out']);guard=completed_guard(directory,record)
                paths=list((directory/'measurements').glob('B*/measurements.json'))
                if len(paths)!=1:raise ValueError('anchor measurement absent or ambiguous')
                value,samples,ownership=validate_llm(paths[0],guard,self.root);identity=dnn_device_identity(value,directory,paths[0])
                if identity['physical_gpu_uuid']!=plan['expected_uuid']or identity['hardware_boot_id']!=plan['expected_boot']:raise ValueError('anchor hardware epoch differs')
                if value['model_config_sha256']!=plan['config_sha256']or any(value['versions'].get(k)!=v for k,v in plan['versions'].items()):raise ValueError('anchor configuration or versions changed')
                accepted.append(dict(job=job['name'],batch=value['batch'],round=value['round_index'],pid=value['pid'],physical_uuid=identity['physical_gpu_uuid'],boot=identity['hardware_boot_id'],
                    **{k:value[k]for k in ['e2e_seconds','ttft_seconds','tpot_seconds','output_tokens_per_second']},acceptance=ownership['acceptance'],measurement=str(paths[0]),telemetry_boundaries=samples))
            except Exception as error:rejected.append(dict(job=job['name'],status=record.get('status'),error=str(error)))
        start=read(self.out/'start-results.json');cells=[]
        for batch in [1,16]:
            group=[r for r in accepted if r['batch']==batch]
            if len(group)!=3 or len({r['pid']for r in group})!=3 or {r['round']for r in group}!={0,1,2}:continue
            previous=next(c for c in start['cells']if c['batch']==batch)
            metrics={k:statistics.median(r[k]for r in group)for k in ['e2e_seconds','ttft_seconds','tpot_seconds','output_tokens_per_second']}
            drift={k:metrics[k]/previous[k]-1 for k in metrics}
            cells.append(dict(batch=batch,fresh_processes=3,physical_uuid=plan['expected_uuid'],boot=plan['expected_boot'],**metrics,relative_drift=drift,
                performance_drift_within_two_percent=all(abs(drift[k])<=.02 for k in ['e2e_seconds','tpot_seconds','output_tokens_per_second']),random_weights=True,acceptance='provisional_container_pid_namespace'))
        complete=len(cells)==2 and not pending and not rejected
        write(self.out/'end-results.json',dict(at=time.time(),complete=complete,cells=cells,accepted=accepted,pending=pending,rejected=rejected,
            performance_drift_within_two_percent=complete and all(c['performance_drift_within_two_percent']for c in cells)))
        return not pending

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();observer=Observer(args.root)
    lib=ctypes.CDLL(None);fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
    for directory in {p.parent for p in observer.paths}|{observer.scheduler}:assert lib.inotify_add_watch(fd,os.fsencode(directory),0x8|0x80)>=0
    while True:
        try:
            released=observer.release()
            write(observer.out/'end-observer-status.json',dict(at=time.time(),released=released))
            if released and observer.collect():break
        except Exception as error:write(observer.out/'end-observer-status.json',dict(at=time.time(),released=False,error=str(error)))
        if select.select([fd],[],[],55)[0]:
            try:
                while os.read(fd,1<<20):pass
            except BlockingIOError:pass
    os.close(fd)
if __name__=='__main__':main()
