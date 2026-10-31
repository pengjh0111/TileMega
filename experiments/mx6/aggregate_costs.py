#!/usr/bin/env python3
"""Aggregate three source-bound fresh processes; retain every rejected key."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked(ref):
    path=Path(ref['path'])
    if sha(path)!=ref['sha256']:raise ValueError('changed cost evidence: '+str(path))
    return path


def aggregate(request):
    source=json.loads(checked(request['source']).read_text());rows=[];rejected=[];seen=set();pids=set()
    expected=json.loads(checked(request['all_keys']).read_text())
    expected_keys={json.dumps(row['key'],sort_keys=True)for row in expected}
    if len(expected_keys)!=len(expected):raise ValueError('duplicate requested key')
    for job in request['jobs']:
        keys=json.loads(checked(job['keys']).read_text());samples=[];numbers=set();job_pids=set()
        if len(job['records'])!=3:raise ValueError('three fresh process records required')
        for record in job['records']:
            number=record['round'];raw=checked(record['raw']);process=json.loads(checked(record['process']).read_text());guard=json.loads(checked(record['guard']).read_text())
            if number not in(1,2,3)or number in numbers:raise ValueError('duplicate/missing round')
            if process['returncode']or guard['code']:raise ValueError('failed guard process')
            argv=process['argv']
            expected_tail=[request['driver']['path'],'--keys',job['keys']['path'],'--out',str(raw),'--round',str(number)]
            if argv[5:]!=expected_tail or Path(argv[0]).name!='mx_gpu_job.sh' or argv[1]!=str(process['card']) or argv[2]!='900' or Path(argv[3])!=Path(record['guard']['path']).parent:
                raise ValueError('actual GPU command differs from cost scope')
            samples_now=[json.loads(line)for line in raw.read_text().splitlines()]
            if len(samples_now)!=len(keys):raise ValueError('missing/extra cost key')
            pid_set={row['PID']for row in samples_now}
            if len(pid_set)!=1:raise ValueError('one process must produce a cost batch')
            pid=pid_set.pop()
            if pid in pids or pid in job_pids:raise ValueError('cost process reused')
            if int((Path(record['guard']['path']).parent/'pgid').read_text())!=pid:raise ValueError('cost PID differs from guard child')
            for row,key in zip(samples_now,keys):
                if row['key']!=key['key']or row['binary']!=key['binary']or row['build']!=key['build']or row['round']!=number:raise ValueError('cost sample/build/key mismatch')
                if row['driver_sha256']!=request['driver']['sha256']or row['key_file_sha256']!=job['keys']['sha256']:raise ValueError('cost driver/key input differs')
                if row['physical_card']!=process['card']or row['guard']['physical_gpu']!=str(process['card']):raise ValueError('physical GPU differs')
                if row['contract']!='bf16'or row['timing_eligible']or row['production_acceptance']or row['input_seed']!=63102:raise ValueError('incorrect cost scope')
                if row['source_digest']!=source['source_digest']:raise ValueError('cost source epoch differs')
            samples.append((number,samples_now,record));numbers.add(number);job_pids.add(pid)
        pids.update(job_pids)
        for index,key in enumerate(keys):
            identity=json.dumps(key['key'],sort_keys=True)
            if identity in seen:raise ValueError('key measured twice')
            seen.add(identity);values=[batch[index]for _,batch,_ in samples]
            statuses={row['status']for row in values}
            if statuses!={'pass'}:
                rejected.append(dict(key=key['key'],status='actual_device_resource_rejection',samples=values,source_build=key));continue
            resources=values[0]['resources']
            if any(row['resources']!=resources for row in values)or resources['resident']!=1 or resources['stack']or resources['static_smem']or not 0<resources['MTreg']<=255:raise ValueError('actual resources differ or exceed limits')
            if not all(row['repeat_bitwise']for row in values):raise ValueError('component determinism failed')
            if any(row.get('addressed_rotating_weight_bytes',row.get('addressed_rotating_KV_bytes',0))<256*1024**2 for row in values):raise ValueError('addressed cold rotation below256MiB')
            costs=[row['cost_ns']for row in values]
            if any(not isinstance(v,(float,int))or v<=0 for v in costs):raise ValueError('invalid cost')
            median=statistics.median(costs);spread=(max(costs)-min(costs))/median
            row=dict(key=key['key'],median_ns=median,p90_ns=statistics.median(v['p90_ns']for v in values),spread=spread,processes=3,
                resources=resources,physical_card=values[0]['physical_card'],source_digest=source['source_digest'],
                binary=key['binary'],build=key['build'],rounds=[dict(round=number,input=dict(**record['raw'],line=index+1),
                    process=record['process'],guard=record['guard'],PID=batch[index]['PID'],cost_ns=batch[index]['cost_ns'],
                    p90_ns=batch[index]['p90_ns'],spread_ns=batch[index]['spread_ns'])for number,batch,record in sorted(samples)],
                statistic=values[0]['cost_statistic'],evidence='verified',timing_eligible=False,default_enabled=False)
            if key['key'].get('attn_version',3)==2:
                proof=json.loads(checked(key['fallback_evidence']).read_text())
                if not proof['fallback_costs_approved']or proof['attn_version']!=2 or proof['model']!=key['key']['model']or proof['batch']!=key['key']['b']:
                    raise ValueError('attention2 lacks its measured approved fallback decision')
                row.update(approved_fallback=True,fallback_evidence=key['fallback_evidence'])
            rows.append(row)
    if seen!=expected_keys:raise ValueError('incomplete cost scope')
    checked(request['driver'])
    return dict(schema_version=1,kind='mx6_moe_measured_stage_costs',target='xcore1000',HEAD=source['head'],
        date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source=request['source'],driver=request['driver'],
        source_digest=source['source_digest'],domain=request['domain'],requested_keys=len(expected),
        rows=rows,resource_rejections=rejected,CPU_resource_rejections=request['CPU_resource_rejections'],
        complete_scope=True,default_enabled=False,final_selection=False,fresh_processes=len(pids),
        physical_cards=sorted({row['physical_card']for row in rows},reverse=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('preserve existing cost catalog')
    result=aggregate(json.loads(args.request.read_text()))
    with args.out.open('x')as output:json.dump(result,output,indent=2,allow_nan=False);output.write('\n')
    print('MX6_COST_CATALOG',len(result['rows']),len(result['resource_rejections']),result['fresh_processes'],flush=True)


if __name__=='__main__':main()
