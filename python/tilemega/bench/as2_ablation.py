"""Audit a predeclared, finite paired TileMega optimization experiment."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

from tilemega.bench.as2_queue import controlled, audit_tilemega_tokens, collect_samples


def paired_interval(ratios):
    # Fixed before final sampling; requests within one process are not replicates.
    critical={5:2.7764451051977987,10:2.2621571628540993}
    if len(ratios) not in critical or any(not math.isfinite(v) or v<=0 for v in ratios):
        raise ValueError('Require exactly five or ten positive process-pair ratios')
    logs=[math.log(v) for v in ratios];mean=statistics.mean(logs)
    radius=critical[len(logs)]*statistics.stdev(logs)/math.sqrt(len(logs))
    return dict(geometric_mean=math.exp(mean),lower95=math.exp(mean-radius),
                upper95=math.exp(mean+radius),log_ratio_sample_sd=statistics.stdev(logs))


def summarize(directory,out):
    directory,out=controlled(directory),controlled(out)
    if out.exists():raise FileExistsError(out)
    protocol_path=directory/'protocol.json';protocol=json.loads(protocol_path.read_text())
    state=json.loads((directory/'state.json').read_text())
    if (protocol['schema']!='tilemega.as2.paired_ablation.v1' or protocol['pairs'] not in (5,10) or
            protocol['primary_metric']!='tpot_ms' or protocol['warmup']!=1 or protocol['repetitions']!=3 or
            protocol['isl']!=64 or protocol['osl']!=1024 or protocol['confidence_method']!='paired_log_t_95' or
            not 0<protocol['max_process_cv']<=.05 or
            hashlib.sha256(protocol_path.read_bytes()).hexdigest()!=state['protocol_sha256']):
        raise ValueError('Unfrozen or unsupported paired measurement protocol')
    expected=protocol['ordered_job_ids'];jobs=state['jobs']
    if len(expected)!=2*protocol['pairs'] or len(set(expected))!=len(expected):
        raise ValueError('Invalid predeclared process count')
    missing=[name for name in expected if jobs.get(name,{}).get('status')!='collected']
    result=dict(schema='tilemega.as2.paired_ablation_result.v1',protocol=str(protocol_path),
        protocol_sha256=state['protocol_sha256'],status='inconclusive',missing=missing,
        original_as2_G_PERF='not_evaluated: optimization versus TileMega control, not eager',
        method='95% t interval on independent process-pair log ratios; approximate normality/independence',
        no_slow_samples_dropped=True)
    if missing:
        out.write_text(json.dumps(result,indent=2)+'\n');return result
    identities=set();rows=[];pair_rows={};reference_bindings={};image=None;previous_end=None
    for name in expected:
        item=jobs[name];job=item['request'];receipt=item['receipt'];kind=job['optimization_variant']
        if kind not in ('original','fifo') or job['pair_id']>=protocol['pairs']:
            raise ValueError('Unexpected paired variant')
        identity=(receipt['pid'],receipt['start_ticks']);identities.add(identity)
        if (receipt['physical_device_id']!=protocol['physical_device_id'] or
                dt.datetime.fromisoformat(protocol['frozen_at'])>=dt.datetime.fromisoformat(receipt['started_at']) or
                previous_end is not None and previous_end>receipt['started_at']):
            raise ValueError('Protocol frozen late, mixed device, or overlapping submissions')
        previous_end=receipt['finished_at']
        if image is None:image=receipt['executable_sha256']
        if image!=receipt['executable_sha256']:raise ValueError('Ablation changed the device binary')
        binding=controlled(job['argv'][2]);identity=json.loads((binding/'request.json').read_text())
        if (str(binding)!=protocol['bindings'][kind] or image!=protocol['binary_sha256']):
            raise ValueError('Candidate binding or image changed after protocol freeze')
        key=tuple(identity[k] for k in ('model_id','batch','groups','config_sha256',
            'weight_manifest_sha256','input_sha256','arena_bytes'))
        reference_bindings[kind]=(binding,key)
        check=audit_tilemega_tokens(job)
        if not check['passed']:raise ValueError('Timed greedy output differs from its numerical proof')
        measured=collect_samples(job)
        if measured!=item['metrics']:raise ValueError('Raw timing samples changed after collection')
        row=dict(id=name,pair_id=job['pair_id'],variant=kind,job=item['job_directory'],
            physical_device_id=receipt['physical_device_id'],timing_status=receipt['timing_status'],
            pid=receipt['pid'],start_ticks=receipt['start_ticks'],
            metrics=item['metrics'],token_audit=check)
        if kind in pair_rows.setdefault(job['pair_id'],{}):raise ValueError('Duplicate paired variant')
        pair_rows[job['pair_id']][kind]=row;rows.append(row)
    if len(identities)!=2*protocol['pairs'] or set(pair_rows)!=set(range(protocol['pairs'])):
        raise ValueError('Require fresh process identities and complete pairs')
    original,original_key=reference_bindings['original'];fifo,fifo_key=reference_bindings['fifo']
    if original_key!=fifo_key:raise ValueError('Model, weights, input or resource identity changed')
    for phase in ('prefill','decode'):
        for name in ('stages.bin','queue.bin','offsets.bin','allocations.bin'):
            if (original/phase/name).read_bytes()!=(fifo/phase/name).read_bytes():
                raise ValueError('Wait-only ablation changed Body, placement, order or memory')
        a=list(struct.iter_unpack('<7I',(original/phase/'tasks.bin').read_bytes()))
        b=list(struct.iter_unpack('<7I',(fifo/phase/'tasks.bin').read_bytes()))
        if len(a)!=len(b) or any(x[:4]!=y[:4] or x[6]!=y[6] for x,y in zip(a,b)):
            raise ValueError('Wait-only ablation changed task ownership or participants')
    ratios=[pair_rows[i]['original']['metrics']['tpot_ms']/pair_rows[i]['fifo']['metrics']['tpot_ms']
            for i in range(protocol['pairs'])]
    interval=paired_interval(ratios)
    variability={kind:statistics.stdev([r['metrics']['tpot_ms'] for r in rows if r['variant']==kind])/
        statistics.mean([r['metrics']['tpot_ms'] for r in rows if r['variant']==kind]) for kind in ('original','fifo')}
    noisy=max(variability.values())>protocol['max_process_cv']
    result.update(status='inconclusive' if noisy else 'pass' if interval['lower95']>1 else 'no_reliable_gain',
        scope=f'{original_key[0]} B{original_key[1]} G{original_key[2]} wait-only optimization versus unchanged TileMega control',
        interval=interval,process_pair_speedups=ratios,process_cv=variability,
        noise_policy='All samples retained; excessive predeclared process CV makes this inconclusive',
        interference='Resident contexts/observed shared load retained; interval is local to these process pairs',
        process_medians={kind:{metric:statistics.median(r['metrics'][metric] for r in rows if r['variant']==kind)
            for metric in ('tpot_ms','ttft_ms','e2e_ms','throughput_tokens_per_s')} for kind in ('original','fifo')},
        binary_sha256=image,pairs=protocol['pairs'],requests_audited=8*protocol['pairs'],rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=summarize(a.directory,a.out)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}))
