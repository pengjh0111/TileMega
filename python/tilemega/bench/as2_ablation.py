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
    change=protocol.get('change','fifo_observation')
    candidate={'fifo_observation':'fifo','valid_qk64':'valid_qk',
               'cached_fp32_kv':'cached_kv', 'strided_swiglu_tile':'swiglu_tile'}.get(change)
    if candidate is None:raise ValueError('Unknown controlled ablation')
    variants=('original',candidate)
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
        if kind not in variants or job['pair_id']>=protocol['pairs']:
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
        if change in ('cached_fp32_kv','strided_swiglu_tile') and hashlib.sha256((binding/'request.json').read_bytes()).hexdigest()!=protocol['binding_sha256'][kind]:
            raise ValueError('Implementation binding changed after protocol freeze')
        key=tuple(identity[k] for k in ('model_id','batch','groups','config_sha256',
            'weight_manifest_sha256','input_sha256')+(() if change=='cached_fp32_kv' else ('arena_bytes',)))
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
    original,original_key=reference_bindings['original'];fifo,fifo_key=reference_bindings[candidate]
    if original_key!=fifo_key:raise ValueError('Model, weights, input or resource identity changed')
    if change=='cached_fp32_kv':validate_cached_kv_bindings(original,fifo)
    if change=='strided_swiglu_tile':validate_swiglu_bindings(original,fifo)
    for phase in (() if change=='cached_fp32_kv' else ('prefill','decode')):
        for name in ('queue.bin','offsets.bin','allocations.bin')+ (('stages.bin',) if change=='fifo_observation' else ('tasks.bin','waits.bin')):
            if (original/phase/name).read_bytes()!=(fifo/phase/name).read_bytes():
                raise ValueError('Wait-only ablation changed Body, placement, order or memory')
        if change=='valid_qk64':
            a=list(struct.iter_unpack('<17Q16I',(original/phase/'stages.bin').read_bytes()))
            b=list(struct.iter_unpack('<17Q16I',(fifo/phase/'stages.bin').read_bytes()))
            validate_qk_stages(a,b,phase)
        a=list(struct.iter_unpack('<7I',(original/phase/'tasks.bin').read_bytes()))
        b=list(struct.iter_unpack('<7I',(fifo/phase/'tasks.bin').read_bytes()))
        if len(a)!=len(b) or any(x[:4]!=y[:4] or x[6]!=y[6] for x,y in zip(a,b)):
            raise ValueError('Wait-only ablation changed task ownership or participants')
    ratios=[pair_rows[i]['original']['metrics']['tpot_ms']/pair_rows[i][candidate]['metrics']['tpot_ms']
            for i in range(protocol['pairs'])]
    interval=paired_interval(ratios)
    variability={kind:statistics.stdev([r['metrics']['tpot_ms'] for r in rows if r['variant']==kind])/
        statistics.mean([r['metrics']['tpot_ms'] for r in rows if r['variant']==kind]) for kind in variants}
    noisy=max(variability.values())>protocol['max_process_cv']
    result.update(status='inconclusive' if noisy else 'pass' if interval['lower95']>1 else 'no_reliable_gain',
        scope=f'{original_key[0]} B{original_key[1]} G{original_key[2]} {change} versus unchanged TileMega control',
        interval=interval,process_pair_speedups=ratios,process_cv=variability,
        noise_policy='All samples retained; excessive predeclared process CV makes this inconclusive',
        interference='Resident contexts/observed shared load retained; interval is local to these process pairs',
        process_medians={kind:{metric:statistics.median(r['metrics'][metric] for r in rows if r['variant']==kind)
            for metric in ('tpot_ms','ttft_ms','e2e_ms','throughput_tokens_per_s')} for kind in variants},
        binary_sha256=image,pairs=protocol['pairs'],requests_audited=8*protocol['pairs'],rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n');return result


def validate_swiglu_bindings(original,candidate):
    """Compare the registered epilogue with identical inputs/state and schedule."""
    identities=[json.loads((p/'request.json').read_text()) for p in (original,candidate)]
    if any((x['model_id'],x['batch'])!=('qwen3_1_7b',16) for x in identities):
        raise ValueError('SwiGLU study requires the registered Qwen1.7 B16 scope')
    for name in ('input_buffers','buffers','kv_buffers','derived_fp32_kv_state','arena_bytes'):
        if identities[0][name]!=identities[1][name]:
            raise ValueError('SwiGLU ablation changed canonical inputs or state')
    for phase in ('prefill','decode'):
        for p,x in zip((original,candidate),identities):
            for name,sha in x['phases'][phase]['bound_hashes'].items():
                if hashlib.sha256((p/phase/name).read_bytes()).hexdigest()!=sha:
                    raise ValueError('SwiGLU bound table changed')
        for name in ('metadata.bin','tasks.bin','waits.bin','queue.bin','offsets.bin','allocations.bin'):
            if (original/phase/name).read_bytes()!=(candidate/phase/name).read_bytes():
                raise ValueError('SwiGLU ablation changed ownership, waits, order or memory')
        stages=[list(struct.iter_unpack('<17Q16I',(p/phase/'stages.bin').read_bytes())) for p in (original,candidate)]
        validate_swiglu_stages(*stages,phase)


def validate_swiglu_stages(original,candidate,phase):
    if len(original)!=len(candidate):raise ValueError('SwiGLU ablation changed the stage domain')
    changed=0
    for a,b in zip(original,candidate):
        if a==b:continue
        if (phase!='decode' or tuple(a[17:19])!=(0,2) or tuple(a[20:23])!=(16,12288,2048) or
                a[31]!=1 or b[31]!=4 or
                a[:31]+a[32:]!=b[:31]+b[32:]):
            raise ValueError('SwiGLU ablation changed more than the registered implementation')
        changed+=1
    if phase=='decode' and changed!=28:
        raise ValueError('SwiGLU ablation requires all 28 Qwen1.7 gate/up stages')


def validate_qk_stages(original,candidate,phase):
    if len(original)!=len(candidate):raise ValueError('QK ablation changed the stage domain')
    changed=0
    for a,b in zip(original,candidate):
        if a==b:continue
        if (phase!='decode' or a[17]!=13 or a[31]!=6 or b[31]!=8 or
                a[:31]+a[32:]!=b[:31]+b[32:]):
            raise ValueError('QK ablation changed more than the registered implementation')
        changed+=1
    if phase=='decode' and not changed:raise ValueError('QK ablation has no changed attention Body')


def validate_cached_kv_bindings(original,candidate):
    """Allow derived state/relocation, preserving the logical graph and FIFO."""
    from tilemega.bench.as2_request import cached_kv_state
    a,b=[json.loads((p/'request.json').read_text()) for p in (original,candidate)]
    if a['input_buffers']!=b['input_buffers']:
        raise ValueError('Cached-KV ablation changed canonical packed inputs')
    maps=[{v['name']:v for v in x['buffers']} for x in (a,b)]
    if any(len(m)!=len(x['buffers']) for m,x in zip(maps,(a,b))):
        raise ValueError('Duplicate logical buffer names')
    for name,v in maps[0].items():
        other=maps[1].get(name,{})
        if {k:x for k,x in v.items() if k not in ('id','offset_bytes')}!={k:x for k,x in other.items() if k not in ('id','offset_bytes')}:
            raise ValueError('Cached-KV ablation changed a logical buffer')
    by_offsets=[{v['offset_bytes']:v['name'] for v in x['buffers']} for x in (a,b)]
    sentinel=(1<<64)-1;states=[]
    for phase,old_variant in (('prefill',6),('decode',8)):
        metadata=[struct.unpack('<12I3Q',(p/phase/'metadata.bin').read_bytes()) for p in (original,candidate)]
        if metadata[0][:12]+metadata[0][13:]!=metadata[1][:12]+metadata[1][13:]:
            raise ValueError('Cached-KV ablation changed launch or execution bounds')
        for p,x,m in zip((original,candidate),(a,b),metadata):
            if m[12]!=x['arena_bytes']:raise ValueError('Cached-KV arena identity mismatch')
            entry=x['phases'][phase]
            for name,sha in entry['bound_hashes'].items():
                if hashlib.sha256((p/phase/name).read_bytes()).hexdigest()!=sha:
                    raise ValueError('Cached-KV bound table changed')
        for name in ('tasks.bin','waits.bin','queue.bin','offsets.bin'):
            if (original/phase/name).read_bytes()!=(candidate/phase/name).read_bytes():
                raise ValueError('Cached-KV ablation changed task ownership, waits or order')
        stages=[list(struct.iter_unpack('<17Q16I',(p/phase/'stages.bin').read_bytes())) for p in (original,candidate)]
        if len(stages[0])!=len(stages[1]):raise ValueError('Cached-KV ablation changed stage count')
        state=cached_kv_state(stages[1],b['buffers'],metadata[1]);states.append(state)
        if not state:raise ValueError('Cached-KV ablation has no derived state')
        for s,t in zip(*stages):
            s,t=list(s),list(t)
            if s[17]==13:
                if s[31]!=old_variant or t[31]!=9 or s[10:12]!=[sentinel,sentinel]:
                    raise ValueError('Cached-KV ablation changed an unregistered Attention Body')
                t[10:12]=s[10:12];t[31]=s[31]
            for value,offsets in zip((s,t),by_offsets):
                for i in range(17):
                    if value[i]!=sentinel:
                        if value[i] not in offsets:raise ValueError('Unknown logical buffer address')
                        value[i]=offsets[value[i]]
            if s!=t:raise ValueError('Cached-KV ablation changed logical operands or mathematics')
    if states[0]!=states[1] or sorted(map(list,states[0]))!=b['derived_fp32_kv_state']:
        raise ValueError('Cached-KV phase state differs')
    added=set(maps[1])-set(maps[0]);derived={n for row in states[0] for n in row[2:]}
    if added!=derived:raise ValueError('Cached-KV ablation added unrelated storage')
    reset={v['name'] for v in b['buffers'] if v['id'] in b['kv_buffers']}
    if not {n for row in states[0] for n in row}<=reset:
        raise ValueError('Cached-KV derived state is not request-reset owned')
    if b['arena_bytes']-a['arena_bytes']!=sum(maps[1][n]['allocated_bytes'] for n in added):
        raise ValueError('Cached-KV memory increase is not the declared derived state')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=summarize(a.directory,a.out)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}))
