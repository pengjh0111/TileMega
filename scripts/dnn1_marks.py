#!/usr/bin/env python3
"""DNN1 forward trace ABI and offline stage ledger (attribution, never timing)."""
from __future__ import annotations

import argparse
import csv
import ctypes as C
import hashlib
import json
import statistics
import struct
from pathlib import Path


class ForwardTrace:
    """Attach after normal plan construction; caller owns guard, lock and stream.

    Warm up first, reset, launch exactly once, then save. Reset/copy synchronize
    the device. The static trace array belongs to the library's CUDA module and
    must not be shared by concurrent launches or reused without resetting.
    """
    def __init__(self, library):
        self.library = library
        self.lib = library if isinstance(library, C.CDLL) else C.CDLL(str(library))
        names = ('schema', 'stages', 'grid', 'segments', 'record_bytes', 'bytes')
        self.info = {}
        for name in names:
            fn = getattr(self.lib, 'tm_forward_trace_' + name)
            fn.argtypes = []
            fn.restype = C.c_uint64 if name == 'bytes' else C.c_uint32
            self.info[name] = int(fn())
        if self.info['schema'] != 1:
            raise ValueError('unknown forward trace ABI')
        expected = 8 * (4 + 4 * self.info['segments'])
        if self.info['record_bytes'] != expected or self.info['bytes'] != (
                expected * self.info['stages'] * self.info['grid']):
            raise ValueError('forward trace layout differs from ABI 1')
        fn = self.lib.tm_forward_trace_segment_name
        fn.argtypes, fn.restype = [C.c_uint32], C.c_char_p
        self.info['segment_names'] = [fn(i).decode() for i in range(self.info['segments'])]
        self.lib.tm_forward_trace_reset.argtypes = []
        self.lib.tm_forward_trace_reset.restype = C.c_int
        self.lib.tm_forward_trace_copy.argtypes = [C.c_void_p, C.c_uint64]
        self.lib.tm_forward_trace_copy.restype = C.c_int

    def reset(self):
        status = self.lib.tm_forward_trace_reset()
        if status:
            raise RuntimeError(f'tm_forward_trace_reset: CUDA status {status}')

    def save(self, destination, *, metadata=None):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        buffer = C.create_string_buffer(self.info['bytes'])
        status = self.lib.tm_forward_trace_copy(buffer, self.info['bytes'])
        if status:
            raise RuntimeError(f'tm_forward_trace_copy: CUDA status {status}')
        raw = bytes(buffer)
        info = dict(self.info, clock='globaltimer nanoseconds', launches=1,
                    aggregation='consumer thread 0: body segments sum/count; scalar small phases use first-to-last windows/count',
                    timing_eligible=False, raw_sha256=hashlib.sha256(raw).hexdigest(),
                    metadata=metadata or {})
        if not isinstance(self.library, C.CDLL):
            info['library'] = str(Path(self.library).resolve())
            info['library_sha256'] = hashlib.sha256(Path(self.library).read_bytes()).hexdigest()
        (destination / 'trace.bin').write_bytes(raw)
        (destination / 'trace.json').write_text(json.dumps(info, indent=2) + '\n')


def read_trace(directory):
    directory = Path(directory)
    info = json.loads((directory / 'trace.json').read_text())
    raw = (directory / 'trace.bin').read_bytes()
    if info['schema'] != 1 or info['launches'] != 1 or info['timing_eligible']:
        raise ValueError('expected one attribution-only ABI 1 launch')
    if len(raw) != info['bytes'] or hashlib.sha256(raw).hexdigest() != info['raw_sha256']:
        raise ValueError('trace bytes differ from their receipt')
    names = info['segment_names']
    record = struct.Struct('<' + 'Q' * (4 + 4 * len(names)))
    if record.size != info['record_bytes']:
        raise ValueError('trace record layout differs from metadata')
    by_stage = []
    for stage in range(info['stages']):
        group = []
        for cta in range(info['grid']):
            values = record.unpack_from(raw, (stage * info['grid'] + cta) * record.size)
            enter, arrive, leave, ready = values[:4]
            if not enter or not enter <= arrive <= leave:
                raise ValueError(f'incomplete/nonmonotonic stage {stage}, CTA {cta}')
            if ready and not enter <= ready <= arrive:
                raise ValueError(f'first_ready outside body: stage {stage}, CTA {cta}')
            spans = {}
            for i, name in enumerate(names):
                total, count, first, last = values[4 + i*4:8 + i*4]
                if count and not enter <= first <= last <= arrive:
                    raise ValueError(f'span outside body: stage {stage}, CTA {cta}, {name}')
                spans[name] = dict(total_ns=total, count=count, first_begin=first, last_end=last)
            group.append(dict(cta=cta, enter=enter, arrive=arrive, leave=leave,
                              first_ready=ready, spans=spans))
        by_stage.append(group)
    return info, by_stage


def classify(work, stage_ns, spans, dram_gbps, cta_gbps, lower_memory_ns, lower_compute_ns):
    get = lambda name: spans.get(name, {}).get('total_ns', 0)
    concurrency = min(work['tasks'], work['grid'])
    if concurrency * cta_gbps < .5 * dram_gbps and lower_memory_ns >= lower_compute_ns:
        return 'RC-1'
    if work.get('k_iter') is not None and work['k_iter'] < 2 * work['slots']:
        return 'RC-2'
    if get('epilogue') > stage_ns / 3:
        return 'RC-3'
    if get('partial_write') + get('splitk_reduce') > stage_ns / 4:
        return 'RC-4'
    if work['body'] in ('layout', 'pool', 'global_pool', 'gemv', 'layer_norm', 'embedding'):
        nonempty = [spans[name] for name in ('small_load', 'small_compute', 'small_store')
                    if spans.get(name, {}).get('count')]
        # Literal §4.2 RC-5: the small body's observed envelope exceeds one
        # twentieth of its stage. It is not an arithmetic sum of scalar spans.
        if nonempty and max(x['last_end'] for x in nonempty) - min(
                x['first_begin'] for x in nonempty) > stage_ns / 20:
            return 'RC-5'
    if get('attention_softmax') > stage_ns / 3 or get('attention_pv') > stage_ns / 3:
        return 'RC-6'
    return 'RC-7'


def analyze(directory, work, target, peak_tflops, process):
    info, stages = read_trace(directory)
    if len(work) != len(stages) or len({int(w['index']) for w in work}) != len(stages):
        raise ValueError('work ledger must cover every stage exactly once')
    work = sorted(work, key=lambda w: int(w['index']))
    constants = target['sm90_native_constants']
    dram, cta = constants['dram_gbps'], constants['cta_dram_gbps']
    previous_completion = min(x['enter'] for x in stages[0])
    previous_arrival = previous_completion
    ledger, segment_rows = [], []
    for index, group in enumerate(stages):
        w = dict(work[index], grid=info['grid'])
        if int(w['index']) != index or any(key not in w for key in (
                'dram_bytes', 'flops', 'tasks', 'body', 'slots', 'geometry_source',
                'k_iter', 'weight_bytes_per_task', 'm', 'tm', 'covered_batches', 'runtime_batch')):
            raise ValueError('work ledger lacks measured-plan geometry/traffic/source fields')
        enter = min(x['enter'] for x in group)
        arrival = max(x['arrive'] for x in group)
        completion = max(x['leave'] for x in group)
        # A successor may enter before the slowest CTA has left the previous
        # local barrier. It may not enter before every previous CTA arrived.
        if index and enter < previous_arrival:
            raise ValueError(f'stage overlap across the grid barrier at stage {index}')
        critical = max(group, key=lambda x: x['arrive'])
        stage_ns = completion - previous_completion
        if stage_ns <= 0:
            raise ValueError('nonpositive program-level stage window')
        memory_ns = w['dram_bytes'] / dram
        compute_ns = w['flops'] / (.8 * peak_tflops * 1e3)
        lower = max(memory_ns, compute_ns)
        if lower <= 0:
            raise ValueError('explicit work ledger must provide a positive lower bound')
        logical_flops=w.get('logical_flops');logical_weights=w.get('logical_weight_bytes')
        logical_compute=(logical_flops/(.8*peak_tflops*1e3) if logical_flops is not None else None)
        logical_memory=(logical_weights/dram if logical_weights is not None else None)
        logical_terms=[x for x in (logical_compute,logical_memory) if x is not None]
        logical_lower=max(logical_terms) if logical_terms else None
        spans = critical['spans']
        small_active=[spans[n] for n in ('small_load','small_compute','small_store') if spans[n]['count']]
        small_envelope=(max(x['last_end'] for x in small_active)-min(x['first_begin'] for x in small_active)
                        if small_active else 0)
        category = classify(w, stage_ns, spans, dram, cta, memory_ns, compute_ns)
        row = dict(process=process, index=index, body=w['body'],
                   representative=w.get('representative', ''), critical_cta=critical['cta'],
                   stage_ns=stage_ns, body_ns=critical['arrive']-critical['enter'],
                   small_envelope_ns=small_envelope,
                   barrier_tail_ns=completion-arrival,
                   critical_barrier_tail_ns=critical['leave']-critical['arrive'],
                   first_ready_ns=(critical['first_ready']-critical['enter']
                                   if critical['first_ready'] else ''),
                   lower_bound_ns=lower, current_implementation_lower_ns=lower,
                   model_logical_valid_lower_ns=logical_lower if logical_lower is not None else '',
                   model_logical_flops=logical_flops if logical_flops is not None else '',
                   model_logical_weight_bytes=logical_weights if logical_weights is not None else '',
                   model_logical_compute_ns=logical_compute if logical_compute is not None else '',
                   model_logical_weight_ns=logical_memory if logical_memory is not None else '',
                   model_logical_status=w.get('logical_status','unknown'),
                   model_logical_weight_scope=w.get('logical_weight_scope','unknown'),
                   model_logical_work_complete=w.get('logical_work_complete',False),
                   lower_memory_ns=memory_ns,
                   lower_compute_ns=compute_ns, excess_ns=stage_ns-lower,
                   measured_over_lower=stage_ns/lower, category=category,
                   tasks=w['tasks'], concurrent_ctas=min(w['tasks'], info['grid']),
                   grid=info['grid'], k_iter=w['k_iter'], slots=w['slots'],
                   weight_bytes_per_task=w['weight_bytes_per_task'],
                   m=w['m'], tm=w['tm'], row_utilization=(w['m'] / (
                       ((w['m']+w['tm']-1)//w['tm'])*w['tm']) if w['m'] and w['tm'] else ''),
                   tail_row_utilization=((w['m'] % w['tm']) / w['tm']
                       if w['m'] and w['tm'] and w['m'] % w['tm'] else 1),
                   geometry_source=w['geometry_source'], trace=str(Path(directory).resolve()),
                   covered_batches=w.get('covered_batches',''),required_batches=w.get('runtime_batch',''),
                   coverage_status=('local_scope_missing_batches' if w.get('covered_batches')!=w.get('runtime_batch')
                                    else 'local_scope_full_batch; upstream correctness not asserted'),
                   bound_scope='actual typed body extent; correctness requires separate evidence',
                   geometry_notes=json.dumps(w.get('notes',[]),ensure_ascii=False))
        for name, span in spans.items():
            row[name+'_ns'], row[name+'_count'] = span['total_ns'], span['count']
            active = [g['spans'][name] for g in group if g['spans'][name]['count']]
            segment_rows.append(dict(process=process, index=index, body=w['body'],
                segment=name, critical_cta=critical['cta'], critical_ns=span['total_ns'],
                critical_count=span['count'], sum_cta_ns=sum(s['total_ns'] for s in active),
                sum_cta_count=sum(s['count'] for s in active),
                max_cta_ns=max((s['total_ns'] for s in active), default=0),
                active_ctas=len(active), additive=False))
        ledger.append(row)
        previous_completion, previous_arrival = completion, arrival
    return ledger, segment_rows


def write_csv(path, rows):
    path = Path(path)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True, action='append')
    p.add_argument('--work', type=Path, required=True,
                   help='stage work JSON: explicit traffic/FLOPs/tasks from the compiled plan')
    p.add_argument('--captured-work', type=Path, help='original captured geometry; permits explicit reanalysis of identical artifact and inputs')
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--peak-tflops', type=float, required=True,
                   help='stated or measured peak; source must be recorded in --peak-source')
    p.add_argument('--peak-source', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.peak_tflops <= 0:
        p.error('peak must be positive')
    work = json.loads(args.work.read_text())
    captured_hash = hashlib.sha256(args.work.read_bytes()).hexdigest()
    geometry_reanalysis = None
    if args.captured_work:
        captured = json.loads(args.captured_work.read_text())
        for key in ('source_sha256', 'manifest_sha256', 'logical_plan_sha256',
                    'logical_inputs_sha256', 'runtime_batch', 'grid'):
            if not work.get(key) or work.get(key) != captured.get(key):
                raise ValueError('corrected geometry refers to another artifact/input: '+key)
        if len(work['stages']) != len(captured['stages']):
            raise ValueError('corrected geometry changes stage count')
        captured_hash = hashlib.sha256(args.captured_work.read_bytes()).hexdigest()
        geometry_reanalysis = dict(captured_work=str(args.captured_work),
            captured_work_sha256=captured_hash, corrected_work=str(args.work),
            corrected_work_sha256=hashlib.sha256(args.work.read_bytes()).hexdigest(),
            source_sha256=work['source_sha256'], method='redecode generated template arity; retain original trace metadata and bytes')
    if isinstance(work, dict):
        if work.get('runtime_batch')!=args.batch:
            raise ValueError('work JSON belongs to another batch')
        work = work['stages']
    target = json.loads(args.target.read_text())
    rows, segments = [], []
    for process, directory in enumerate(args.trace):
        trace_info,_=read_trace(directory)
        binding=trace_info.get('metadata',{})
        expected_work=binding.get('work_sha256')
        if expected_work and expected_work!=captured_hash:
            raise ValueError('trace was captured with another work JSON')
        if binding.get('model',args.model)!=args.model or binding.get('batch',args.batch)!=args.batch:
            raise ValueError('trace metadata belongs to another cell')
        current, current_segments = analyze(directory, work, target, args.peak_tflops, process)
        rows.extend(dict(model=args.model, batch=args.batch, **r) for r in current)
        segments.extend(dict(model=args.model, batch=args.batch, **r) for r in current_segments)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / 'stage_ledger_processes.csv', rows)
    write_csv(args.out / 'segments.csv', segments)
    merged = []
    for index in range(len(work)):
        group = [r for r in rows if r['index'] == index]
        median = statistics.median(r['stage_ns'] for r in group)
        median_spans = {name:dict(total_ns=statistics.median(r[name+'_ns'] for r in group))
                        for name in ('epilogue', 'partial_write', 'splitk_reduce',
                                     'attention_softmax', 'attention_pv')}
        # RC-5 uses each process's envelope; take its median without pretending
        # the load/compute/store clocks form a pure or additive decomposition.
        envelope = statistics.median(r['small_envelope_ns'] for r in group)
        if envelope:
            median_spans['small_compute']=dict(total_ns=0,count=1,first_begin=0,last_end=envelope)
        w = dict(sorted(work,key=lambda x:int(x['index']))[index],grid=group[0]['grid'])
        category = classify(w,median,median_spans,target['sm90_native_constants']['dram_gbps'],
                            target['sm90_native_constants']['cta_dram_gbps'],
                            group[0]['lower_memory_ns'],group[0]['lower_compute_ns'])
        row = dict(group[0])
        for key in row:
            numeric = [r[key] for r in group if isinstance(r[key], (int, float))
                       and not isinstance(r[key], bool)]
            if (key.endswith(('_ns', '_count')) or key == 'measured_over_lower') and numeric:
                row[key] = statistics.median(numeric)
        row.pop('process')
        row.update(stage_ns=median, category=category,
            critical_ctas=';'.join(str(r['critical_cta']) for r in group),
            processes=len(group),
            aggregation='median of per-process critical-CTA fields; spans need not add')
        # The representative CTA belongs to the process closest to the stage
        # median. All process-level CTA/span pairings remain in the raw CSV.
        representative = min(group, key=lambda r: abs(r['stage_ns']-median))
        row['critical_cta'] = representative['critical_cta']
        row['representative_process'] = representative['process']
        for key in ('n','k','pixels','channels'):
            row[key] = w.get(key) or ''
        merged.append(row)
    write_csv(args.out / 'stage_ledger.csv', merged)
    total = sum(r['stage_ns'] for r in merged)
    shares = {f'RC-{i}':sum(r['stage_ns'] for r in merged if r['category']==f'RC-{i}')/total
              for i in range(1,8)}
    lower = sum(r['lower_bound_ns'] for r in merged)
    low_util = sum(r['stage_ns'] for r in merged if r['body'] in ('gemm','conv')
                   and r['row_utilization'] != ''
                   and r['row_utilization'] < .8) / total
    rc1_first = sum(r['stage_ns'] for r in merged if r['category']=='RC-1'
                    and r['first_ready_ns'] != '' and r['first_ready_ns'] > r['stage_ns']/4) / total
    summary = dict(model=args.model, batch=args.batch, processes=len(args.trace),
                   stage_time_ns=total, category_shares=shares,
                   serial_stage_lower_bound_ns=lower, lower_over_instrumented_stage_time=lower/total,
                   low_m_row_utilization_time_share=low_util,
                   rc1_first_ready_over_quarter_time_share=rc1_first,
                   top10_excess=[{key:r[key] for key in ('index','representative','body','category',
                       'stage_ns','lower_bound_ns','excess_ns','measured_over_lower','tasks',
                       'concurrent_ctas','k_iter','slots','weight_bytes_per_task','m','tm',
                       'row_utilization','covered_batches','required_batches')}
                       for r in sorted(merged,key=lambda r:r['excess_ns'],reverse=True)[:10]],
                   dt_a='STOP_RC7' if shares['RC-7']>.2 else 'continue',
                   peak_tflops=args.peak_tflops, peak_source=args.peak_source,
                   geometry_reanalysis=geometry_reanalysis, coverage_scope='typed implementation extent; no full-correct-forward assertion',
                   missing_batch_stage_indices=[int(w['index']) for w in work
                       if w.get('covered_batches')!=w.get('runtime_batch')],
                   critical_cta='argmax CTA arrival timestamp for each stage/process',
                   stage_window='max leave[i] - max leave[i-1]; first starts at min enter[0]',
                   segment_aggregation='critical CTA thread0 sum across local tasks; all-CTA sum/max are separate nonadditive diagnostics',
                   limitations=['instrumented builds cannot provide E2E latency',
                       'small phases use two clock reads per phase/CTA; windows contain interleaved operations and cannot be added',
                       'slot/halo/KV waits may nest in compute envelopes; never sum all segments',
                       'CPU work geometry/traffic must be explicit, not copied blindly from old solver shapes'])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    return 2 if summary['dt_a']=='STOP_RC7' else 0


if __name__ == '__main__':
    raise SystemExit(main())
