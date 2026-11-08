#!/usr/bin/env python3
"""Install measured F4 constants, retaining all native and F3 values verbatim."""
import argparse
import csv
import datetime
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/dnn4/phase_c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort(directory):
    paths = list((BASE / directory).glob('raw/*/attempt1/cal/result.json'))
    if len(paths) != 1:
        raise ValueError('missing unique calibration cohort: ' + directory)
    path = paths[0]
    result = json.loads(path.read_text())
    if result['status'] != 'passed' or any(row['passes'] != '5/5' for row in result['rows']):
        raise ValueError('incomplete calibration cohort: ' + str(path))
    build = json.loads(Path(result['build']).read_text())
    if any(sha(p) != value for p, value in build['inputs'].items()):
        raise ValueError('frozen calibration source changed')
    return path, result['rows'], build


def append_csv(path, rows, identity):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        old = [r for r in reader if r.get(identity) not in {x[identity] for x in rows}]
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(old + rows)


def install_section(path, section):
    # Historical sections mix compact and expanded JSON. Preserve their bytes.
    text = path.read_text()
    metadata = {k: v for k, v in section.items() if k != 'by_gpu'}
    value = json.dumps(metadata, indent=2).rstrip()[:-1].rstrip() + ',\n  "by_gpu": {"1": {\n'
    metrics = []
    for metric, rows in section['by_gpu']['1'].items():
        entries = ['      ' + json.dumps(key) + ': ' + json.dumps(row, separators=(',', ':'))
                   for key, row in rows.items()]
        metrics.append('    ' + json.dumps(metric) + ': {\n' + ',\n'.join(entries) + '\n    }')
    value += ',\n'.join(metrics) + '\n  }}\n}'
    value = value.replace('\n', '\n  ')
    found = re.search(r'"sm90_f4_constants"\s*:\s*', text)
    if found:
        _, size = json.JSONDecoder().raw_decode(text[found.end():])
        text = text[:found.end()] + value + text[found.end() + size:]
    else:
        end = text.rfind('}')
        text = text[:end].rstrip() + ',\n  "sm90_f4_constants": ' + value + '\n}\n'
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=ROOT / 'configs/targets/sm_90_h800.json')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {}
    tables = {}
    constants_csv = []
    calibration_csv = []
    header_sets = {}

    def read(name, directory):
        path, rows, build = cohort(directory)
        sources[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                             build_sha256=sha(path.parent.parent.parent.parent.parent / 'build.json')
                             if (path.parent.parent.parent.parent.parent / 'build.json').exists()
                             else sha(rows[0]['probe']['binary']))
        for p, value in build['inputs'].items():
            if '/include/tilemega/Backend/Sm90F3' in p:
                key = 'include/tilemega/Backend/' + Path(p).name
                if key in header_sets and value != header_sets[key]:
                    raise ValueError('different production headers across cohorts: ' + key)
                header_sets[key] = value
        manifest_sha = hashlib.sha256(json.dumps(build['inputs'], sort_keys=True).encode()).hexdigest()
        for row in rows:
            probe = row.get('probe', row.get('case', {}))
            measured = row['processes'][0]['measurement']
            medians = row.get('median', {})
            calibration_csv.append(dict(
                family='f4_' + name, threads=384,
                tile_m=probe.get('tm', probe.get('tile_m', '')),
                tile_n=probe.get('tn', probe.get('tile_n', '')),
                tk=probe.get('tk', ''), slots=probe.get('slots', measured.get('slots', '')),
                mode=probe.get('mode', 'swapab' if name == 'swap' else ''),
                kind=measured.get('kind', ''), attention_rows=measured.get('rows', ''),
                src=row.get('source', ''), active_ctas=measured.get('active_ctas', measured.get('grid', 132)),
                ns_per_k=medians.get('ns_per_k', medians.get('ns', '')),
                value_ns=row.get('median_ns', medians.get('ns_per_k', medians.get('ns', ''))),
                per_cta_gbps=medians.get('per_cta_gbps', ''), total_gbps=medians.get('total_gbps', ''),
                aliased_operands=measured.get('aliased_operands', False), passes='5/5',
                headers_manifest_sha256=manifest_sha,
                sass_sha256=probe.get('sass_sha256', build.get('sass_sha256', '')),
                spill_bytes=probe.get('spill_bytes', 0), split_k=probe.get('slices', ''),
                chain=probe.get('chain', ''), store=probe.get('store', ''),
                calibration_id=name + ':' + hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()[:16] + ':' +
                    str(row.get('args', row.get('source', ''))),
                evidence='verified', source=str(path.relative_to(ROOT))))
        return rows, str(path.relative_to(ROOT))

    def add(metric, selector, value, source, unit='ns', **extra):
        key = ';'.join(f'{k}={v}' for k, v in sorted(selector.items()))
        table = tables.setdefault(metric, {})
        if key in table:
            raise ValueError('duplicate measured key: ' + metric + '[' + key + ']')
        if value <= 0:
            raise ValueError('nonpositive measured value')
        table[key] = dict(median=value, unit=unit, processes=5, source=source,
                          evidence='verified', **extra)
        constants_csv.append(dict(key=metric + '[' + key + ']', mode=selector.get('mode', ''),
            tile_m=selector.get('tm', ''), tile_n=selector.get('tn', ''),
            split_k=selector.get('slices', ''), tk=selector.get('tk', ''),
            value_ns=value if unit.startswith('ns') else '', value=value, unit=unit,
            active_ctas=132, passes='5/5', gpu=1, selector=key,
            source=source, evidence='verified', **extra))

    main_rows, main_source = read('mainloop', 'mainloop_constants_v2')
    for row in main_rows:
        p = row['probe']
        tile = dict(mode=p['mode'], tm=p['tm'], tn=p['tn'], tk=p['tk'])
        if p['kind'] == 'resident':
            add('f4_mma_ns', tile, row['median']['ns'], main_source,
                sass_sha256=p['sass_sha256'])
        elif p['slots'] == 2 and int(row['args'][1]) == 132:
            add('f4_ring_ns', dict(tile, src=row['source']), row['median']['ns_per_k'],
                main_source, sass_sha256=p['sass_sha256'])
            if p['tm'] == p['tn'] == 128 and p['tk'] == 64:
                add('f4_first_ns', dict(src=row['source']), row['median']['first_ns'], main_source,
                    qualification='canonical128x128_TK64_Slots2_first_task', sass_sha256=p['sass_sha256'])

    bw, bw_source = read('bandwidth', 'bandwidth_constants_v2')
    for row in bw:
        p = row['probe']
        if p['tm'] == p['tn'] == 128 and p['tk'] == 64:
            add('f4_rate_gbps', dict(src=row['source'], n=int(row['args'][1])),
                row['median']['per_cta_gbps'], bw_source, unit='GB/s',
                qualification='canonical128x128_TK64_max_fitting_Slots7', sass_sha256=p['sass_sha256'])

    gather, gather_source = read('gather', 'gather_constants_v1')
    for row in gather:
        p = row['probe']
        if int(row['args'][1]) != 132:
            continue
        add('f4_ring_ns', dict(mode='gather', tm=p['tm'], tn=p['tn'], tk=p['tk'], src=row['source']),
            row['median']['ns_per_k'], gather_source, sass_sha256=p['sass_sha256'],
            qualification='production96lane_copy_protocol_dense16column_taps')

    epilogue, epi_source = read('epilogue', 'epilogue_constants_v1')
    for row in epilogue:
        c = row['case']
        add('f4_epilogue_ns', dict(tm=c['tile_m'], tn=c['tile_n'], store=c['store'], **{'class': c['chain']}),
            row['median_ns'], epi_source)

    split, split_source = read('split', 'split_constants_v2')
    for row in split:
        p = row['probe']
        add('f4_splitk_ns', dict(tm=p['tm'], tn=p['tn'], slices=p['slices']), row['median_ns'], split_source,
            sass_sha256=p['sass_sha256'], spill_bytes=p['spill_bytes'], qualification=p['qualification'])

    tasks, tasks_source = read('tasks', 'tasks_constants_v2')
    for row in tasks:
        p = row['probe']
        kind = row['processes'][0]['measurement']['kind']
        attention = kind == 'attention'
        add('f4_attention_ns' if attention else 'f4_small_ns',
            dict(rows=p['rows']) if attention else dict(kind=kind), row['median_ns'], tasks_source,
            unit='ns/task' if attention else 'ns/work_item', sass_sha256=p['sass_sha256'])

    swap, swap_source = read('swap', 'swap_constants_v1')
    for row in swap:
        p = row['probe']
        tile = dict(mode='swapab', tm=p['tm'], tn=128, tk=p['tk'])
        if p['resident']:
            add('f4_mma_ns', tile, row['median_ns'], swap_source, sass_sha256=p['sass_sha256'])
        else:
            add('f4_ring_ns', dict(tile, src=row['source']), row['median_ns'], swap_source,
                sass_sha256=p['sass_sha256'])
            if p['tk'] == 64 and row['source'] == 'l2':
                add('f4_epilogue_ns', dict(tm=p['tm'], tn=128, store='reg', **{'class': 'swapab_bias_act'}),
                    row['epilogue_ns'], swap_source, sass_sha256=p['sass_sha256'])

    gate_dirs = ['phase_r/rn18_b1_marks3_v1', 'phase_r/bert_b1_marks3_v1',
                 'phase_c/gate_v1/bert_b16_marks3', 'phase_c/gate_v1/vgg16_b16_marks3']
    gate_values = []
    for directory in gate_dirs:
        path = next((ROOT / 'runs/dnn4' / directory).glob('raw/*/attempt1/trace/ledger.csv'))
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        gate_values += [float(row['gate_tail_ns']) for row in rows[:-1]]
        sources[directory] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    # The specified gate constant comes from four sentinel traces rather than P-CAL.
    gate = statistics.median(gate_values)
    add('f4_gate_ns', {}, gate, 'four_sentinel_P_TRACE3_gate_tail_median',
        qualification='P_TRACE3_cross_card_sentinel_statistic_not_five_process_probe')
    tables['f4_gate_ns']['']['processes'] = 4
    tables['f4_gate_ns']['']['sampling_protocol'] = 'P-TRACE3-four-sentinels'
    constants_csv[-1].update(passes='4/4 sentinel traces', gpu='2/3')

    for header, value in header_sets.items():
        if sha(ROOT / header) != value:
            raise ValueError('production header differs from calibrated source: ' + header)
    target = json.loads(args.target.read_text())
    protected = {name: target[name] for name in ['sm90_native_constants', 'sm90_f3_constants']}
    target['sm90_f4_constants'] = dict(schema=1, variant='dnn4_384thread_measured_v1',
        method=dict(protocol='P-CAL five fresh processes; P-TRACE3 gate-tail exception',
            canonical_bandwidth='128x128 TK64 max fitting Slots7',
            first_latency='128x128 TK64 Slots2 first task',
            split_limitation='standalone168-register cap spills are recorded and included',
            qualification='provisional_container_pid_namespace', sources=sources),
        header_sha256=header_sets, by_gpu={'1': tables})
    if any(target[name] != value for name, value in protected.items()):
        raise ValueError('protected historical values changed')
    install_section(args.target, target['sm90_f4_constants'])
    receipt = dict(status='passed', evidence='verified', timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        target_sha256=sha(args.target), constant_count=len(constants_csv), sources=sources)
    (args.out / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    append_csv(ROOT / 'docs/results/DNN4_constants.csv', constants_csv, 'key')
    append_csv(ROOT / 'docs/results/DNN4_cal.csv', calibration_csv, 'calibration_id')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
