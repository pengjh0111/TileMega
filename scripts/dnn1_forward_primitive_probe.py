#!/usr/bin/env python3
"""Generate production Finish/ScalePanel probes from explicit CPU inventories."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
FAMILIES = {'pool': ('pool_finish_store_ns', 'ns', 'dnn1_pool_finish_probe.cuh'),
            'rowstat': ('rowstat_finish_store_ns', 'ns', 'dnn1_rowstat_finish_probe.cuh'),
            'sca': ('sca_panel_ns', 'ns/k64/cta', 'dnn1_sca_panel_probe.cuh')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def key_string(key):
    return ';'.join(f'{name}={value}' for name, value in key.items())


def pool_keys(inventory, layouts):
    inputs = read(inventory)['rows']
    layout = {(r['model'], r['batch'], r['producer_stage']): r for r in read(layouts)}
    keys = {}
    def add(key, case, instance):
        entry = keys.setdefault(key_string(key), dict(key=key, representative=case, instances=[]))
        entry['instances'].append(instance)
    for r in inputs:
        if r['batch'] != 1:
            continue
        model, stage = r['model'], r['pool_producer_stage']
        body, pixels, channels = r['producer_body'], r['rows_per_image'], r['pool_channels']
        pitch = layout[model, 1, stage]['typed_output_stride']
        if layout[model, 16, stage]['typed_output_stride'] != pitch:
            raise ValueError('B1/B16 output pitch differs')
        if body in ('gemm', 'conv'):
            for tm in (32, 64, 128):
                parts = (pixels + 2*tm - 2)//tm
                for batch in (1, 2, 4, 8, 16):
                    for rb in range(0, batch*pixels, tm):
                        live = min(tm, batch*pixels-rb)
                        for tn in (32, 64, 128, 256):
                            if channels % tn:
                                raise ValueError('pool channels require a new tail-width class')
                            key = dict(pool_rows=pixels, parts=parts, pool_channels=channels,
                                tm=tm, tn=tn, live_rows=live, width=tn, row_base_mod_image=rb%pixels,
                                stride=pitch, tw=0, th=0, image_width=0, live_tile_width=0)
                            add(key, dict(row_base=rb, allocation_batch=batch, channel=0),
                                dict(model=model, producer_stage=stage, batch=batch, row_base=rb))
        else:
            if body not in ('depthwise', 'dwpw', 'inverted_residual'):
                raise ValueError('unsupported spatial pool producer')
            for choice in r['choices']:
                tw, th = choice['tile_w'], choice['tile_h']
                w, h = r['output_width'], r['output_height']
                lw, lh = min(w, tw), min(h, th)
                if not ((w%tw == 0 and h%th == 0) or (w <= tw and h <= th)):
                    raise ValueError('unrepresented spatial boundary class')
                for tn in ((64,) if body == 'depthwise' else (64, 128, 256)):
                    tm = tw*th if body == 'depthwise' else 64
                    width = 32 if body == 'depthwise' else min(tn, channels)
                    key = dict(pool_rows=pixels, parts=choice['parts'], pool_channels=channels,
                        tm=tm, tn=tn, live_rows=lw*lh, width=width, row_base_mod_image=0,
                        stride=pitch, tw=tw, th=th, image_width=w, live_tile_width=lw)
                    add(key, dict(row_base=0, allocation_batch=1, channel=0),
                        dict(model=model, producer_stage=stage, batch='all five',
                             row_base='first tile; structurally equal full tiles or sole edge'))
    return list(keys.values()), {str(p): sha(p) for p in (inventory, layouts)}


def rowstat_keys(directory):
    keys, sources = {}, {}
    for model in ('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'):
        for batch in (1, 16):
            path = directory/f'{model}_b{batch}.json'
            sources[str(path)] = sha(path)
            for stage in read(path)['stages']:
                g = stage['dnn1_geometry']
                if not g.get('has_row_stats') or stage['body'] not in ('gemm', 'conv'):
                    continue
                if (g['gated'] or g['store_kind'] != 'vector_16byte' or
                        stage['m'] != g['rows'] or g['store_columns'] != stage['n']):
                    raise ValueError('unrepresented row-statistics write mapping')
                for tm in (32, 64, 128):
                    if stage['m'] % tm:
                        raise ValueError('actual tail rows require a separate key')
                    for tn in (32, 64, 128, 256):
                        if tn > stage['n']:
                            continue
                        if stage['n'] % tn:
                            raise ValueError('actual tail columns require a separate key')
                        key = dict(tm=tm, tn=tn, live_rows=tm, tiles=stage['n']//tn,
                            store_columns=tn, store='dense_vector16', output_pitch=g['output_pitch'])
                        entry = keys.setdefault(key_string(key), dict(key=key, instances=[]))
                        entry['instances'].append(dict(model=model, batch=batch, stage=stage['index']))
    return list(keys.values()), sources


def sca_keys(paths):
    shapes = set()
    for path in paths:
        shapes.update(tuple(map(int, m)) for m in re.findall(
            r'Sm90ForwardInputScale<(\d+),(\d+),(\d+)>', path.read_text()))
    if not shapes or any(rows%128 or (channels != 32 and channels%64) or stride < channels
                         for rows, channels, stride in shapes):
        raise ValueError('unrepresented ScalePanel shape')
    live = sorted({32 if channels == 32 else 64 for rows, channels, stride in shapes})
    keys = [dict(key=dict(tm=tm, tk=64, live_k=k, implementation='typed_cached_per_lane_bf16'))
            for tm in (32, 64, 128) for k in live]
    return keys, {str(p): sha(p) for p in paths}


def target_keys(path, family):
    """Reproduce a frozen catalog without requiring ignored inventory files."""
    section = read(path)['sm90_forward_constants']
    metric, unit, _ = FAMILIES[family]
    if section['schema'] == 1:
        card = section['by_gpu'][sorted(section['by_gpu'])[0]]
        encoded = [(metric, key, row['unit']) for key, row in card[metric].items()]
    elif section['schema'] == 2:
        encoded = section['keys']
    elif section['schema'] == 3:
        encoded = []
        for row in section['keys']:
            m, u, names = section['key_layouts'][row[0]]
            encoded.append((m, ';'.join(k+'='+v for k, v in zip(names, row[1:])), u))
    else:
        raise ValueError('unknown calibration schema')
    rows = []
    for m, text, u in encoded:
        if m != metric:
            continue
        if u != unit:
            raise ValueError('calibration unit differs from the production primitive')
        key = {}
        for field in text.split(';'):
            name, value = field.split('=', 1)
            if name in key:
                raise ValueError('duplicate selector')
            if value.isdecimal():
                if str(int(value)) != value:
                    raise ValueError('geometry must use canonical decimal integers')
                value = int(value)
            key[name] = value
        row = dict(key=key)
        if family == 'pool':
            if key['tw']:
                representative = dict(row_base=0, allocation_batch=1, channel=0)
            else:
                representatives = [dict(row_base=rb, allocation_batch=batch, channel=0)
                    for batch in (1,2,4,8,16)
                    for rb in range(0, batch*key['pool_rows'], key['tm'])
                    if rb%key['pool_rows'] == key['row_base_mod_image'] and
                       min(key['tm'], batch*key['pool_rows']-rb) == key['live_rows']]
                if not representatives:
                    raise ValueError('frozen key lacks an actual batch1/2/4/8/16 producer tile')
                representative = representatives[0]
            row['representative'] = representative
        rows.append(row)
    if not rows:
        raise ValueError('target contains no keys for this primitive')
    return rows, {str(path): sha(path)}


def sca_oracle(tm, live):
    # Independent CPU BF16 multiplication; no device kernel or CUDA API runs.
    import torch
    x = ((torch.arange(tm*64, dtype=torch.float32, device='cpu')%97-48)/37).reshape(tm,64).to(torch.bfloat16)
    scale = ((torch.arange(64, dtype=torch.float32, device='cpu')%17-8)/11).to(torch.bfloat16)
    scale[0] = 0
    expected = (x.float()*scale.float()).to(torch.bfloat16)
    expected[:, live:] = 0
    def physical(tensor):
        words, data = [0]*(tm*64), tensor.view(torch.int16).tolist()
        for row in range(tm):
            for col in range(64):
                words[row*64+((col//8)^(row%8))*8+col%8] = data[row][col] & 65535
        return words
    return dict(tm=tm, live_k=live, scale_bf16_words=[int(v)&65535 for v in scale.view(torch.int16)],
                input_physical_bf16_words=physical(x), expected_physical_bf16_words=physical(expected))


def generate(args):
    catalog = read(args.required)
    family = next(name for name, value in FAMILIES.items() if value[0] == catalog['metric'])
    source_root = args.source_root.resolve()
    if not (source_root/'include/tilemega/Backend/Sm90ForwardEpilogue.cuh').is_file():
        raise ValueError('source-root lacks the production forward headers')
    headers = {str(p.relative_to(source_root)): sha(p) for p in
        (source_root/'include/tilemega/Backend').glob('Sm90Forward*') if p.is_file()}
    source = args.base_source.read_text()
    if family == 'sca':
        source = source.replace('#include <tilemega/Backend/Sm90ForwardSmall.cuh>',
            '#include <tilemega/Backend/Sm90ForwardSmall.cuh>\n#include <tilemega/Backend/Sm90ForwardGemm.cuh>')
    spans = []
    for match in re.finditer(r'void Catalog\(\) \{', source):
        depth, end = 1, match.end()
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        spans.append((match.start(), end))
    if len(spans) != 2:
        raise ValueError('base probe must contain direct and segments catalogs')
    args.out.mkdir(parents=True, exist_ok=False)
    lines, oracles = [], {}
    for index, row in enumerate(catalog['keys']):
        k = row['key']
        if args.tm is not None and k['tm'] not in args.tm:
            continue
        if any(type(v) is not int or v < 0 for n,v in k.items() if n not in ('store','implementation')):
            raise ValueError('nonnegative integer geometry required')
        key = json.dumps(key_string(k))
        if family == 'pool':
            r = row['representative']
            template = [k[n] for n in ('tm','tn','pool_rows','parts','pool_channels','tw','th','image_width')]
            runtime = [r['row_base'], k['live_rows'], k['width'], k['stride'], r['allocation_batch']]
            lines.append(f'  PoolFinishCase<{",".join(map(str,template))}>("pool_finish_{index}",{key},{",".join(map(str,runtime))});')
        elif family == 'rowstat':
            if k['store_columns'] != k['tn'] or k['live_rows'] != k['tm']:
                raise ValueError('rowstat probe does not support this boundary class')
            template = [k[n] for n in ('tm','tn','tiles')]
            lines.append(f'  RowstatFinishCase<{",".join(map(str,template))}>("rowstat_finish_{index}",{key},{k["live_rows"]},{k["output_pitch"]});')
        else:
            oracle = sca_oracle(k['tm'], k['live_k'])
            path = args.out/f'oracle_tm{k["tm"]}_k{k["live_k"]}.json'
            write(path, oracle);oracles[str(path)] = sha(path)
            arrays = [oracle['input_physical_bf16_words'], oracle['scale_bf16_words'][:k['live_k']], oracle['expected_physical_bf16_words']]
            values = ','.join('{'+','.join(map(str, array))+'}' for array in arrays)
            lines.append(f'  ScaCase<{k["tm"]},{k["live_k"]}>({values});')
    if args.legacy_pending_tm32:
        if family != 'sca' or args.tm is None or 32 in args.tm:
            raise ValueError('legacy pending32 applies only to the old64/128 recipe')
        for live in (32, 64):
            lines.append(f'  Pending("sca_panel_32_{live}","sca_panel_ns","tm=32;tk=64;live_k={live};implementation=typed_cached_per_lane_bf16","real TM32 body integration pending; TM64 with32rows is not an equivalent probe");')
    primitive = HERE/FAMILIES[family][2]
    body = primitive.read_text()+'\nvoid Catalog() {\n'+'\n'.join(lines)+'\n}\n'
    for index, (start, end) in reversed(list(enumerate(spans))):
        source = source[:start]+(body if index == 0 else 'void Catalog() {}')+source[end:]
    output = args.out/'dnn1_constants.cu';output.write_text(source)
    write(args.out/'generation.json', dict(scope='CPU generation only; use original guard/scheduler for every GPU probe',
        family=family, required_sha256=sha(args.required), base_sha256=sha(args.base_source),
        primitive_sha256=sha(primitive), generated_sha256=sha(output), source_root=str(source_root), production_header_sha256=headers,
        oracle_sha256=oracles, selected_tm=args.tm, legacy_pending_tm32=args.legacy_pending_tm32))
    if args.build:
        sys.path.insert(0, str(args.base_source.parent))
        import dnn1_constants as constants
        constants.SOURCE = output
        # TM32 math is an explicit dependency even when only ScalePanel runs.
        if (source_root/'include/tilemega/Backend/Sm90ForwardMma32.cuh').is_file():
            constants.FORWARD_HEADERS += ('Backend/Sm90ForwardMma32.cuh',)
        constants.build(argparse.Namespace(out=args.out/'build', source_root=args.source_root,
            implementation='fixed', nvcc=args.nvcc, timeout_s=args.timeout_s))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    keys = sub.add_parser('keys')
    keys.add_argument('--family', choices=FAMILIES, required=True)
    keys.add_argument('--target', type=Path, help='reproduce a frozen schema1/2/3 catalog')
    keys.add_argument('--pool-inventory', type=Path)
    keys.add_argument('--layouts', type=Path)
    keys.add_argument('--inventory-dir', type=Path)
    keys.add_argument('--generated-cu', type=Path, nargs='+')
    keys.add_argument('--out', type=Path, required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--required', type=Path, required=True)
    gen.add_argument('--source-root', type=Path, required=True)
    gen.add_argument('--base-source', type=Path, default=HERE/'dnn1_constants.cu')
    gen.add_argument('--out', type=Path, required=True)
    gen.add_argument('--tm', type=int, choices=(32,64,128,256), nargs='+')
    gen.add_argument('--legacy-pending-tm32', action='store_true')
    gen.add_argument('--build', action='store_true')
    gen.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    gen.add_argument('--timeout-s', type=int, default=5400)
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args)
        return
    if args.target:
        rows, sources = target_keys(args.target, args.family)
    elif args.family == 'pool':
        if not args.pool_inventory or not args.layouts:
            parser.error('pool keys need --pool-inventory and --layouts')
        rows, sources = pool_keys(args.pool_inventory, args.layouts)
    elif args.family == 'rowstat':
        if not args.inventory_dir:
            parser.error('rowstat keys need --inventory-dir')
        rows, sources = rowstat_keys(args.inventory_dir)
    else:
        if not args.generated_cu:
            parser.error('sca keys need --generated-cu')
        rows, sources = sca_keys(args.generated_cu)
    metric, unit, _ = FAMILIES[args.family]
    write(args.out, dict(status='CPU required keys; not calibration', metric=metric, unit=unit,
        key_count=len(rows), source_sha256=sources, keys=rows,
        scope='Pool: exact dense row residues/live rows; spatial channel0 first tile is explicit structural transfer. Rowstat: complete dense row tiles. SCA: runtime offsets, actual typed helper; higherK full panels transferred by instruction shape only.'))


if __name__ == '__main__':
    main()
