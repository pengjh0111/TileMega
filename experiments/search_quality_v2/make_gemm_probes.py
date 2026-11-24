"""Emit compact per-family GEMM probe suites from the native coordinate catalogs."""

import hashlib
import json

from calibration_queue import CALIB, CASE
from guarded import ROOT, read, write


V1 = ROOT / 'docs/results/sm90_search_quality/qwen3_1_7b_b1/frozen_search.json'
OP = {'qkv': 'kStore', 'o': 'kResidual', 'gate_up': 'kSwiGLU',
      'down': 'kResidual', 'lm_head': 'kArgmaxPartial'}
MODELS = {
    'qwen3_1_7b': (CASE / 'native-domains.jsonl', 'configs/serving/sm90d1/inputs/qwen3_1_7b/config.json'),
    'llama3_1_8b': (CALIB / 'llama-native-domains.jsonl', 'configs/serving/sm90d1/inputs/llama3_1_8b/config.json'),
}
CSV_FIELDS = ('geometry_id,repeat,tasks,grid,head_entry_ns,head_body_ns,steady_ns,'
              'stage_entry_ns,stage_body_ns,kernel_gpu_event_ns,cta_weight_bytes_per_ns,evicted_bytes')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source(rows, epsilon):
    entry_points = []
    for row in rows:
        g = row['geometry']
        if g['multicast'] or g['mma'] != 'mma_sync':
            raise ValueError(f'unsupported GEMM probe geometry {g}')
        norm = 'true' if row['family'] in ('qkv', 'gate_up', 'lm_head') else 'false'
        u = row['swiglu_u'] if row['family'] == 'gate_up' else 16
        dimensions = [row['n'], row['k'], g['tile_n'], row['tile_k'], g['slots'],
                      g['warps_n'], g['warps_k'], g['register_stages'], row['residency'],
                      f'tilemega::backend::ServingEpilogueOp::{OP[row["family"]]}', u, norm]
        template = ', '.join(map(str, dimensions))
        incoming = f'{epsilon:.8e}f'
        entry_points.append(f'    GemmCase<{template}>(buffers,{incoming},shared,"{row["id"]}");')
    return '\n'.join([
        '#include "gemm_probe.cuh"',
        'using namespace tilemega::sq2;',
        'int main() {',
        '  try {',
        '    Check(cudaSetDevice(0));',
        '    cudaDeviceProp prop{};',
        '    Check(cudaGetDeviceProperties(&prop, 0));',
        '    if (prop.multiProcessorCount != 132) throw std::runtime_error("expected a 132-SM H800");',
        '    Buffers buffers(prop);',
        '    int shared = prop.sharedMemPerBlockOptin;',
        f'    std::puts("{CSV_FIELDS}");',
        *entry_points,
        '    return 0;',
        '  } catch (std::exception const& e) {',
        '    std::fprintf(stderr, "%s\\n", e.what());',
        '    return 2;',
        '  }',
        '}',
        '',
    ])


def main():
    frozen = read(V1)
    directory = CALIB / 'gemm'
    directory.mkdir(parents=True, exist_ok=True)
    all_rows = []
    suites = []
    for model, (catalog, config_file) in MODELS.items():
        config = read(ROOT / config_file)
        domains = read(catalog)['domains']
        shapes = {}
        geometric = {}
        for d in domains:
            family = d['family']
            if family == 'attention':
                continue
            if d['residency'] != 1 and d['geometries']:
                raise ValueError('probe must first support valid multi-CTA residency')
            for cell in d['geometries']:
                g = cell['geometry']
                n = cell['tasks'] * g['tile_n']
                k = int(cell['weight_bytes']) // (2 * n)
                previous = shapes.setdefault(family, (n, k))
                assert previous == (n, k)
                u = 16
                if family == 'gate_up':
                    while g['tile_n'] % (2 * u) and u > 4:
                        u //= 2
                desc = {'model': model, 'family': family, 'n': n, 'k': k, 'geometry': g,
                        'residency': d['residency'], 'tile_k': g['slot_bytes'] // (2 * g['tile_n']),
                        'swiglu_u': u}
                geometric[identity(desc)] = dict(desc, id=identity(desc), sources=['catalog'])
        if model == 'qwen3_1_7b':
            for row in frozen['candidates']:
                c = row['coordinates']
                for family, g in c['gemms'].items():
                    n, k = shapes[family]
                    desc = {'model': model, 'family': family, 'n': n, 'k': k, 'geometry': g,
                            'residency': c['residency'], 'tile_k': g['slot_bytes'] // (2 * g['tile_n']),
                            'swiglu_u': c['swiglu_u'] if family == 'gate_up' else 16}
                    key = identity(desc)
                    current = geometric.setdefault(key, dict(desc, id=key, sources=[]))
                    current['sources'].append(row['original_q_id'])
        for family in OP:
            rows = [r for r in geometric.values() if r['family'] == family]
            assert rows and all(r['residency'] == 1 for r in rows)
            all_rows.extend(rows)
            for part in range(0, len(rows), 12):
                batch = rows[part:part + 12]
                tag = f'{model}-{family}-{part // 12}'
                path = directory / f'{tag}.cu'
                text = source(batch, float(config['rms_norm_eps']))
                if path.exists():
                    assert path.read_text() == text
                else:
                    path.write_text(text)
                suites.append({'id': tag, 'source': str(path),
                               'binary': str(path.with_suffix('')),
                               'geometry_ids': [r['id'] for r in batch]})
    write(directory / 'geometries.json', all_rows)
    write(directory / 'suites.json', suites)
    print(len(all_rows), 'GEMM geometries in', len(suites), 'suites')


if __name__ == '__main__':
    main()
