"""Emit attention probes for each legal split and merge realization."""

from calibration_queue import CALIB
from guarded import ROOT, read, write
from make_gemm_probes import MODELS, V1, identity


def generate(rows, epsilon):
    calls = []
    for row in rows:
        a = row['attention']
        args = [row['hkv'], row['head_dim'], row['queries'], row['norm'],
                a['chunks'], a['block_keys'], a['slots'], a['new_row_work'],
                str(a['merge'] == 'dsmem').lower(), str(a['merge'] == 'stage').lower()]
        calls.append(f'    AttentionCase<{", ".join(map(str,args))}>'
                     f'(buffers,{epsilon:.8e}f,shared,"{row["id"]}");')
    return '\n'.join([
        '#include "attention_probe.cuh"',
        'using namespace tilemega::sq2;',
        'int main() {',
        '  try {',
        '    Check(cudaSetDevice(0));',
        '    cudaDeviceProp prop{};',
        '    Check(cudaGetDeviceProperties(&prop, 0));',
        '    if (prop.multiProcessorCount != 132) throw std::runtime_error("expected a 132-SM H800");',
        '    Buffers buffers(prop);',
        '    int shared = prop.sharedMemPerBlockOptin;',
        '    std::puts("geometry_id,past,repeat,grid,stage_entry_ns,stage_body_ns,merge_ns,'
        'combined_body_ns,kernel_gpu_event_ns,evicted_bytes");',
        *calls,
        '    return 0;',
        '  } catch (std::exception const& e) {',
        '    std::fprintf(stderr, "%s\\n", e.what());',
        '    return 2;',
        '  }',
        '}',
        '',
    ])


def main():
    directory = CALIB / 'attention'
    directory.mkdir(parents=True, exist_ok=True)
    frozen = read(V1)
    all_rows = []
    suites = []
    for model, (catalog, config_file) in MODELS.items():
        config = read(ROOT / config_file)
        hkv = config['num_key_value_heads']
        hq = config['num_attention_heads']
        dim = config.get('head_dim', config['hidden_size'] // hq)
        desc = {'model': model, 'hkv': hkv, 'head_dim': dim, 'queries': hq // hkv,
                'norm': int(model == 'qwen3_1_7b'), 'residency': 1}
        keys = {}

        def add(a, source):
            row = dict(desc, attention=a)
            key = identity(row)
            keys.setdefault(key, dict(row, id=key, sources=[]))['sources'].append(source)

        domains = read(catalog)['domains']
        for d in domains:
            # The residency-2 complete-program stratum has empty GEMM domains.
            if d['family'] != 'attention' or d['residency'] != 1:
                continue
            for a in d['geometries']:
                add(dict(a, merge='la'), 'catalog')
                if a['chunks'] > 1:
                    add(dict(a, merge='stage'), 'catalog')
                if a['chunks'] == 2:
                    add(dict(a, merge='dsmem'), 'catalog')
        if model == 'qwen3_1_7b':
            for row in frozen['candidates']:
                c = row['coordinates']
                assert c['residency'] == 1
                add(c['attention'], row['original_q_id'])
        rows = list(keys.values())
        all_rows.extend(rows)
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            name = f'{model}-attention-{offset // 8}'
            path = directory / f'{name}.cu'
            text = generate(batch, float(config['rms_norm_eps']))
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            suites.append({'id': name, 'source': str(path), 'binary': str(path.with_suffix('')),
                           'geometry_ids': [r['id'] for r in batch]})
    write(directory / 'geometries.json', all_rows)
    write(directory / 'suites.json', suites)
    print(len(all_rows), 'attention geometries in', len(suites), 'suites')


if __name__ == '__main__':
    main()
