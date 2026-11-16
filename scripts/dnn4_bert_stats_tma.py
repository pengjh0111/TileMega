"""Reconstruct rejected private row-statistics TMA trials from raw receipts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def words(path):
    return re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',
        subprocess.check_output(['cuobjdump', '-sass', str(path)], text=True))


def unit(gate_path, build_path, count, archived):
    gate, build = read(gate_path), read(build_path)
    require(gate['passed'] and len(gate['processes']) == count
            and len({p['pid'] for p in gate['processes']}) == count
            and len({p['output_hash'] for p in gate['processes']}) == 1
            and all(not p['exit_code'] and not read(p['guard'])['code']
                    and p['binary_sha256'] == build['binary_sha256'] for p in gate['processes'])
            and sha(build['binary']) == build['binary_sha256'], 'functional process or binary changed')
    if count == 17:
        require(Counter(p['name'].split('-')[0] for p in gate['processes']) ==
                Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5),
                'functional gate lacks the original sanitizer repetitions')
    require(all(sha(archived.get(path, Path(path))) == digest for path, digest in build['inputs'].items()),
            'functional compile input changed')
    return gate


def collect(base):
    from dnn4_bert_tune_followup import rejected_pair
    from tilemega.build.identity import verify
    rows, paragraphs = [], []
    original = base / 'stats_tma_current_b16_v1'
    if not (original / 'source.json').exists():
        return rows, paragraphs
    prototype = read(original / 'source.json')['files']
    archived = {path: original / ('prototype_' + Path(path).name) for path in prototype}
    require(all(sha(archived[path]) == digest for path, digest in prototype.items()),
            'archived transfer prototype changed')
    for batch in (16, 8):
        name = f'stats_tma_current_b{batch}_v1'
        folder = base / 'resume_allcards_20261112_v1' / name
        if not (folder / 'result.json').exists():
            continue
        closed = read(folder / 'result.json')
        if closed['status'] != 'pair_rejected':
            continue
        prepared, experiment = read(base / name / 'result.json'), read(folder / 'experiment.json')
        identity, reference = verify(Path(experiment['library'])), verify(Path(experiment['reference']['library']))
        flags = dict(identity['defines'])
        require(flags.pop('TILEMEGA_F3_STATS_TMA', None) == '1' and flags == reference['defines']
                and identity['plan']['native_forward'] == reference['plan']['native_forward']
                and identity['plan']['buffers'] == reference['plan']['buffers']
                and prepared['model']['artifact_id'] == identity['artifact_id']
                and prepared['reference'] == experiment['reference'],
                'transfer trial changes its fixed current coordinates, profile or buffers')
        snapshot = identity['source']
        require(all(snapshot['source_files'][str(Path(path).relative_to(snapshot['root']))] == digest
                    for path, digest in prototype.items()), 'compiled prototype differs from the archived headers')
        defaults = read(base / name / 'default_proofs.json')['rows']
        require([p['batch'] for p in defaults] == [1, 2, 4, 8, 16]
                and all(p['passed'] and sha(p['library']) == p['binary_sha256']
                        and sha(p['reference']) == p['reference_sha256']
                        and len(words(p['library'])) == p['encoded_words']
                        and words(p['library']) == words(p['reference']) for p in defaults),
                'default five instruction controls changed')
        binding = read(base / name / 'binding_contract.json')
        require(binding['passed'] and binding['checks'] == 1380 and binding['stats_map_count'] == 23
                and len(binding['stats_users']) == 46 and binding['source'] == experiment['library'] + '.cu'
                and sha(binding['source']) == binding['source_sha256'], 'actual statistics-map contract changed')
        builds = read(base / f'stats_tma_current_b{batch}_units_v1/result.json')['rows']
        control = unit(folder / 'control/result.json', builds[0]['build'], 5, archived)
        body = unit(folder / 'body/result.json', builds[1]['build'], 17, archived)
        marker = base / f'stats_tma_current_b{batch}_map_v1'
        mapping = unit(folder / 'mapping/result.json', marker / 'build.json', 17, archived)
        require(control['output_hash'] == body['output_hash'] and read(marker / 'plan.json')['checks'] == 1380
                and read(marker / 'plan.json')['real_generated_source_sha256'] == sha(experiment['library'] + '.cu'),
                'raw body bits or actual mapping fixture changed')
        order = read(base / f'stats_tma_current_b{batch}_order_v1/result.json')['rows']
        require(len(order) == 4 and all(row['whole_words_equal'] for row in order),
                'ordering suites lack their declared exact reuse proof')
        for row in order:
            build = read(row['build'])
            require(words(build['binary']) == words(build['reference'])
                    and len(words(build['binary'])) == row['encoded_words']
                    and all(sha(archived.get(path, Path(path))) == digest
                            for path, digest in build['inputs'].items()), 'ordering instruction or source changed')
            gate = read(row['reference_gate'])
            require(gate['passed'] and len(gate['processes']) == 17
                    and all(not p['exit_code'] and not read(p['guard'])['code']
                            and p['binary_sha256'] == sha(build['reference']) for p in gate['processes']),
                    'historical ordering process was not executed')
        expected_gates = [str(folder / 'body/result.json'), str(folder / 'mapping/result.json'),
                          *[row['reference_gate'] for row in order]]
        def signature(value):
            return [(case['epoch'], case['mode'], [(output['name'], output['output_sha256'])
                    for output in case['outputs']]) for case in value['cases']]
        require(experiment['unit_gates'] == expected_gates
                and read(folder / 'model_byte_comparison.json')['bitwise_equal']
                and signature(read(folder / 'model/result.json')) == signature(read(experiment['reference']['gate'])),
                'trial bypassed a component or incumbent-byte gate')
        row = rejected_pair(folder, experiment, folder / 'model/result.json',
                            f'UInt64 statistics TMA on current B{batch}')
        rows.append(row)
        paragraphs.append(f'Verified private B{batch} replaces twelve FP32 statistics Bulk reads with one UInt64 '
            '2D TMA;original64-column grain,ascending arithmetic,coordinates and all other current features fixed. '
            'All5 default instruction controls pass;actual1380 binding fields and23 maps/46 readers checked. '
            'Fresh control5/body17/map17 and strict model5 pass;body and model outputs equal incumbents. '
            'Four original visibility suites explicitly reuse historical17 only after full instruction equality; '
            'their optional-map Bind stub alone is extended,oracle unchanged. '
            f'GPU{row["gpu"]} pair {row["reference_ms"]:.9f}→{row["candidate_ms"]:.9f}ms, '
            f'ratio{row["ratio"]:.9f},noise{row["threshold_pct"]:.6f}%:reject,no P-DNN or current update. '
            'Code size and component correctness establish no end-to-end speed gain. '
            'This is an explicit autonomous extension of section5.2,not literal twelve-Bulk protocol.')
    return rows, paragraphs
