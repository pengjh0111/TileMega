"""Verify the isolated nonwriter tile-policy trial without admitting it."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(base):
    from dnn4_bert_tune_followup import rejected_pair
    from tilemega.build.identity import verify
    folder = base / 'resume_allcards_20261112_v1/terminal_n96_current_b8_v2'
    if not (folder / 'result.json').exists():
        return [], []
    if read(folder / 'result.json')['status'] != 'pair_rejected':
        return [], []
    setup, policy = base / 'terminal_n96_current_b8_v2', base / 'terminal_n96_writer_v1'
    prepared, experiment = read(setup / 'result.json'), read(folder / 'experiment.json')
    candidate, current = verify(Path(experiment['library'])), verify(Path(experiment['reference']['library']))
    expected = json.loads(json.dumps(current['plan']['native_forward']))
    expected['stages'][60].update(tile_m=64, tile_n=96, slots=8, store='reg')
    require(candidate['defines'] == current['defines'] and candidate['plan']['native_forward'] == expected
            and candidate['plan']['buffers'] == current['plan']['buffers']
            and prepared['model']['artifact_id'] == candidate['artifact_id'],
            'terminal trial changes more than its one geometry')
    prototype = read(policy / 'prototype_sources.json')
    archived = {path: policy / ('prototype_' + Path(path).name) for path in prototype}
    for name in ('dnn4_f3_reader_geometry_test.cpp', 'dnn4_f3_reader_candidates_test.cpp'):
        archived[str(Path(candidate['source']['root']) / 'test/unit' / name)] = policy / name
    require(all(sha(archived[path]) == digest for path, digest in prototype.items()),
            'private reader-policy source changed')
    for name, assertions in (('reader_geometry_result.json', 9), ('reader_candidates_result.json', 6)):
        test = read(policy / name)
        require(test['passed'] and test['assertions'] == assertions and test['assertions_enabled']
                and '-UNDEBUG' in test['command'] and '-DNDEBUG' not in test['command']
                and sha(test['binary']) == test['binary_sha256']
                and sha(archived[test['source']]) == test['source_sha256']
                and test['header_sha256'] == prototype[str(Path(candidate['source']['root']) /
                    'include/tilemega/Solver/ForwardGeometry.h')], 'CPU writer/read policy gate changed')
    controls = read(setup / 'default_source_controls.json')
    require(controls['passed'] and controls['device_headers_identical']
            and [row['batch'] for row in controls['rows']] == [1, 2, 4, 8, 16]
            and all(row['byte_identical'] and sha(row['source']) == row['source_sha256']
                    and Path(row['source']).read_bytes() == Path(row['control_source']).read_bytes()
                    for row in controls['rows']) and read(setup / 'writer_rejection.json')['passed'],
            'default-source or actual writer rejection gate changed')
    binding = read(setup / 'binding_contract.json')
    require(binding['passed'] and binding['checks'] == 1380
            and binding['optional_ffn2_output_maps'][-1] is None
            and all(value is not None for value in binding['optional_ffn2_output_maps'][:-1])
            and sha(binding['source']) == binding['source_sha256'], 'terminal optional map contract changed')
    builds = read(base / 'terminal_n96_current_b8_units_v1/result.json')['rows']
    gate_paths = []
    for name, build_path, count in (('control', builds[0]['build'], 5), ('body', builds[1]['build'], 17),
                                  ('mapping', base / 'terminal_n96_current_b8_map_v1/build.json', 17)):
        gate, build = read(folder / name / 'result.json'), read(build_path)
        require(gate['passed'] and len(gate['processes']) == count
                and len({p['pid'] for p in gate['processes']}) == count
                and len({p['output_hash'] for p in gate['processes']}) == 1
                and all(not p['exit_code'] and not read(p['guard'])['code']
                        and p['binary_sha256'] == build['binary_sha256'] for p in gate['processes'])
                and sha(build['binary']) == build['binary_sha256']
                and all(sha(archived.get(path, Path(path))) == digest for path, digest in build['inputs'].items()),
                'terminal functional gate or compile input changed')
        if count == 17:
            require(Counter(p['name'].split('-')[0] for p in gate['processes']) ==
                    Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5),
                    'terminal sanitizer process counts changed')
            gate_paths.append(str(folder / name / 'result.json'))
    orders = read(base / 'terminal_n96_current_b8_order_v1/result.json')['rows']
    require(len(orders) == 4, 'terminal ordering evidence incomplete')
    def words(path):
        return re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',
            subprocess.check_output(['cuobjdump', '-sass', str(path)], text=True))
    for row in orders:
        build, gate = read(row['build']), read(row['reference_gate'])
        require(row['whole_words_equal'] and build['source_identical']
                and len(words(build['binary'])) == row['encoded_words']
                and words(build['binary']) == words(build['reference'])
                and gate['passed'] and len(gate['processes']) == 17
                and all(not p['exit_code'] and not read(p['guard'])['code']
                        and p['binary_sha256'] == sha(build['reference']) for p in gate['processes'])
                and all(sha(archived.get(path, Path(path))) == digest for path, digest in build['inputs'].items()),
                'terminal visibility suite lost its exact historical reuse proof')
        gate_paths.append(row['reference_gate'])
    require(experiment['unit_gates'] == gate_paths, 'terminal model bypassed a functional gate')
    row = rejected_pair(folder, experiment, folder / 'model/result.json', 'Terminal M64N96 REG on current B8')
    paragraph = ('Verified private reader-policy correction distinguishes statistics consumers from producers; '
        'only prospective nonwriter64x96 opens and all old spill/producer restrictions remain. '
        'Nine feasibility and six actual solver CPU assertions pass,writer-stage55 rejects the same tile, '
        'and all5 existing coordinate CUs remain byte-identical. Only B8 terminal stage60 changes '
        '128x64TMA8→64x96REG8,original NT12/grain64 and other62 coordinates/current profile fixed. '
        'Fresh control5/body17/actual1380-map17/model5 pass;four visibility suites explicitly reuse '
        'identical sources/instructions and historical17. Native264064B/R166,zero spill. '
        f'GPU{row["gpu"]} pair {row["reference_ms"]:.9f}→{row["candidate_ms"]:.9f}ms, '
        f'ratio{row["ratio"]:.9f},noise{row["threshold_pct"]:.6f}%:reject,no P-DNN/current update. '
        'Initial false-exclusion CPU failure retained;private policy is not promoted without '
        'end-to-end gain. This single legal-section9 follow-up lies outside the old top-six shortlist; '
        'it establishes no P-TUNE/P-MODEL success.')
    return [row], [paragraph]
