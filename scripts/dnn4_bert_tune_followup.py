"""Reconstruct the current-profile boundary and rejected tuning evidence."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trace_samples(cohort_path, library, reference):
    from tilemega.build.identity import sha, verify
    cohort = read(cohort_path)
    trace, native = verify(Path(library)), verify(Path(reference))
    require(cohort['status'] == 'passed' and len(cohort['processes']) == 3
            and len(cohort['traces']) == 3 and not read(cohort['guard'])['code'],
            'boundary needs three accepted children')
    require(trace['defines']['TILEMEGA_TRACE_STAGE'] == '3' and not native['trace']
            and trace['plan']['native_forward'] == native['plan']['native_forward']
            and Path(library + '.cu').read_text().replace('#define TILEMEGA_TRACE_STAGE 3',
                '#define TILEMEGA_TRACE_STAGE 0') == Path(reference + '.cu').read_text(),
            'boundary source differs beyond its trace level')
    samples, cards, pids, inputs = [], set(), set(), set()
    for process, folder in zip(cohort['processes'], cohort['traces']):
        result = read(Path(folder) / 'result.json')
        frames = read(Path(folder) / 'samples.json')
        require(not process['exit_code'] and process['pid'] == result['pid']
                and result['pid'] not in pids
                and Path(folder) == Path(process['out']) / 'trace'
                and result['artifact_id'] == trace['artifact_id']
                and result['reference_sha256'] == sha(Path(reference))
                and result['trace_level'] == 3 and result['warmup'] == 20
                and result['samples'] == 50
                and all(math.isfinite(result[k]) and result[k] > 0 for k in
                    ('start_sm_clock_mhz', 'end_sm_clock_mhz')),
                'boundary child identity, process or protocol changed')
        require(len(frames) == 50 and all([s['stage'] for s in frame] == list(range(63))
                and all(math.isfinite(s['stage_ns']) and s['stage_ns'] > 0 for s in frame)
                for frame in frames), 'boundary stage windows incomplete')
        samples.append(frames)
        pids.add(result['pid'])
        cards.add(result['gpu'])
        inputs.add(result['input_sha256'])
    require(len(cards) == len(inputs) == 1, 'boundary mixed cards or inputs')
    return samples, cards.pop(), inputs.pop()


def model_gate(path, library):
    from dnn4_bert_model_cohort import validate_bert_checks
    from tilemega.build.identity import verify
    model = read(path)
    identity = verify(Path(library))
    require(model['passed'] and model['passes'] == '5/5'
            and len(model['processes']) == 5
            and len({p['pid'] for p in model['processes']}) == 5
            and all(not p['exit_code'] and not read(p['guard'])['code']
                    for p in model['processes']), 'model needs five accepted children')
    require(not identity['trace'] and not any(r['spill'] for r in identity['resources'].values()),
            'native resource gate changed')
    validate_bert_checks([read(p['check']) for p in model['processes']], identity['artifact_id'])
    return identity


def rejected_pair(folder, experiment, model_source, name, pair_name='pair'):
    from dnn4_bert_current import validate_timing_gpu
    identity = model_gate(model_source, experiment['library'])
    closed, pair = read(folder / 'result.json'), experiment['pair']
    require(pair == read(folder / pair_name / 'result.json')
            and read(model_source)['gate'] == experiment['gate']
            and not experiment['accepted'] and not closed['production_admitted']
            and closed.get('final') is None and experiment['minimum_gain'] == .005
            and pair['ratio'] >= 1 - max(.005, pair['threshold']),
            'negative pair changed its independent gain gate')
    validate_timing_gpu(pair, pair['gpu'], 'ABABAB')
    reference = statistics.median(r['median_ms'] for r in pair['measurements'] if r['arm'] == 'A')
    candidate = statistics.median(r['median_ms'] for r in pair['measurements'] if r['arm'] == 'B')
    require(pair['reference_ms'] == reference and pair['candidate_ms'] == candidate
            and pair['ratio'] == candidate / reference, 'pair summary differs from raw medians')
    noise = max((max(r['median_ms'] for r in pair['measurements'] if r['arm'] == arm) -
                 min(r['median_ms'] for r in pair['measurements'] if r['arm'] == arm)) /
                statistics.median(r['median_ms'] for r in pair['measurements'] if r['arm'] == arm)
                for arm in 'AB') * 2
    require(pair['threshold'] == max(.005, noise), 'pair threshold differs from process ranges')
    source = Path(experiment['library'] + '.cu').read_text()
    repeat = re.search(r',Stage_0,Repeat<(6|12),', source)
    return dict(experiment=name, batch=experiment['batch'], gpu=pair['gpu'],
        reference_ms=reference, candidate_ms=candidate, ratio=pair['ratio'],
        gain_pct=100*(1-pair['ratio']), threshold_pct=100*pair['threshold'], accepted=False,
        scope='Rejected private current-profile pair;no P-DNN or production admission',
        correctness='P-CORR5/5 with cross-process output byte hashes',
        artifact_id=identity['artifact_id'], binary_sha256=identity['binary_sha256'],
        source_digest=identity['source']['source_digest'],
        clocks=json.dumps([{k: r[k] for k in ('start_sm_clock_mhz', 'end_sm_clock_mhz')}
                           for r in pair['measurements']], separators=(',', ':')),
        evidence='verified', qualification=pair['qualification'], gate=str(model_source),
        source=str(folder / pair_name / 'result.json'), repeat_layers=12//int(repeat[1]) if repeat else 0)


def collect(base):
    session = base / 'resume_allcards_20261112_v1'
    rows, paragraphs, tune_rows = [], [], []
    boundary = base / 'prefetch_qkv_b8_boundary_v1'
    if not (boundary / 'result.json').exists():
        return rows, paragraphs, tune_rows
    plan, qualified = read(boundary / 'plan.json'), read(boundary / 'result.json')
    tracer = read(boundary / 'trace_model/result.json')['library']
    frames, gpu, input_hash = trace_samples(boundary / 'traces/result.json', tracer, plan['reference']['library'])
    windows = [statistics.median(sum(s['stage_ns'] for s in f) for f in child) for child in frames]
    native = read(plan['reference']['source'])['measurement']
    from dnn4_bert_current import validate_timing_gpu
    validate_timing_gpu(native, native['gpu'], 'BBB')
    relative = statistics.median(windows) / (native['candidate_ms'] * 1e6) - 1
    require(plan['max_relative_difference'] == .05 and plan['native_reference_ms'] == native['candidate_ms']
            and native['input_sha256'] == input_hash and qualified['passed'] and abs(relative) <= .05
            and qualified['process_windows_ns'] == windows and qualified['relative_difference'] == relative,
            'current-profile boundary qualification changed')
    paragraphs.append('Verified current PF1/repeated-QKV-TMA/Rows128 B8 production level3 boundary '
        f'trace passes on GPU{gpu}: three fresh20/50 window medians {windows}ns; median '
        f'{statistics.median(windows):.0f}ns is {100*relative:.6f}% above the predeclared '
        f'P-DNN3 {native["candidate_ms"]:.9f}ms, within5%. Source differs only in trace level; '
        'no native retiming or previous failed qualification is replaced. This qualifies '
        'current-profile window selection, not a P-MODEL pass or performance gain.')
    tuning = session / 'prefetch_qkv_b8_tune_v1'
    if (tuning / 'result.json').exists():
        setup = base / 'prefetch_qkv_b8_tune_v1'
        plan, selected = read(setup / 'plan.json'), read(tuning / 'selection.json')
        require(hashlib.sha256(Path(plan['source']).read_bytes()).hexdigest() == plan['source_sha256']
                and len(plan['groups']) == 6 and len(selected['variants']) == 6,
                'six-candidate input changed')
        expected = copy.deepcopy(read(plan['reference']['library'] + '.plan.json')['native_forward'])
        measured = []
        for rank, result in enumerate(selected['variants']):
            prepared = read(setup / f'variant{rank}/result.json')
            model_gate(result['model_source'], result['library'])
            frames, gpu, _ = trace_samples(result['cohort'], prepared['trace']['library'], result['library'])
            actual = [statistics.median(statistics.median(sum(f[i]['stage_ns'] for i in group['stages'])
                for f in child) for child in frames) for group in plan['groups']]
            require(result['rank'] == rank and result['group_windows_ns'] == actual,
                    'group window is not the median of per-launch joint sums')
            measured.append(actual)
            for group_index, group in enumerate(plan['groups']):
                require(group['share'] >= .02 and len(group['candidates']) == 6
                        and all(abs(a['prediction_ns'] - b['prediction_ns']) /
                            min(a['prediction_ns'], b['prediction_ns']) >= .005
                            for i, a in enumerate(group['candidates']) for b in group['candidates'][i+1:]),
                        'candidate prediction separation changed')
                coordinate = group['candidates'][rank]['coordinate']
                for stage in group['stages']:
                    require(plan['variants'][rank]['stages'][stage] == coordinate,
                            'raw solver coordinate was replaced')
                    tune_rows.append(dict(model='bert', batch=8, gpu=gpu, group=group_index, stage=stage,
                        candidate=rank, group_window_ns=actual[group_index],
                        selected=selected['groups'][group_index]['rank'] == rank,
                        coordinate=json.dumps(coordinate, sort_keys=True, separators=(',', ':')),
                        processes='3/3', samples_per_process=50, warmup=20,
                        evidence='verified P-TRACE3;median of per-launch group sums;private rejected assembly',
                        source=result['cohort']))
        for index, group in enumerate(plan['groups']):
            rank = min(range(6), key=lambda r: (measured[r][index], r))
            require(selected['groups'][index]['rank'] == rank, 'selected group is not its measured minimum')
            for stage in group['stages']:
                expected['stages'][stage] = plan['variants'][rank]['stages'][stage]
        experiment = read(tuning / 'experiment.json')
        require(expected == selected['coordinates'] == read(experiment['library'] + '.plan.json')['native_forward']
                and len(tune_rows) == 282 and read(tuning / 'result.json')['status'] == 'tuned_pair_rejected',
                'assembled coordinates or negative scope changed')
        row = rejected_pair(tuning, experiment, tuning / 'tuned_model/result.json',
                            'Six-way qualified in-situ PF1 B8', 'tuned_pair')
        rows.append(row)
        paragraphs.append('Verified six original raw candidates per group are legal on current B8: '
            'six groups/47 stages,18 fresh trace children; five new native model5 cohorts plus '
            'explicit rank2 model5 reuse, then fresh assembled model5. The282 rows in DNN4_tune.csv '
            'retain actual cards and joint group windows. Selected ranks[2,2,2,0,1,2] change11 '
            'projection stores toREG, terminal FFN2 to64x128REG8 and first QKV to128x192TMA4. '
            f'Independent GPU{row["gpu"]} pair {row["reference_ms"]:.9f}→{row["candidate_ms"]:.9f}ms '
            f'is {100*(row["ratio"]-1):.6f}% slower, inside{row["threshold_pct"]:.6f}% noise. '
            'Retain incumbent under the prospectively stated conservative gain gate; this deviates '
            'from section12 tuned tie preference. No P-SEL, P-DNN, selected mixed SAN or formal '
            'update. U6 interaction remains possible; historical predictions establish candidate '
            'eligibility only. The previously rejected rank2 whole-program pair was never repeated.')
    for n, tn, slots in [('ffn2_m64n128_current_b16_v1', 128, 8),
                          ('ffn2_m64n256_current_b16_v1', 256, 4)]:
        negative = session / n
        if not (negative / 'result.json').exists():
            continue
        experiment = read(negative / 'experiment.json')
        functional = read(negative / 'functional_components.json')
        body = read(functional['body'])
        control = read(functional['control'])
        require(functional['passed'] and len(experiment['unit_gates']) == 5
                and experiment['unit_gates'] == functional['unit_gates']
                and body['passed'] and len({p['pid'] for p in body['processes']}) == 17
                and Counter(p['name'].split('-')[0] for p in body['processes']) ==
                    Counter(unit=5, memcheck=1, initcheck=1, racecheck=5, synccheck=5)
                and len({p['output_hash'] for p in body['processes']}) == 1
                and all(not p['exit_code'] and not read(p['guard'])['code'] for p in body['processes'])
                and control['passed'] and len({p['pid'] for p in control['processes']}) == 5
                and all(not p['exit_code'] and not read(p['guard'])['code'] for p in control['processes']),
                'B16 negative lacks its accepted exact-body17')
        expected = copy.deepcopy(read(experiment['reference']['library'] + '.plan.json')['native_forward'])
        for stage in range(5, 61, 5):
            expected['stages'][stage].update(tile_m=64, tile_n=tn, slots=slots, store='reg')
        require(read(experiment['library'] + '.plan.json')['native_forward'] == expected,
                'B16 isolated FFN2 coordinate changed')
        row = rejected_pair(negative, experiment, negative / 'model/result.json', f'FFN2 M64N{tn} current B16')
        rows.append(row)
        if tn == 256:
            paragraphs.append('Verified isolated current B16 FFN2-only64x256REG4 has zero spill,327936B/R161. '
                'Original full/odd control5/body17 and strict model5 pass; four ordering17 gates are '
                'explicit word reuse. This geometry existed in a compound catalog loser; this distinct '
                'whole program preserves all other current coordinates. '
                f'GPU{row["gpu"]} pair {row["reference_ms"]:.9f}→{row["candidate_ms"]:.9f}ms regresses '
                f'{100*(row["ratio"]-1):.6f}%, noise{row["threshold_pct"]:.6f}%: reject, no P-DNN. '
                'GPU0 exact trace1 FFN2 stage/main/epilogue/store462.496/332.352/151.488/36.064us '
                'versus historical currentGPU2 384.128/311.008/78.848/14.560us; epilogue worsens72.640us '
                'and main21.344us. Nested maxima remain nonadditive/cross-card diagnostics.')
            continue
        paragraphs.append('Verified one current B16 FFN2-only64x128REG8 choice has zero spill,288384B/R160. '
            'Original full/odd FP32 control5/body17 and strict model5 pass; four ordering17 gates are '
            'explicit instruction-word reuse. Initial GPU0 body guard75 is retained and excluded; '
            'new body17 ran on GPU3 after687s preferred-card unavailability. '
            f'GPU{row["gpu"]} pair {row["reference_ms"]:.9f}→{row["candidate_ms"]:.9f}ms regresses '
            f'{100*(row["ratio"]-1):.6f}%, beyond{row["threshold_pct"]:.6f}% noise: reject, no P-DNN. '
            'Exact trace1 FFN2 stage/main/epilogue/store547.008/463.968/55.456/23.552us versus '
            'historical current GPU2 384.128/311.008/78.848/14.560us; epilogue improves23.392us '
            'while main loop worsens152.960us. Cross-card nested maxima are diagnostic, nonadditive. '
            'No production, formal current, fixed target or TensorRT recheck data changed.')
    projection = session / 'projection_reg_pf1_b8_v1'
    if (projection / 'result.json').exists():
        experiment = read(projection / 'experiment.json')
        expected = copy.deepcopy(read(experiment['reference']['library'] + '.plan.json')['native_forward'])
        for stage in range(8, 59, 5):
            expected['stages'][stage]['store'] = 'reg'
        require(read(experiment['library'] + '.plan.json')['native_forward'] == expected
                and not experiment['unit_gates'], 'private projection scope changed')
        row = rejected_pair(projection, experiment, projection / 'model_resume_v1/result.json',
                            'Repeated projection REG current PF1 B8')
        rows.append(row)
        paragraphs.append('Verified qualified-window projection choice isolated on PF1 B8 changes only11 '
            'repeated projection stores TMA→REG, retaining128x64/S1/TK64/Slots8 and other52 stages. '
            'Native262144B/R168,zero spill; all1380 bindings and strict model5 pass. '
            f'GPU{row["gpu"]} pair {row["reference_ms"]:.9f}→{row["candidate_ms"]:.9f}ms is '
            f'{100*(row["ratio"]-1):.6f}% slower,noise{row["threshold_pct"]:.6f}%: reject. '
            'No selected body/order SAN, P-DNN or formal change. The generic trace after smoke>3% '
            'retains rejected preflight75 then accepted attempt2; this negative does not '
            'retroactively change original six-way window data or replay that pair.')
    return rows, paragraphs, tune_rows
