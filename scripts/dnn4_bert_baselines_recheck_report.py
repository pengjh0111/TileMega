"""Verify and publish the explicitly requested unfrozen BERT baseline study."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from dnn1_baselines import measurement_clocks, receipt_problem, strict_correctness
from tilemega.dnn.baselines import engine_cache_key


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect_cell(folder):
    result, plan = read(folder / 'result.json'), read(folder / 'plan.json')
    system = plan['arm']['system']
    expected_roles = (['correctness'] + ['measurement'] * 3 + ['correctness']
                      if system == 'torchcompile' else ['measurement'] * 3 + ['correctness'] * 2)
    require(result['passed'] and result['correctness_passes'] == '5/5'
            and result['model'] == 'bert' and result['batch'] == plan['batch']
            and result['arm'] == plan['arm']['arm'] and not read(result['guard'])['code']
            and plan['roles'] == expected_roles and len(result['processes']) == 5,
            'baseline needs the declared five accepted fresh processes')
    specification = read(result['specification'])
    require(sha(result['specification']) == result['specification_sha256']
            and all(specification.get(key) == value for key, value in plan.items())
            and specification['gpu'] == result['gpu']
            and all(sha(path) == digest for path, digest in specification['frozen'].items()),
            'baseline specification or compile inputs changed')
    pids, engines, inputs, versions = set(), set(), set(), set()
    output_hashes, reference_hashes, checked_engines = set(), set(), {}
    device = read(ROOT / 'configs/dnn/dnn3_h800_devices.json')['devices'][str(result['gpu'])]
    medians, p90s, clocks, cosines = [], [], [], []
    for process, role in zip(result['processes'], expected_roles):
        value = read(process['measurement'])
        require(sha(process['measurement']) == process['measurement_sha256']
                and sha(process['log']) == process['log_sha256']
                and process['role'] == value['process_role'] == role
                and process['pid'] == value['pid'] and value['pid'] not in pids
                and not receipt_problem(specification, plan['arm'], role, value)
                and value['correctness_passed'] and strict_correctness('bert', value['correctness'])[0],
                'baseline raw identity, protocol or FP32 correctness changed')
        pids.add(value['pid'])
        config = value['configuration']
        require(config['input_seed'] == 20261007 and config['inputs_sha256'] == plan['inputs_sha256']
                and value['exported_program_sha256'] == plan['exported_program_sha256']
                and value['source_checkpoint_sha256'] == plan['source_checkpoint_sha256'],
                'baseline weights, export or input identity changed')
        if system == 'torchcompile':
            require(config['compile_mode'] == 'max-autotune' and config['fullgraph'] is True
                    and config['freezing'] is False and config['freezing_requested'] is False
                    and '--freezing' not in process['command'], 'freezing entered the unfrozen arm')
            require(process['command'][:3] == ['taskset', '-c',
                    plan['compile_cpu_set'] if len(pids) == 1 else plan['cpu_set']],
                    'compilation and measurement CPU placement changed')
        else:
            cache = config['engine_cache']
            identity = cache['identity']
            metadata = read(cache['identity_file'])
            require(identity['static_batch'] == plan['batch'] and identity['precision'] == 'bf16'
                    and identity['builder_flags']['builder_optimization_level'] == 5
                    and identity['builder_flags']['precision_flag'] == 'BF16'
                    and identity['gpu_uuid'] == device['uuid']
                    and identity['tensorrt_version'] == value['versions']['tensorrt']
                    and identity['exported_program_sha256'] == plan['exported_program_sha256']
                    and identity['source_checkpoint_sha256'] == plan['source_checkpoint_sha256']
                    and identity['onnx_sha256'] == config['onnx_sha256']
                    and identity['builder_flags']['tactic_sources'] == 'official default'
                    and identity['builder_flags']['workspace_limit'] == 'official default'
                    and identity['builder_flags']['remaining_builder_configuration'] == 'official defaults'
                    and cache['key'] == metadata['key'] == engine_cache_key(identity)
                    and metadata['identity'] == identity
                    and sha(cache['identity_file']) == cache['identity_sha256'],
                    'TensorRT static batch or original engine identity changed')
            engine = cache['engine_file']
            if engine not in checked_engines:
                checked_engines[engine] = sha(engine)
            require(checked_engines[engine] == metadata['serialized_engine_sha256'] == config['engine_sha256'],
                    'TensorRT serialized engine bytes changed')
            engines.add(config['engine_sha256'])
        inputs.add(config['inputs_sha256'])
        output_hashes.add(config['actual_outputs_sha256'])
        reference_hashes.add(config['reference_outputs_sha256'])
        versions.add(json.dumps(value['versions'], sort_keys=True))
        cosines.extend(row['cosine_min_vs_fp32'] for row in value['correctness'])
        if role == 'measurement':
            samples = value['samples_ms']
            require(len(samples) == 200 and all(math.isfinite(x) and x > 0 for x in samples)
                    and value['median_ms'] == statistics.median(samples)
                    and value['p90_ms'] == sorted(samples)[179], 'CUDA event statistics changed')
            medians.append(value['median_ms'])
            p90s.append(value['p90_ms'])
            clocks.append(measurement_clocks(value, result['gpu']))
        else:
            require(not value['samples_ms'] and value['timing_status'] == 'not_measured',
                    'correctness-only process contributed timing')
    require(len(inputs) == len(versions) == 1 and len(medians) == 3
            and (system != 'tensorrt' or len(engines) == 1), 'baseline mixed inputs, versions or engines')
    median, p90 = statistics.median(medians), statistics.median(p90s)
    process_range = max(medians) - min(medians)
    require(result['median_ms'] == median and result['p90_ms'] == p90
            and result['process_range_ms'] == process_range and result['process_medians_ms'] == medians
            and result['output_hash_stable'] == (len(output_hashes) == 1)
            and result['reference_hash_stable'] == (len(reference_hashes) == 1),
            'baseline summary differs from its three process medians')
    return dict(model='bert', batch=plan['batch'], system=system, arm=plan['arm']['arm'],
        precision='bf16', gpu=result['gpu'], gpu_uuid=device['uuid'],
        freezing=False if system == 'torchcompile' else '',
        compile_mode='max-autotune' if system == 'torchcompile' else '',
        median_ms=median, p90_ms=p90, process_range_ms=process_range,
        process_medians_ms=json.dumps(medians, separators=(',', ':')),
        previous_same_arm_ms=plan['frozen_ms'], ratio_to_previous=median / plan['frozen_ms'],
        correctness_passes='5/5', minimum_cosine_vs_fp32=min(cosines),
        output_hash_stable=result['output_hash_stable'], reference_hash_stable=result['reference_hash_stable'],
        inputs_sha256=plan['inputs_sha256'], exported_program_sha256=plan['exported_program_sha256'],
        source_checkpoint_sha256=plan['source_checkpoint_sha256'],
        engine_sha256=next(iter(engines), ''), versions=next(iter(versions)),
        clocks=json.dumps(clocks, separators=(',', ':')), evidence='verified',
        qualification=result['qualification'], source=str(folder / 'result.json'))


def collect(base):
    require(all(sha(path) == digest for path, digest in read(base / 'protected_results.json').items()),
            'historical results changed during the baseline study')
    rows = [collect_cell(base / system / f'b{batch}')
            for system in ('tensorrt', 'torchcompile') for batch in (1, 2, 4, 8, 16)]
    require(Counter((row['system'], row['batch']) for row in rows) ==
            Counter((system, batch) for system in ('tensorrt', 'torchcompile') for batch in (1, 2, 4, 8, 16)),
            'baseline study lacks its ten cells')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.base.resolve())
    args.out.mkdir(exist_ok=True, parents=True)
    with (args.out / 'DNN4_BERT_baselines_unfrozen_recheck.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# BERT BF16 baseline recheck', '',
        'Verified BERT-base, seq128, H800. Each cell uses three fresh50/200 CUDA-event processes '
        'and two additional FP32 correctness processes;5/5 pass. Compilation is excluded. '
        'torch.compile uses max-autotune/fullgraph with requested and actual freezing=False. '
        'P90 is the median of three process P90s.', '',
        '| Batch | TensorRT ms | torch.compile, no freezing ms | TRT GPU | TC GPU |',
        '| --- | --- | --- | --- | --- |']
    for batch in (1, 2, 4, 8, 16):
        a = next(row for row in rows if row['system'] == 'tensorrt' and row['batch'] == batch)
        b = next(row for row in rows if row['system'] == 'torchcompile' and row['batch'] == batch)
        lines.append(f'| {batch} | {a["median_ms"]:.6f} | {b["median_ms"]:.6f} | {a["gpu"]} | {b["gpu"]} |')
    lines += ['', 'Verified source receipts, all200 individual events, five FP32 correctness checks, '
        'input/checkpoint identities and actual freezing=False are reconstructed by '
        'scripts/dnn4_bert_baselines_recheck_report.py. The CSV retains P90, process range, '
        'versions, boundary clocks and raw sources. Baseline hash stability is diagnostic. '
        'Random-weight exports have no dataset accuracy certification; GPU PID attribution '
        'uses the existing container-namespace qualification.', '',
        'Historical TileMega data, the earlier TRT recheck and fixed DNN4 targets are preserved. '
        'These fresh baseline measurements do not replace the frozen TC+freezing targets.']
    (args.out / 'DNN4_BERT_baselines_unfrozen_recheck.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(passed=True, cells=len(rows), out=str(args.out))))


if __name__ == '__main__':
    main()
