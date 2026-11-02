"""Render the qualified pair-lane TP results against the frozen baselines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from baseline_results import frozen_baselines
from tilemega.tp.process_cohorts import checked, validate_run


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/mx7/sm90_comparison'))
    args = parser.parse_args()
    root, results = args.root, Path('docs/results')
    costs = Path('configs/targets/xcore1000_mx7_tp.json')
    cost_sha = hashlib.sha256(costs.read_bytes()).hexdigest()
    assert read(costs.with_suffix('.qualification.json'))['solver_eligible']
    initial, final = {}, {}
    for row in csv.DictReader((results / 'MX7_tp_final_c550.csv').open()):
        key = (int(row['tp']), int(row['batch']))
        initial.setdefault(key, row)
        final[key] = row
    assert len(final) == 10 and all(r['cost_table_sha256'] == cost_sha for r in final.values())
    baselines = frozen_baselines(Path('configs/bench/mx7_frozen_baselines.json'))
    assert len(baselines) == 30 and all(r['status'] == 'pass' for r in baselines.values())
    selected = [read(root / f'selection/tp{t}/B{b}/results.json') for t, b in sorted(final)]
    for data in selected:
        assert len(data['candidates']) == 4 and len(data['final_correctness']['C2']) == 5
        assert read(checked(data['final_correctness']['C1']))['pass']
        for candidate in data['candidates']:
            assert read(checked(candidate['correctness']['C1']))['pass']
            assert len(candidate['correctness']['C2']) == 2
        assert len({read(checked(x))['PID'] for x in data['final_correctness']['C2']}) == 5
    delay = read(root / 'selected_mapping_qualification/QUALIFIED.json')
    assert len(delay['cohorts']) == 4
    for item in delay['cohorts']:
        proof = read(checked(item))
        assert len({read(checked(x['check']))['PID'] for x in proof['runs']}) == 5
        for run in proof['runs']:
            validate_run(run['run']['manifest'])
            assert read(checked(run['check']))['pass']
    trace = {b: read(root / f'optimized_trace/final_trace/B{b}/trace/trace.json') for b in (1, 16)}
    usage = read(root / 'final_gpu_usage.json')
    start = '6cab45b9604e306fef1916863fe16b09d895c32f'
    audit_start = 'a7c4a7968a787a5bf5a36fbbfc6532ae1de7a176'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    count = subprocess.check_output(['git', 'rev-list', '--count', start + '..' + head], text=True).strip()
    lines = [
        '# MX-7: Qwen3-1.7B TP2 / TP4 on C550', '',
        'Evidence: **verified** = executed or source-checked here; **stated** = specification/remote report; **inferred** = reasoning.',
        'BF16 greedy, ISL/OSL=64/1024. Raw successes, failures, generated sources and binaries remain under `runs/mx7/`.', '',
        '## Scope and identity', '',
        f'- **verified** `tilemega-metax`, MX-7 start `{start}`, SM90 audit start `{audit_start}`.',
        f'- **verified** aggregation HEAD `{head}`;{count} commits since MX-7 start; publication commit follows.',
        '- **verified** prompt SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.',
        '- **verified** resume1 SHA256 `f67014253a3cca7234429ff3671e293c0104fb04b68fde1bb9dc52b518c0b284`.',
        '- **stated** later user authorization: all GPUs; SDK-owned ExternalStream adapter; baselines measured once; compare SM90 and optimize TP bodies.',
        '- **verified** MoE protected files/final CSV remain unchanged; eight frozen TP1/MoE plans retain identical generated CU and function instruction counts.', '',
        '## Environment and execution (10-line onboarding summary)', '',
        '1. **verified** MACA3.8.2.6, cucc/mxcc1.0.0(890c70f3aa), driver3.8.1.',
        '2. **verified** Torch2.10.0+metax3.8.2.2, Transformers5.12.0; full versions/topology in `runs/mx7/sync/onboarding.md`.',
        '3. **verified** vLLM0.25.0+empty; SGLang0.5.14+detached.c432db2fb.maca3.8.2.122.',
        '4. **verified** NCCL/Gloo available; NCCL used only by torch.compile baseline; TileMega uses MACA P2P.',
        '5. **verified** C0 selected LL8, fine allocation, volatile AS1 scalar64 store and `__ldcg`64 load; no runtime switches.',
        '6. **verified** original pair/all-to-all P1024/P262144: each5/5 fresh processes ×2000epochs ×104CTAs, no tears/timeouts.',
        '7. **verified** final TP2 group{1,3}; final TP4 group{1,2,3,5}; guard admission/locks unchanged.',
        '8. **inferred** each rank: qkv→attention→o+AR→gate_up→down+AR per layer, then head LL argmax exchange.',
        '9. **verified** ascending-rank FP32 sum of BF16 partials, BF16 sum then residual; epoch=past+1; two asynchronous range launches/rank.',
        '10. **verified** prompt mode starts at past0, exchanges every step, suppresses early token writes; one SDK nonblocking stream/rank.', '',
        '## SM90 comparison and accepted optimization', '',
        '- **verified** current complete SM90 results/source: `origin/tilemega@213a3884a4e14629ca99f63755d501c65a7d8f53`; no merge or H800 remeasurement.',
        '- **verified** sharding, persistent launch pipeline, AR placement and BF16 arithmetic match. No missing collective or per-token host loop.',
        '- **verified** the original MX-7§6.4 mapping gave each lane four LL8 packets at32-byte inter-lane strides. SM90 assigns one adjacent column pair/lane.',
        '- **verified** shared pair-lane mapping preserves wire layout, rank order, rounding, geometry and synchronization; production peer primitives stay unchanged.',
        '- **verified** restoring two-pass sends or replacing only peer primitives did not improve performance; those candidates stay off.',
        '- **verified** same-geometry TP4 B16 A/B/A/B control:12.526515→3.081392ms(4.065×), both arms2 fresh warm1/timed2 processes.',
        '  This control is separate from the solver-selected final matrix below. Audit and microbenchmarks: `MX7_tp_sm90_comparison_c550.md`.',
        '- **verified** all1840 single-rank stage rows are reused;90 communication constants are newly measured,3 processes/key plus competitive-noise supplements.',
        f'- **verified** eligible cost SHA256 `{cost_sha}`. All10 cells rerun k-best/top4 and the1% E2E selection rule.', '',
        '## Frozen baselines and final performance', '',
        '**verified** all30 external cells pass: vLLM/SGLang/torch.compile × TP2/TP4 × B1/2/4/8/16.',
        '**stated** original baseline samples are reused across GPU groups; no external baseline is repeated for this optimization.',
        '**verified** baseline TPOT=(E2E_N1024−E2E_N1)/1023; TileMega TPOT=(wall E2E−CUDA TTFT)/1023.',
        'Both use two fresh processes/cell, warm1+timed2. TTFT includes prompt mode; tokens/s counts B×1024 outputs.', '',
        '| TP | B | vLLM ms | SGLang ms | torch.compile ms | TileMega ms | fastest/TM | old TM ms | old/new |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, row in sorted(final.items()):
        t, b = key
        values = [float(baselines[t, b, s]['tpot_ms']) for s in ('vllm', 'sglang', 'torchcompile')]
        new, old = float(row['tpot_ms']), float(initial[key]['tpot_ms'])
        lines.append(f'| {t} | {b} | ' + ' | '.join(f'{v:.3f}' for v in values + [new, min(values)/new, old, old/new]) + ' |')
    lines += ['', '| TP | B | TTFT ms | E2E ms | tokens/s | TP1/TM | scaling efficiency |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for (t, b), row in sorted(final.items()):
        new = float(row['tpot_ms'])
        values = [float(row[k]) for k in ('ttft_ms', 'e2e_ms', 'tokens_per_s')]
        lines.append(f'| {t} | {b} | ' + ' | '.join(f'{v:.3f}' for v in values) +
                     f' | {float(row["tp1_frozen_tpot_ms"])/new:.3f} | {100*float(row["scaling_efficiency"]):.1f}% |')
    lines += ['', 'Efficiency=TP1_frozen_TPOT/(TP×TP_TPOT). Each engine ratio, process result, binary/pin/cost SHA is retained in `MX7_tp_final_c550.csv`.',
        'Old published pins are preserved under `configs/pins/mx7/history/vector8/`; old CSV rows and original artifacts are retained.', '',
        '## Selected geometries', '',
        'Coordinates below are TN/WN/D/Pw/partition; TR=TileRange, SK=StreamK. Attention remains V2.', '',
        '| TP | B | qkv | o | gate_up | down | head | attn pf/b_min |',
        '|---:|---:|---|---|---|---|---|---|']
    for (t, b), row in sorted(final.items()):
        cfg = read(row['pin'])['mx_decode_v2']
        values = []
        for stage in ('qkv', 'o', 'gate_up', 'down', 'head'):
            c = cfg[stage]
            values.append('/'.join(str(c[k]) for k in ('tn', 'wn', 'd', 'prefetch')) + '/' + ('TR' if c['part'] == 'kTileRange' else 'SK'))
        lines.append(f'| {t} | {b} | ' + ' | '.join(values) + f' | {cfg["attn_prefetch"]}/{cfg["b_min"]} |')
    lines += ['', '## Correctness, synchronization and resources', '',
        '- **verified** pair-lane AR units: TP2/TP4 × B1/B16, each5/5 fresh processes; nine legal geometries/process against ordered-BF16 reference.',
        '- **verified** original head exchange units pass5/5 for rank-local/cross-rank/all-equal ties; head code unchanged by this optimization.',
        '- **verified** all40 candidates pass512-position HF C-1 and2 fresh1024-token C-2 processes.',
        '- **verified** all10 selected rebuilds pass512-position HF C-1 and5 fresh1024-token C-2 processes; all ranks agree in every run.',
        '- **verified** selected pair-lane injected-delay checks: TP2/TP4 × B1/B16, each5/5 new1024-token processes exactly match ordinary selected outputs.',
        '- **verified** MTreg≤255, stack0, static0, all-rank actual resident1, dynamic LDS≤64KiB. Rejections remain in the resource ledger.',
        '- **verified** eight TP1/MoE frozen generated sources and per-function instruction counts pass8/8 after the shared-body edit.',
        '- **verified**50 targeted CPU tests pass after the SDK lifetime correction; detailed raw evidence remains under `runs/mx7/sm90_comparison/`.', '',
        '## Final TP4 stage trace (rank0, µs)', '',
        '**verified** one diagnostic process/B;1023 decode steps ×104CTAs/rank. o/down include AR; head includes exchange.',
        'Trace durations use max CTA release minus previous-stage release; these are diagnostic, not E2E acceptance measurements.', '',
        '| B | layer | qkv | attention | o+AR | gate_up | down+AR |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for b, data in trace.items():
        assert len(data['stages']) == 141 and data['decode_samples'] == 1023
        for layer in range(28):
            values = [x['rank0_median_ns']/1000 for x in data['stages'] if x['layer'] == layer]
            assert len(values) == 5
            lines.append(f'| {b} | {layer} | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
        h = next(x for x in data['stages'] if x['stage'] == 'head')
        lines.append(f'| {b} | head | {h["rank0_median_ns"]/1000:.2f} | — | — | — | — |')
        delta = {s: max(x['max_rank_difference_ns'] for x in data['stages'] if x['stage'] == s)/1000 for s in ('o', 'down')}
        lines.append(f'\n**verified** B{b} maximum inter-rank duration difference: o={delta["o"]:.3f}µs, down={delta["down"]:.3f}µs.\n')
    pred = [float(x['pred_over_meas']) for x in final.values()]
    lines += ['', '## Autonomous handling and retained failures', '',
        '- **verified, B2/B4** torch.compile: isolated accelerate1.15.0 plus compiled_autograd=True fixes HF TP gradient-hook compilation; no eager substitution.',
        '- **verified, B1** original AR builds spilled at256MTreg; approved per-item send/receive and constant peer-pointer selection fit resources.',
        '- **verified, B4** unavailable GPUs and external interference required group changes and bounded fresh-process retries; original failures/guard75 records remain.',
        '- **verified, B4** original calibration B8/TN128 LDS underallocation was repaired; old illegal-access evidence remains; no failed sample is promoted.',
        '- **verified, B4** low disk pressure is managed with immutable-input hardlinks and lossless rank-wise trace compression.',
        '- **verified, B4** initial SDK stream full-model check produced exact tokens but exited−6 during pinned-storage cleanup after stream destruction.',
        '  The failed exit is retained. Owned pageable return storage and GPU/pinned release before stream destruction repair the lifetime order;',
        '  four-card retained-pinned-lifetime and fresh full-model delay cohorts each pass5/5 after the fix.',
        '- **verified, B4** optional relaxed-system primitives pass their LL/AR qualification but do not improve stage time; production primitives stay unchanged.',
        '- **stated** all external baselines are frozen once; interrupted group-change repeats are archived and excluded.', '',
        '## Deviations and limitations', '',
        f'- **verified, C** final pred/meas range{min(pred):.3f}–{max(pred):.3f}; outliers remain in selection CSV; trace locates device-stage costs.',
        '- **stated, C** prompt-mode TTFT includes63 additional head evaluations compared with the separate TP1 prefill path.',
        '- **verified, C** original torch.compile TP4 B1 round spread30.2%; both frozen rounds remain; no baseline repeat is made.',
        '- **inferred** scalar volatile-access waits alone did not explain the slowdown; the controlled lane-mapping result establishes the accepted performance improvement.',
        '- **stated** SM90 reports TP2/TP4 B16=1.5844/1.5283ms on H800; hardware/platform differences remain and this comparison is not a same-device speedup.', '',
        '## GPU accounting and completion', '',
        '**inferred** reserved child-job wall time includes initialization/cleanup, excludes lock/preflight waiting, and is not hardware busy time.',
        'Per-card hours: ' + ', '.join(f'{k}={v:.3f}' for k,v in sorted(usage['card_hours'].items())) + f'; total={usage["sum_card_hours"]:.3f} card-h.',
        '**verified** new ten-cell solver selection, final E2E/C-1/C-2, four delay cohorts and two final traces are complete.',
        '**verified** all required CSVs and selected pins are present; external baselines and protected MoE paths are unchanged.',
        '**inferred** no unimplemented correctness gate remains; performance targets are expectations, not acceptance gates.']
    assert len(lines) <= 300, len(lines)
    (results/'MX7_tp_c550.md').write_text('\n'.join(lines)+'\n')
    print('optimized report lines', len(lines))


if __name__ == '__main__':
    main()
