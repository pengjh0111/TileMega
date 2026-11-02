"""Render the MX-7 report only after all final evidence is present."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path.cwd()
BASE = ROOT / 'runs/mx7/resume1'
RESULTS = ROOT / 'docs/results'


def latest(name, keys, record_type=None):
    result = {}
    for row in csv.DictReader((RESULTS / name).open()):
        if record_type and row['record_type'] != record_type:
            continue
        result[tuple(row[key] for key in keys)] = row
    return result


def num(value, digits=3):
    return f'{float(value):.{digits}f}'


def main():
    final = latest('MX7_tp_final_c550.csv', ['tp', 'batch'])
    baselines = latest('MX7_tp_baselines_c550.csv', ['tp', 'batch', 'system'],
                       'final_two_fresh_processes')
    assert len(final) == 10 and len(baselines) == 30
    assert all(row['status'] == 'pass' for row in baselines.values())
    traces = {b: json.loads((BASE / f'final_trace/B{b}/trace/trace.json').read_text()) for b in (1, 16)}
    costs = ROOT / 'configs/targets/xcore1000_mx7_tp.json'
    qualification = json.loads(costs.with_suffix('.qualification.json').read_text())
    assert qualification['solver_eligible']
    start = '6cab45b9604e306fef1916863fe16b09d895c32f'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    count = subprocess.check_output(['git', 'rev-list', '--count', f'{start}..{head}'], text=True).strip()
    lines = [
        '# MX-7: Qwen3-1.7B TP2 / TP4 on C550', '',
        'Evidence: **verified** = executed here; **stated** = user specification; **inferred** = reasoning.',
        'Raw successes, failures, generated sources and binaries remain under `runs/mx7/`.', '',
        '## Scope and source identity', '',
        '- **stated** BF16 greedy, ISL/OSL=64/1024, TP2/TP4 × B=1/2/4/8/16.',
        '- **verified** model directory `/root/models/qwen3_1_7b_rand`.',
        '- **verified** branch `tilemega-metax`; starting HEAD `' + start + '`.',
        '- **verified** evidence aggregation HEAD `' + head + '`; commits since start: ' + count + '.',
        '  The report/publish commit follows this evidence HEAD; final HEAD/push status is in the delivery message.',
        '- **verified** specification SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.',
        '- **stated** subsequent all-card authorization and torch.compile continuation:',
        '  SHA256 `f67014253a3cca7234429ff3671e293c0104fb04b68fde1bb9dc52b518c0b284`.',
        '- **verified** TP1 passes the frozen CU-byte/function-instruction-count gate; protected MoE files/final CSV are unchanged; no MoE performance runs.', '',
        '## Environment, C0 and execution sequence (10-line onboarding summary)', '',
        '- **verified** MACA3.8.2.6, cucc/mxcc1.0.0(890c70f3aa), driver3.8.1.',
        '- **verified** Torch2.10.0+metax3.8.2.2; Transformers5.12.0; SGLang0.5.14+detached.c432db2fb.maca3.8.2.122; vLLM0.25.0+empty.',
        '- **verified** NCCL/Gloo available; torch.compile TP uses NCCL. TileMega uses direct MACA P2P.',
        '- **verified** read-only `mx-smi topo -m/-p/-d`: MX links, PCIe NODE labels; reachability is separately qualified.',
        '- **verified** LL8/fine-grained allocation/volatile AS1 scalar64 store/`__ldcg`64 load: pair/all-to-all P1024/P262144 each5/5 ×2000epochs ×104CTAs, zero tearing/timeouts.',
        '- **verified** TP2 main{1,3}:2373.504ns single-way; secondary{4,5}:2629.120ns (five processes); TP4{1,3,4,5}.',
        '- **verified** At104CTAs, selected coalesced writes:8.121GB/s to one peer,19.253GB/s aggregate to three peers.',
        '- **verified** GPUs0/6/7 are unavailable and skipped; no reset/clock/driver changes.',
        '- **inferred** Each rank executes qkv→attention→o+AR→gate_up→down+AR per layer, then head LL argmax exchange produces the common token.',
        '- **inferred** Producers write BF16 partials to all peers, then read q in ascending order; epoch=past+1, prompt runs from past0 and suppresses early token writes.', '',
        'Source hashes, frozen references, exact build/run commands and topology: `runs/mx7/sync/onboarding.md`.', '',
        '## Baselines', '',
        '**verified** Two fresh processes/cell, each warm1+timed2; TPOT=(E2E_N1024−E2E_N1)/1023.',
        'TP2 baselines use secondary{4,5}; TileMega uses C0-selected main{1,3}. TP4 uses{1,3,4,5} for all systems.',
        'TP1 values and baseline scaling efficiencies are retained in `MX7_tp_baselines_c550.csv`.', '',
        '| TP | B | vLLM ms | SGLang ms | torch.compile ms |',
        '|---:|---:|---:|---:|---:|']
    for t in (2, 4):
        for b in (1, 2, 4, 8, 16):
            values = [num(baselines[str(t), str(b), engine]['tpot_ms'])
                      for engine in ('vllm', 'sglang', 'torchcompile')]
            lines.append(f'| {t} | {b} | ' + ' | '.join(values) + ' |')
    lines += ['', '## TileMega final selected plans', '',
        '**verified** Two additional fresh processes/cell, warm1+timed2; TPOT=(E2E−TTFT)/1023.',
        'TTFT includes prompt mode and63 extra head evaluations. Ratios use the same TP degree.',
        'Efficiency=TP1_frozen_TPOT/(TP×TP_TPOT); throughput counts all B×1024 output tokens.', '',
        '| TP | B | TPOT ms | TTFT ms | E2E ms | tokens/s | fastest/TM | TP1/TM | efficiency |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in (2, 4):
        for b in (1, 2, 4, 8, 16):
            row = final[str(t), str(b)]
            values = [num(row[k]) for k in ('tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s', 'speedup_vs_fastest')]
            values += [num(float(row['tp1_frozen_tpot_ms']) / float(row['tpot_ms'])),
                       num(float(row['scaling_efficiency']) * 100, 1) + '%']
            lines.append(f'| {t} | {b} | ' + ' | '.join(values) + ' |')
    lines += ['', 'Each engine ratio, both process values, pin/binary/cost SHA and source identity are in `MX7_tp_final_c550.csv`.',
        'All ten pins come from source-bound k-best/top4 and the1% measured selection rule; no development coordinates are final.',
        '**verified** Cost table SHA256 `' + hashlib.sha256(costs.read_bytes()).hexdigest() + '`.',
        'Competitive noisy keys receive two extra fresh processes; remaining noisy keys use p90.', '',
        'Selected communication cost terms (diagnostic, µs; final column is an **inferred**28-layer projection):', '',
        '| TP | B | o TN | down TN | extra o AR | extra down AR | total AR ms/step |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    constants = json.loads(costs.read_text())['tp_constants']['t_ar']
    for t in (2, 4):
        for b in (1, 16):
            config = json.loads((ROOT / final[str(t), str(b)]['pin']).read_text())['mx_decode_v2']
            values = []
            for stage in ('o', 'down'):
                row = next(x for x in constants if x['key'] == dict(tp=t, b=b, stage=stage, tn=config[stage]['tn']))
                values.append((row['p90_ns'] if row['noisy'] else row['median_ns']) / 1000)
            lines.append(f'| {t} | {b} | {config["o"]["tn"]} | {config["down"]["tn"]} | '
                         f'{values[0]:.2f} | {values[1]:.2f} | {28 * sum(values) / 1000:.3f} |')
    lines += ['',
        '## Correctness and resources', '',
        '- **verified** shard/guard/ABI/selection/trace CPU tests pass (individual runs retained in correctness CSV).',
        '- **verified** AR and head units: TP2/TP4 × B1/B16, each5/5 fresh processes; exact ordered BF16 reference.',
        '- **verified** injected-delay development litmus: TP2/TP4 × B1/B16, each5/5 fresh1024-token processes;',
        '  sequences exactly match ordinary builds; ordinary development C-2 also passes5/5 in all four cells.',
        '- **verified** all40 qualified candidates pass512-position HF C-1 and two fresh1024-token C-2 processes.',
        '- **verified** all ten selected rebuilds pass512-position C-1 and five fresh1024-token C-2 processes;',
        '  every generated token agrees across ranks, including all timing runs.',
        '- **verified** actual all-rank resident=1; MTreg≤255, stack0, static-smem0, dynamic LDS≤64KiB.',
        '  Rejected candidates and actual resources remain in `MX7_resources_c550.csv`.',
        '- **verified** eight frozen plans pass generated-CU byte equality and per-function instruction-count equality',
        '  after relevant frontend/codegen/backend/runtime changes (`MX7_frozen_gate_c550.csv`).', '',
        '## TP4 trace: rank0 per-layer medians (µs)', '',
        '**verified** One diagnostic process/B,1023 decode steps,104CTAs/rank; o/down include AR, head includes exchange.',
        'Each duration uses max CTA release minus the previous stage release. Trace values are not E2E acceptance data.', '',
        '| B | Layer | qkv | attention | o+AR | gate_up | down+AR |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for b, data in traces.items():
        for layer in range(28):
            stages = [x for x in data['stages'] if x['layer'] == layer]
            assert len(stages) == 5
            lines.append(f'| {b} | {layer} | ' + ' | '.join(num(x['rank0_median_ns'] / 1000, 2) for x in stages) + ' |')
        head_row = next(x for x in data['stages'] if x['stage'] == 'head')
        lines.append(f'| {b} | head | ' + num(head_row['rank0_median_ns'] / 1000, 2) + ' | — | — | — | — |')
    lines += ['', 'Maximum inter-rank stage-duration differences (all layers/steps):']
    for b, data in traces.items():
        diffs = {stage: max(x['max_rank_difference_ns'] for x in data['stages'] if x['stage'] == stage) / 1000
                 for stage in ('o', 'down')}
        lines.append(f'- **verified** B{b}: o={diffs["o"]:.3f}µs, down={diffs["down"]:.3f}µs.')
    lines += ['', 'Per-rank medians, raw losslessly compressed traces and source hashes: `MX7_tp_trace_c550.csv`.', '',
        '## Autonomous handling and preserved failures', '',
        '- **verified, B2/B4** original torch.compile lacked accelerate, then fullgraph rejected HF TP gradient hooks.',
        '  Isolated accelerate1.15.0 plus compiled_autograd=True adapts the compiler; no hooks removed or eager fallback.',
        '- **verified, B1** initial native builds reached256MTreg and336–472 stack bytes. Approved per-item send/receive',
        '  plus constant-field peer-pointer selection removes operand materialization; TP1 frozen output remains identical.',
        '- **verified, B4** old GPU6 queue initialization caused the historical A5 stop before LL launch.',
        '  All-card authorization permits healthy groups; optional system-atomic probes also coincided with GPU0/7 loss.',
        '  Causation is unresolved. Old failures remain; unsafe optional combinations are not retried.',
        '  The optional primitive-combination matrix is incomplete; the chosen production combination is fully qualified.',
        '- **verified, B4** one driver bundled20 fresh TP4 checks under one900s guard: aggregate timeout124.',
        '  Individual runs completed in52–65s. B16 injected-delay checks were isolated and pass5/5; no limit reduction.',
        '- **verified, B4** calibration B8/TN128 non-AR kernel used an undersized fixed LDS workspace.',
        '  Host/device now use the actual GEMM LDS bound; old illegal-access record retained. Repaired scopes pass.',
        '- **verified, B4** cost queue rebalanced five incomplete chunks onto idle cards, based solely on completion;',
        '  completed three-process chunks retained; old partial records preserved and relative paths canonically validated.',
        '- **verified, B4** resource ledger assumption corrected by appending CU-derived LDS values; old rows retained.',
        '  One pending query was prematurely labeled rejected; an appended correction and terminal-cohort checks fix the ledger.',
        '  Trace aggregation now normalizes empty layer keys; one repeated head reference remains, pointing to the same process.',
        '- **verified, B4** low disk space handled by hardlinking byte/permission/mtime-identical sealed inputs (219.9MiB freed);',
        '  logical paths and hashes are unchanged. Rank-by-rank lossless trace compression bounds temporary disk to about1.5GiB.',
        '- **verified, B7** cost/constants noise supplements follow the competitive2% rule, otherwise p90.', '',
        '- **verified, B4** later timing pairs share one guard preflight while retaining two fresh processes, each warm1/timed2;',
        '  per-process PID/command/exit records and guard provenance remain; each timing job retains the900s limit.', '',
        '- **verified, B5** TP4 B2 candidate3/4 timing cohorts exited guard75 after external GPU3/GPU5 load appeared.',
        '  Invalid cohorts are excluded; each original cohort gets at most one new two-process timing cohort after idle checks.',
        '  Successful candidate builds/checks/timings remain unchanged; failed raw metrics and source-bound retry receipts remain.',
        '- **verified, B4/B5** final TP4 B2 checks later met external GPU3 interference after three C-2 processes.',
        '  The whole interrupted cohort is unqualified; selected artifact and four valid timings are retained for one final-check retry.',
        '- **verified, B4** resume validation first found a manifest field mismatch and then an archived-root path error.',
        '  Both occurred before a GPU child launch; original logs/receipts are retained separately from the one B5 retry.', '',
        '## Deviations from inferred expectations', '']
    for t, b, low, high in [(2, 1, 2.35, 2.50), (2, 16, 3.3, 3.4), (4, 1, 1.85, 2.05), (4, 16, 2.5, 2.7)]:
        row = final[str(t), str(b)]
        lines.append(f'- **verified** TP{t} B{b}: {num(row["tpot_ms"])}ms versus stated inferred{low}–{high}ms; no goal-based rework.')
    preds = [float(row['pred_over_meas']) for row in final.values()]
    lines += [f'- **verified, C** selected pred/meas range{min(preds):.3f}–{max(preds):.3f}; outliers retained in selection CSV.',
        '- **inferred** P2P packet throughput and measured per-AR overhead limit scaling; trace gives stage locations.',
        '- **stated, C** prompt-mode TTFT includes work absent from the separate TP1 prefill path.', '',
        '## GPU accounting and completion', '']
    compiler_rows = [row for key, row in baselines.items() if key[2] == 'torchcompile']
    widest = max(compiler_rows, key=lambda row: abs(float(row['round1_tpot_ms']) - float(row['round2_tpot_ms'])) / float(row['tpot_ms']))
    spread = abs(float(widest['round1_tpot_ms']) - float(widest['round2_tpot_ms'])) / float(widest['tpot_ms'])
    lines.insert(-3, f'- **verified, C** largest torch.compile fresh-process range={spread:.1%} (TP{widest["tp"]} B{widest["batch"]}); '
                 'both round values remain in CSV, using the specified two-process median.')
    usage = json.loads((BASE / 'gpu_usage_final.json').read_text())
    lines += ['**inferred** child-job wall time, includes initialization/cleanup, excludes lock waiting/preflight; not hardware busy time.',
        'Per-card hours: ' + ', '.join(f'{g}={h:.3f}' for g, h in sorted(usage['card_hours'].items())) +
        f'; total={usage["sum_card_hours"]:.3f} card-h.',
        '**verified** CP0–CP3 production-path evidence and all required CSVs/pins are present; selected-path tests are complete.',
        '**inferred** further communication designs require a new specification; none added beyond this prompt.']
    assert len(lines) <= 300, len(lines)
    (RESULTS / 'MX7_tp_c550.md').write_text('\n'.join(lines) + '\n')
    print('report lines', len(lines))


if __name__ == '__main__':
    main()
