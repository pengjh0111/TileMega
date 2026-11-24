"""Render the final report using only the quantities in numbers.json."""

import collections
import csv
import shutil

from calibration_queue import CASE
from guarded import ROOT, read


def fmt(x, places=4):
    return 'unavailable' if x is None else f'{x:.{places}f}'


def main():
    n = read(CASE / 'numbers.json')
    assert (CASE / 'measurement_complete.json').exists()
    c = n['costs']
    s = n['selection']
    pred, measured = s[1], s[2]
    e, q, final = (int(x) for x in n['funnel_counts'][1:4])
    gaps = [key for key, passed in n['criteria'].items() if not passed]
    pair = n['local_matched_attention']
    with (CASE / 'timings.csv').open() as handle:
        status = collections.Counter(row['status'] for row in csv.DictReader(handle))
    text = (
        f'For Qwen3-1.7B at batch 1 on H800, the estimated legal space contained '
        f'{n["legal_space_estimate"] / 1e6:.1f} million configurations. The search scored {e} '
        f'configurations, refined {q} shortlisted candidates, and compiled {final} finalists. '
        f'Pooled search, replay, and finalist construction consumed '
        f'{c["production_predicted_search_and_compile_seconds"] / 60:.2f} minutes. '
        f'The audit measured {n["population_valid_unique"]} distinct programs, including '
        f'{n["population_U_unique"]} uniform samples. The lowest measured TPOT was {n["oracle_ms"]:.4f} ms. '
        f'The predicted winner achieved {fmt(pred["tpot_ms"])} ms '
        f'({fmt(pred["gap_pct"], 2)}% regret), and finalist measurement selected '
        f'{fmt(measured["tpot_ms"])} ms ({fmt(measured["gap_pct"], 2)}% regret). '
        f'Across the same measured population, pooled ranking achieved Spearman rho of {n["spearman"]:.3f} '
        f'and retained {n["recall_at_Q"]:.1%} of the measured top-five-percent set in the shortlist. '
        'For this workload, Eq. (5) and the native template produced the same shortlist plans. '
        'These timings use distinct processes with separate confirmation measurements.'
    )
    (CASE / 'paper_paragraph.txt').write_text(text + '\n')
    lines = [
        '# H800 SQ2 search selection quality', '', text, '',
        'Qwen3-1.7B, batch 1, TP1, BF16, ISL64/OSL1024. The shared checkpoint uses deterministic seed-0 weights.',
        'Each process ran one warmup and three timed requests at N1 and N1024. TPOT is the request-wall median difference divided by 1023.',
        'The first process screened 256 positions against HF and checked single-step/persistent token equality outside the timing interval. The guard records interference and device UUID. Three independent H800 lanes filter the same shuffle and its reverse using frozen assignments.',
        'The search budget was 600 s with four additional seeds from seed 20261124. Q uses a 1.5% band, round-robin implementation signatures, and an 8–24 bound. S is capped at six with at most two from each structural class.',
        'P includes the frozen 30-per-decile sample of E, the full shortlist, 100 uniform draws, and the baseline coordinates. The audit oracle is the best measured in P. Deduplication requires CUDA, task/event IR, and execution-manifest equivalence.',
        '', f'Attempt status counts: {dict(status)}.',
        f'Criteria met: {n["criteria"]}. Criteria requiring review: {", ".join(gaps) or "none"}.',
        '', '| Method | Audit TPOT (ms) | Audit regret (%) | Confirmation TPOT (ms) | Confirmation gap (%) | Compiles | Search + compile (s) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in s:
        lines.append(f'| {row["method"]} | {fmt(row["tpot_ms"])} | {fmt(row["gap_pct"], 2)} '
                     f'| {fmt(row["confirmation_ms"])} | {fmt(row.get("confirmation_gap_pct"), 2)} '
                     f'| {row["compiled_candidates"]} | {fmt(row["search_plus_compile_seconds"], 2)} |')
    lines += [
        '', 'Confirmation gaps use the independently confirmed TPOT of the audit oracle identity. A confirmation median requires all five processes. B-rand-k is an expectation from P. The measured mode chooses within S.',
        '', '| Cost stage | Candidates | Time |', '|---|---:|---:|',
        f'| Legal space estimate (95% interval) | {n["legal_space_estimate"]:,.0f} ({n["legal_space_95"][0]:,.0f}–{n["legal_space_95"][1]:,.0f}) | {n["funnel_seconds"][0]:.2f} s Monte Carlo |',
        f'| Pooled search | {e} | {c["search_pooled_seconds"]:.2f} s |',
        f'| Eq. (5) shortlist materialization and replay | {q} | {c["shortlist_eq5_seconds"]:.2f} s |',
        f'| Finalist construction and capacity check | {final} | {c["compile_finalists_wall_seconds"]:.2f} s |',
        f'| Exhaustive E, extrapolated sequential compile and measurement | {e} | {c["extrapolated_E_compile_and_measure_s"] / 3600:.2f} h |',
        f'| Exhaustive legal space, extrapolated sequential compilation | {n["legal_space_estimate"]:,.0f} | {c["extrapolated_legal_space_compile_s"] / 86400:.1f} days |',
        '', f'Finalist compile-command work sum: {c["compile_finalists_command_work_seconds"]:.2f} s. See costs.json for additional audit costs. The extrapolations multiply the candidate counts by median per-candidate costs without parallel speedup.',
        '', f'The local comparison scanned {pair["scanned"]} matched attention pairs. Reversal outside the recorded noise ranges: {pair["reversal"]}.',
        '', '**Fig. SQ-1.** Funnel counts and phase costs. U is the uniformly sampled legal-domain distribution. The dashed line marks the audit oracle.',
        '', '**Fig. SQ-2.** Best-retained regret over compile budgets. Replay ends at the measured Q size. Random shading covers the 5th–95th percentiles of finite-population subsamples. The markers use the actual S selection and class cap.',
        '', '**Fig. SQ-3.** Pooled prediction against measured TPOT on valid distinct programs in P.',
        '', '**Fig. SQ-4.** Midpoint isolated attention stage cost and end-to-end TPOT for the matched attention decision. The rule and noise check are in numbers.json.',
        '', 'The frozen hashes, compact candidate tables, checks and plotting script accompany this report. Full receipts, binaries, generated programs and checkpoint remain in runs/.',
    ]
    (CASE / 'report.md').write_text('\n'.join(lines) + '\n')
    shutil.copyfile(ROOT / 'experiments/search_quality_v2/plot.py', CASE / 'plot.py')
    with (CASE / 'progress.md').open('a') as handle:
        handle.write(f'\n## Phases 4–5 — measured and analyzed\n\n'
                     f'- Accepted {n["population_valid_unique"]} distinct programs in P. '
                     f'Declared criteria: {n["criteria"]}.\n'
                     '- Follow-up workloads require review if any criterion is unmet.\n')


if __name__ == '__main__':
    main()
