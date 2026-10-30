"""Build the SM90R4 result CSVs (prompt §10 T1, T2, T4, T9, T10, T11) from the run directory.

Every row carries the status of its own evidence rather than a bare number:
`accepted` (P-CORR passed and the timing is comparable), `timing NOT comparable`
(measured beside other jobs after the user relaxed §1.5's machine-serial rule for
MoE), `PROVISIONAL` (the binary fails C-2) or `not measured`. T4 marks MPK rows
`conditions differ`, since MPK was measured beside other jobs while the TileMega
numbers it would be compared with are machine-serial; its TileMega column is the
R3 accepted cell, so the ratio is an indication and not a controlled comparison.
"""
import csv, glob, json, os, statistics, collections
from pathlib import Path
OUT = Path('/root/TileMega-r4m/docs/results')
A = '/root/TileMega/runs/sm90r4/accept'
R3 = '/root/TileMega-r4m/configs/serving/sm90r3'
SOLVE = '/root/TileMega-r4m/runs/sm90r4/solve'

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def table(name, rows):
    # Unix newlines: csv's default terminator is CRLF (the R3 closeout precedent).
    with (OUT / name).open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    return len(rows)

# ---- T1: accepted / measured TileMega cells -------------------------------
rows = []
for b in (1, 2, 4, 8, 16):
    f = load(f'{A}/u-final-mistral_7b_v0_3-B{b}/result.json')
    c = load(f'{A}/u-corr-mistral_7b_v0_3-B{b}/result.json')
    if not f: continue
    a = next(iter(f['arms'].values()))
    rows.append(dict(model='mistral_7b_v0_3', batch=b, status='accepted', tpot_ms=round(a['tpot_ms'], 4),
                     ttft_ms=round(a['ttft_ms'], 1), e2e_ms=round(a['e2e_ms'], 1),
                     tokens_per_second=round(a['tokens_per_s'], 1), process_range_pct=round(a['range_pct'], 3),
                     fresh_processes=a['processes'], c1=c and c['c1_pass'], c2=c and c['c2_pass'],
                     concurrency='trial', baseline_source='new (R16 protocol, SM90R4)', note=''))
for b in (2, 4, 8, 16):
    f = load(f'{A}/u-final-r4moe-qwen3_30b_a3b-B{b}/result.json')
    c = load(f'{A}/u-corr-r4moe-qwen3_30b_a3b-B{b}/result.json')
    if not f:
        rows.append(dict(model='qwen3_30b_a3b', batch=b, status='not measured', tpot_ms='', ttft_ms='', e2e_ms='',
                         tokens_per_second='', process_range_pct='', fresh_processes='',
                         c1=c and c['c1_pass'], c2=c and c['c2_pass'], concurrency='trial',
                         baseline_source='frozen R16', note='final stopped on user instruction (MoE re-test dropped)'))
        continue
    a = next(iter(f['arms'].values()))
    r3 = load(f'{R3}/qwen3_30b_a3b_b{b}.json')
    rows.append(dict(model='qwen3_30b_a3b', batch=b, status='timing NOT comparable', tpot_ms=round(a['tpot_ms'], 4),
                     ttft_ms=round(a['ttft_ms'], 1), e2e_ms=round(a['e2e_ms'], 1),
                     tokens_per_second=round(a['tokens_per_s'], 1), process_range_pct=round(a['range_pct'], 3),
                     fresh_processes=a['processes'], c1=c and c['c1_pass'], c2=c and c['c2_pass'],
                     concurrency='trial', baseline_source='frozen R16',
                     note='measured with other jobs on neighbouring cards; %+.2f%% vs R3 machine-serial %.4f ms; inside R3 4-15%% parallel penalty'
                          % ((a['tpot_ms'] / r3['measured']['tpot_ms'] - 1) * 100, r3['measured']['tpot_ms']) if r3 else ''))
for b in (1, 2, 4, 8, 16):
    f = load(f'{A}/u-prov-r4olmoe-olmoe_1b_7b-B{b}/result.json')
    if not f: continue
    a = next(iter(f['arms'].values()))
    rows.append(dict(model='olmoe_1b_7b', batch=b, status='PROVISIONAL (C-2 fails)', tpot_ms=round(a['tpot_ms'], 4),
                     ttft_ms=round(a['ttft_ms'], 1), e2e_ms=round(a['e2e_ms'], 1),
                     tokens_per_second=round(a['tokens_per_s'], 1), process_range_pct=round(a['range_pct'], 3),
                     fresh_processes=a['processes'], c1='not reached', c2=False, concurrency='trial',
                     baseline_source='new (R16 protocol, SM90R4)',
                     note='binary is not reproducible run to run (~1 token in 64); not an accepted cell'))
print('T1', table('SM90R4_llm_final.csv', rows), 'rows')

# ---- T2: ratios against each engine --------------------------------------
base = collections.defaultdict(dict)
for src, tag in ((OUT / 'SM90R4_baselines_new.csv', 'new'), (OUT / 'R16_final_baselines.csv', 'frozen')):
    if not src.exists(): continue
    for r in csv.DictReader(src.open()):
        try: value = float(r.get('tpot_ms') or 'nan')
        except ValueError: continue
        if value == value: base[(r['model'], int(r['batch']))][r['system']] = (value, tag)
out = []
for row in rows:
    if not row['tpot_ms']: continue
    eng = base.get((row['model'], row['batch']), {})
    if not eng: continue
    fastest = min(eng, key=lambda k: eng[k][0])
    rec = dict(model=row['model'], batch=row['batch'], tilemega_tpot_ms=row['tpot_ms'], status=row['status'],
               baseline_source=eng[fastest][1], fastest_engine=fastest, fastest_tpot_ms=eng[fastest][0],
               ratio_vs_fastest=round(eng[fastest][0] / row['tpot_ms'], 4))
    for s in ('vllm', 'sglang', 'tensorrt-llm', 'torch.compile'):
        rec['ratio_vs_' + s] = round(eng[s][0] / row['tpot_ms'], 4) if s in eng else 'N/A'
    out.append(rec)
print('T2', table('SM90R4_llm_ratios.csv', out), 'rows')

# ---- T4: MPK -------------------------------------------------------------
MODELS = ('qwen3_0_6b', 'qwen3_1_7b', 'qwen3_30b_a3b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b',
          'mistral_7b_v0_3', 'olmoe_1b_7b')
UNSUPPORTED = {
    'mistral_7b_v0_3': 'ValueError: Unknown builder: MistralForCausalLM (no MPK demo for this architecture)',
    'olmoe_1b_7b': 'ValueError: Unknown builder: OlmoeForCausalLM (no MPK demo for this architecture)',
    'qwen3_30b_a3b': 'CUDA error at test.cu:1060: out of memory -- MPK cudaMallocs a fused qkv buffer per '
                     'layer while keeping q/k/v, and dies at layer 33 of 48 once the 61 GB expert stack, the '
                     '7 GB KV pool and the duplicates exceed the 80 GB card'}
# The TileMega side of a MPK comparison is the R3 accepted cell of that model
# (R4 re-measures none of them); T4's note records that the conditions differ.
TM = {}
src = OUT / 'SM90R3_llm_final.csv'
if src.exists():
    for r in csv.DictReader(src.open()):
        TM[(r['model'], int(r['batch']))] = float(r['tpot_ms'])
mpk = []
for m in MODELS:
    per = collections.defaultdict(list)
    for f in glob.glob(f'/root/TileMega/runs/sm90r4/mpk_fix/full/{m}/meas*/B*/measurements.json'):
        d = load(f)
        if d: per[d['batch']].append(d)
    if not per:
        mpk.append(dict(model=m, batch='', tpot_ms='N/A', tilemega_tpot_ms='', ratio_mpk_over_tilemega='',
                        ttft_ms='', e2e_ms='', process_spread_pct='', rounds=0, comparable='no',
                        error=UNSUPPORTED.get(m, 'not collected'),
                        protocol_notes='MPK 63af900 + scripts/mpk_patches/mpk-63af900-sm90.patch; '
                                       'demo KV budget kept (16 pages x 4096)'))
        continue
    for b in sorted(per):
        if len(per[b]) < 3:
            mpk.append(dict(model=m, batch=b, tpot_ms='N/A', tilemega_tpot_ms='', ratio_mpk_over_tilemega='',
                            ttft_ms='', e2e_ms='', process_spread_pct='', rounds=len(per[b]), comparable='no',
                            error='incomplete: %d of 3 rounds; at the demo KV budget (16 pages x 4096 = '
                                  '65536 tokens/slot, 128 KB/token) this model needs 80 GB at B8' % len(per[b]),
                            protocol_notes='not published'))
            continue
        t = [x['tpot_seconds'] * 1e3 for x in per[b]]
        tm = TM.get((m, b))
        mpk.append(dict(model=m, batch=b, tpot_ms=round(statistics.median(t), 4),
                        tilemega_tpot_ms=tm if tm else 'N/A',
                        ratio_mpk_over_tilemega=round(statistics.median(t) / tm, 4) if tm else 'N/A',
                        ttft_ms=round(statistics.median([x['ttft_seconds'] * 1e3 for x in per[b]]), 1),
                        e2e_ms=round(statistics.median([x['e2e_seconds'] * 1e3 for x in per[b]]), 1),
                        process_spread_pct=round((max(t) - min(t)) / statistics.median(t) * 100, 3),
                        rounds=len(per[b]), comparable='conditions differ', error='',
                        protocol_notes='output length compiled per kernel; one mpk() call timed; EOS disabled; '
                                       'measured beside other jobs while TileMega numbers are machine-serial'))
print('T4', table('SM90R4_mpk.csv', mpk), 'rows')

# ---- T9: the decision tables (prompt §3 DT-M0..DT-C2) --------------------
# Each row names the input that was actually measured and the branch taken, so
# a row reading NOT REACHED says which input is missing rather than nothing.
dec = [
 dict(table='DT-M0', when='MR0', input='B1/B16 MoE stage ledger split by stage type',
      branch='continue', detail='every type falls under MC1-MC7; none outside them over 15%'),
 dict(table='DT-M1', when='MR2', input='P-PAIR d5 vs deep ring, B1 and B16, machine-serial',
      branch='ADOPT the deep ring',
      detail='B1 +1.563% vs threshold 11.19% indistinguishable; B16 -1.085% vs 0.50% FASTER; neither slower'),
 dict(table='DT-M2', when='after MR4/MR5', input='7 dense development cells, REF-R3 vs post-fix arms',
      branch='keep the R3 dense numbers',
      detail='none slower, mean -0.066%, under the 1% bar: no REF-R4 dense re-measurement'),
 dict(table='DT-M3', when='after the re-solve', input='whether any MoE cell stays under 1.10x',
      branch='NOT REACHED',
      detail='the MoE finals are not comparable (measured beside other jobs), so the gate has no valid '
             'input; MR8 stays unimplemented'),
 dict(table='DT-C1', when='after C3', input='AR protocol litmus', branch='NOT REACHED',
      detail='Phase C not started'),
 dict(table='DT-C2', when='C0', input='visible card count', branch='NOT REACHED',
      detail='Phase C not started'),
 dict(table='MR1 stop', when='body pair', input='P-PAIR REF-R3 vs MR1, machine-serial', branch='keep',
      detail='B1 +2.789% vs threshold 16.36% indistinguishable (arm bimodal: four rounds near 4.80, one '
             '4.41); B16 -0.841% vs 0.50% FASTER'),
 dict(table='MR3 stop', when='body pair', input='P-PAIR REF-R3 vs MR3', branch='keep',
      detail='B1 +3.574% vs 12.27%; B16 +0.316% vs 0.50%; both indistinguishable, no measurable gain'),
 dict(table='MR4 stop', when='body pair', input='P-PAIR REF-R3 vs MR4 (post merge-weight fix)', branch='keep',
      detail='B1 -4.498% vs 9.30%; B16 +0.299% vs 0.50%; both indistinguishable, no measurable gain'),
 dict(table='MR7 selection', when='re-solve', input='P-SEL over the top-4 predicted candidates',
      branch='the fused stage is selected',
      detail='expert_fuse true won at Qwen3 B2/B4/B8/B16 and at OLMoE B1/B2; the end-to-end gain is '
             'UNMEASURED because the finals are not comparable'),
 dict(table='§1.5 override', when='2026-10-30', input='user instruction to stop reserving the machine for MoE',
      branch='applied', detail='MoE P-SEL/P-FINAL take --concurrency; every affected cell records concurrency=trial'),
]
print('T9', table('SM90R4_decisions.csv', dec), 'rows')

# ---- T10: every P-CORR result -------------------------------------------
corr = []
def corr_row(model, batch, unit, note=''):
    d = load(f'{A}/{unit}/result.json')
    if not d: return
    corr.append(dict(model=model, batch=batch, protocol='P-CORR', c1_pass=d.get('c1_pass'),
                     c2_pass=d.get('c2_pass'), overall_pass=d.get('pass'),
                     fresh_processes=d.get('fresh_processes'),
                     processes_token_equal=d.get('processes_token_equal'),
                     hf64_divergence=str(d.get('hf64_divergence'))[:60], note=note))
for b in (1, 2, 4, 8, 16):
    corr_row('mistral_7b_v0_3', b, f'u-corr-mistral_7b_v0_3-B{b}', 'accepted cell')
for b in (2, 4, 8, 16):
    corr_row('qwen3_30b_a3b', b, f'u-corr-r4moe-qwen3_30b_a3b-B{b}', 're-solved fused cell')
corr_row('qwen3_30b_a3b', 1, 'u-corr-fusedfix-qwen3_30b_a3b-B1',
         'MR7 fused cell at R3 coordinates, after the warp-uniform gu_done fix')
corr_row('qwen3_0_6b', 1, 'u-corr-pm-prompt-qwen3_0_6b-B1', 'prompt mode (teacher-forced prefill)')
corr_row('qwen3_0_6b', 1, 'u-corr-pm-control-qwen3_0_6b-B1', 'control: the same cell with its real prefill binary')
# OLMoE: C-2 fails on every cell, so no unit produced a P-CORR result. The
# C-1 half was then run alone (c2 disabled) so the HF comparison could run at
# all: it PASSES on the B1 cell, which separates a reproducibility defect from
# a numerical one. C-1 needs one process, so the unit's later rounds, stopped
# with the rest of the collection, do not affect it.
C1 = load(f'{A}/u-c1only-r4olmoe-olmoe_1b_7b-B1/try1/raw/'
          f'c1only-r4olmoe-olmoe_1b_7b-B1-00-hf/attempt1/job/measurements/c1.json')
for b in (1, 2, 4, 8, 16):
    if Path(f'{A}/corr-r4olmoe-olmoe_1b_7b-B{b}/spec.json').exists():
        c1 = ('PASS (256 positions, max gap %g, %.1f%% of gaps exactly zero)'
              % (C1['max_gap'], C1['gap_zero_ratio'] * 100)) if (C1 and b == 1) else 'not run'
        corr.append(dict(model='olmoe_1b_7b', batch=b, protocol='P-CORR', c1_pass=c1,
                         c2_pass=False, overall_pass=False, fresh_processes=0, processes_token_equal='',
                         hf64_divergence='',
                         note='C-2 FAILS: the executor token comparison; the same binary in the same mode '
                              'differs run to run (about one token in 64)'))
print('T10', table('SM90R4_correctness.csv', corr), 'rows')

# ---- T11: prediction, measurement and tau -------------------------------
sol = []
for root, model, tree in (('r4final', 'mistral_7b_v0_3', '/root/TileMega-r4a3'),
                          ('r4moe', 'qwen3_30b_a3b', '/root/TileMega-r4m'),
                          ('r4olmoe', 'olmoe_1b_7b', '/root/TileMega-r4m')):
    for b in (1, 2, 4, 8, 16):
        cell = f'{model}-B{b}'
        folder = f'{tree}/runs/sm90r4/solve/{root}/{cell}'
        bind = load(f'{folder}/binding.json')
        if not bind: continue
        pred = bind.get('predicted_ns')
        pred = (pred[bind['winner']] if isinstance(pred, list) else list(pred.values())[0]) / 1e6
        unit = (f'u-final-mistral_7b_v0_3-B{b}' if model == 'mistral_7b_v0_3' else
                f'u-final-r4moe-{cell}' if model == 'qwen3_30b_a3b' else f'u-prov-r4olmoe-{cell}')
        m = load(f'{A}/{unit}/result.json')
        tpot = next(iter(m['arms'].values()))['tpot_ms'] if m else None
        coord = load(f'{folder}/decode.so.tmexec.json') or {}
        sol.append(dict(model=model, batch=b, schema=coord.get('schema'),
                        expert_fuse=coord.get('expert_fuse', ''), winner=bind.get('winner'),
                        predicted_ms=round(pred, 4), measured_ms=round(tpot, 4) if tpot else 'N/A',
                        tau=round(tpot / pred, 4) if tpot else 'N/A',
                        measurement_status=('accepted' if model == 'mistral_7b_v0_3' else
                                            'timing NOT comparable' if model == 'qwen3_30b_a3b' else
                                            'PROVISIONAL (C-2 fails)')))
print('T11', table('SM90R4_solver.csv', sol), 'rows')

# ---- every collected P-PAIR (the MoE ablation and DT-M2's dense arms) ----
PAIRS = Path('/root/TileMega/runs/sm90r4/pair')
pair = []
for f in sorted(PAIRS.glob('*/result.json')):
    d = load(f)
    if not d or d['spec']['kind'] != 'pair': continue
    s, labels = d['spec'], list(d['spec']['arms'])
    A_, B_ = (d['arms'][l] for l in labels)
    pair.append(dict(name=s['name'], model=s['model'], batch=s['batch'],
                     concurrency=s.get('concurrency', 'serial'), arm_a=labels[0], arm_b=labels[1],
                     tpot_a_ms=round(A_['tpot_ms'], 4), tpot_b_ms=round(B_['tpot_ms'], 4),
                     range_a_pct=round(A_['range_pct'], 2), range_b_pct=round(B_['range_pct'], 2),
                     processes=A_['processes'], change_pct=round(d['change_pct'], 3),
                     threshold_pct=round(d['threshold_pct'], 3), decision=d['decision']))
print('pairs', table('SM90R4_pairs.csv', pair), 'rows')
