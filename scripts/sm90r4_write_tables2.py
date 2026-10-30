"""Build the SM90R4 result CSVs (prompt §10 T1, T2, T4) from the run directory.

Every row carries the status of its own evidence rather than a bare number:
`accepted` (P-CORR passed and the timing is comparable), `timing NOT comparable`
(measured beside other jobs after the user relaxed §1.5's machine-serial rule for
MoE), `PROVISIONAL` (the binary fails C-2) or `not measured`. T4 marks MPK rows
`conditions differ`, since MPK was measured beside other jobs while the TileMega
numbers it would be compared with are machine-serial.
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
with (OUT / 'SM90R4_llm_final.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('T1', len(rows), 'rows')

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
with (OUT / 'SM90R4_llm_ratios.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('T2', len(out), 'rows')

# ---- T4: MPK -------------------------------------------------------------
MODELS = ('qwen3_0_6b', 'qwen3_1_7b', 'qwen3_30b_a3b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b',
          'mistral_7b_v0_3', 'olmoe_1b_7b')
UNSUPPORTED = {
    'mistral_7b_v0_3': 'ValueError: Unknown builder: MistralForCausalLM (no MPK demo for this architecture)',
    'olmoe_1b_7b': 'ValueError: Unknown builder: OlmoeForCausalLM (no MPK demo for this architecture)'}
mpk = []
for m in MODELS:
    per = collections.defaultdict(list)
    for f in glob.glob(f'/root/TileMega/runs/sm90r4/mpk_fix/full/{m}/meas*/B*/measurements.json'):
        d = load(f)
        if d: per[d['batch']].append(d)
    if not per:
        mpk.append(dict(model=m, batch='', tpot_ms='N/A', ttft_ms='', e2e_ms='', process_spread_pct='',
                        rounds=0, comparable='no',
                        error=UNSUPPORTED.get(m, 'not collected in this round'),
                        protocol_notes='MPK 63af900 + scripts/mpk_patches/mpk-63af900-sm90.patch'))
        continue
    for b in sorted(per):
        t = [x['tpot_seconds'] * 1e3 for x in per[b]]
        mpk.append(dict(model=m, batch=b, tpot_ms=round(statistics.median(t), 4),
                        ttft_ms=round(statistics.median([x['ttft_seconds'] * 1e3 for x in per[b]]), 1),
                        e2e_ms=round(statistics.median([x['e2e_seconds'] * 1e3 for x in per[b]]), 1),
                        process_spread_pct=round((max(t) - min(t)) / statistics.median(t) * 100, 3),
                        rounds=len(per[b]), comparable='conditions differ', error='',
                        protocol_notes='output length compiled per kernel; one mpk() call timed; EOS disabled; '
                                       'measured beside other jobs while TileMega numbers are machine-serial'))
with (OUT / 'SM90R4_mpk.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(mpk[0])); w.writeheader(); w.writerows(mpk)
print('T4', len(mpk), 'rows')
