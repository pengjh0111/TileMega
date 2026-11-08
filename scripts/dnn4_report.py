#!/usr/bin/env python3
"""Render the bounded DNN4 report from the published evidence tables."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('resnet18', 'resnet50', 'vgg16', 'bert')
BATCHES = (1, 2, 4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'docs/results')
    parser.add_argument('--out', type=Path, default=ROOT/'docs/results/DNN4_h800.md')
    args = parser.parse_args()
    def rows(name):
        with (args.results/name).open() as stream:
            return list(csv.DictReader(stream))
    final = rows('DNN4_final.csv')
    targets = {(r['model'], int(r['batch'])): r for r in rows('DNN3_final.csv')}
    measured = {(r['model'], int(r['batch'])): r for r in final}
    wins = sum(float(r['ratio']) < 1 for r in final)
    lines = []
    def add(text):
        lines.extend(text.strip().splitlines())
        lines.append('')
    def table(headers, values):
        add('\n'.join(['| '+' | '.join(headers)+' |',
                       '| '+' | '.join(['---']*len(headers))+' |',
                       *['| '+' | '.join(map(str, row))+' |' for row in values]]))
    add('# DNN4 H800 measured report')
    add('Snapshot UTC: '+datetime.now(timezone.utc).isoformat()+'.\n'
        f'Verified Phase Z: {len(final)}/20 cells measured; {wins}/{len(final)} below fixed targets. '
        'CP-0 is complete;CP-1/2/3 remain unmet as detailed below. '
        'The 20/20 goal is not achieved. P-ANCHOR completion is reported separately.')
    add('## Synchronization (six lines)')
    add('''1. Verified CLAUDE.md/AGENTS.md, review, required F3 sources and CUTLASS references read; G1–G22 match.
2. Verified origin/dnn3 unchanged at 6d8cc4cae621c1d995e8b4a53a2c67e7969d6d29; dnn4 starts there.
3. Verified 700 matrix-stage instances: 682 Slots2, 18 Slots3, no TN256; all 20 DNN3 winner paths/SHA256 recorded under runs/dnn4/sync.
4. Verified nvcc 12.8.93, sanitizer 2025.2.1.0 build 35969825, CUTLASS dc45f979ae336a235da1676b311f35efeb30149a.
5. Verified opening 316-test coverage: 309 pass, six existing failures, one skip; four inherited dense-test compile failures retained.
6. Verified guarded GPU jobs, NUMA0 timing cores separate from nvcc; GPU0 residual 28 MiB, others 4 MiB; container PID ownership remains provisional.''')
    add('''Prompt SHA256:`5c034140cd49711ada2cb48e955c454be8520dcca02fc986d644d656a8134e28`.  
Review SHA256:`a2c6567caa99ba7766c93e24193045e1e4e81976fb65b52c01c2e377cc4be9c1`.  
Diagnosis SHA256:`1bb34856e94c78d2bc751983d3b654b13b8e0bd372463634b89213d03e25b567`.''')
    add('Verified existing failures: generated_feature_contract, r15_model_registry, sm90_dense_epilogue/local_route/moe/router; skip sm90_dense_tp. Opening impl_contract was rerun guarded after a hidden-device fixture error.')
    add('## Phase P and code size')
    cal = rows('DNN4_cal.csv')
    tile_shapes = ((64,32),(64,64),(128,64),(128,128),(128,256))
    values = []
    for tm, tn in tile_shapes:
        for src in ('l2','dram'):
            found = {int(r['slots']): float(r['ns_per_k']) for r in cal
                     if r['family']=='pipe' and r['threads']=='288' and
                     int(r['tile_m'])==tm and int(r['tile_n'])==tn and r['src']==src}
            values.append([f'{tm}x{tn}',src,*[f'{found[s]:.1f}' if s in found else 'oversized'
                                           for s in (2,3,4,6,8)]])
    add('Verified ns/k: 132 CTAs, two tasks, second task, real TMA/bulk, five fresh processes on GPU1. K=2048; 64x32 DRAM uses K=4096 for 64 MiB. DNN4_cal.csv retains source/SASS/header identities.')
    table(['Tile','Source','Slots2','3','4','6','8'],values)
    add('Verified DT-PIPE t6/t2: 288 threads 0.5490/0.4941; 384 threads 0.4830/0.4479 for 64x32/128x128. Both pass. All 20 DNN3 winner SASS sizes and 374 unique stage-template builds (830 uses) are recorded.')
    bandwidth = [r for r in cal if r['family']=='bandwidth' and r['threads']=='288']
    for src in ('l2','dram'):
        bandwidth.append(next(r for r in cal if r['family']=='pipe' and r['threads']=='288'
                              and r['tile_m']=='128' and r['tile_n']=='256'
                              and r['slots']=='4' and r['src']==src))
    bandwidth.sort(key=lambda r:(int(r['tile_n']),r['src']=='dram',int(r['active_ctas'])))
    table(['Tile','Source','CTAs','Slots','CTA GB/s','Total GB/s'],
          [[f"{r['tile_m']}x{r['tile_n']}",r['src'],r['active_ctas'],r['slots'],
            f"{float(r['per_cta_gbps']):.1f}",f"{float(r['total_gbps']):.1f}"] for r in bandwidth])
    resident = []
    for tm,tn in tile_shapes:
        found = {int(r['slots']):float(r['ns_per_k']) for r in cal
                 if r['family']=='resident' and r['threads']=='288' and
                 int(r['tile_m'])==tm and int(r['tile_n'])==tn}
        resident.append([f'{tm}x{tn}',*[f'{found[s]:.1f}' for s in (2,4,8)],f'{found[2]-found[8]:.1f}'])
    add('Verified resident probes: single CTA, 72 samples, discard first eight; handshake=t2−t8. Oversized Slots8 panels alias immutable operands only in these resident probes.')
    table(['Tile','Slots2 ns','Slots4 ns','Slots8 ns','Handshake ns'],resident)
    code = rows('DNN4_icache.csv')
    table(['Code KiB','p_cold us/KiB','p_l2 us/KiB'],
          [[r['code_kb'],f"{float(r['p_cold_us_per_kb']):.6f}",f"{float(r['p_l2_us_per_kb']):.6f}"]
           for r in code if r['kind']=='cold_code_probe'])
    add('Verified cold-code eviction SASS: 1,049,088 B. DNN3 BERT B1/B16 SASS: 1,560,704/2,688,256 B. Inferred conservative cold costs 177.5/305.8 us exceed 3% triggers; Repeat was implemented and measured, then rejected below.')
    add('## Implementation gates and decisions')
    table(['Design','Verified evidence','Disposition'],[
        ['384-thread skeleton','P1 side/prefetch;96-lane gather;TK64/128;Slots2–8;schema4;shared canonical layout','production F3'],
        ['DT-REG','40-register gather spills28B;fallback56/224;consumer uses registers aboveR168','256x128 residual-LN writer spills64B;excluded;full gate unmet'],
        ['Split-K','fragment v4;all slice reads before ascending sums;DNN3 bitwise5/5','64x32 S6=2.041us>1.5;64x128 S16=30.827us>6;time gates unmet'],
        ['Row statistics','64-column parts;reader bulk reduction;cached owned-row normalization;odd-M tests5/5','no last-arriver counters;FP32 gates retained'],
        ['DT-PF','distance2/1 geomean0.984107;RN50B1 +0.47%<2%','distance2'],
        ['DT-GELU','all finite BF16+2^24 points;error1.972e-7;epilogue−25.78%;BERT16 P-PAIR−4.07%','accept A&S erf'],
        ['DT-ST2','VGG16B16 reg1.353872→TMA1.255408ms;3-stage ordering200 launches×5','accept eligible TMA stores;RN50 +1.07% within noise'],
        ['Attention','Rows16/32/64/128;ex2;packed stores;per-head/unit5/5','BERT1 later-layer windows8.99–9.76us>5;time gate unmet'],
        ['MaxPool','32-bit constant geometry;four-item reads ahead;bitwise5/5','VGG features.4=53.792us>50 (DNN3 77.216)'],
        ['TN96/192 and large tiles','88/89 real NT12 tile/chain builds zero spill','exclude256x128 residual-LN writer andTM64 TN192 stats writers'],
        ['DT-MC2','128x128 L2/DRAM ratios2.3858/2.1303;128x256 1.3985/1.4642;66 clusters','reject both;mc0'],
        ['DT-STEM','C8/interleave TMA equals gather5/5;VGG P-PAIR +4.56%;RN50B16 +2.29%','reject;96-lane gather'],
        ['DT-CODE','Repeat6x10 SASS ratios0.2114/0.2040;B1/B16 P-PAIR +2.33%/+0.096%','size gate passes;no3% end-to-end gain;flat default'],
        ['DT-PP WG64','trigger share BERT16=21.818% (VGG7.604%);P-PAIR +10.24%','reject;cooperative default'],
    ])
    pp = ROOT/'runs/dnn4/phase_x/pingpong128_model_v1/result.json'
    if pp.exists():
        state = json.loads(pp.read_text())
        add(f"Verified prescribed WG128 ping-pong recheck:unit5/5,sanitizers1/1/5/5,full-model5/5 byte hashes;P-PAIR ratio{state['pair']['ratio']:.6f}. "
            + ('The3% mechanism gate passes;final-cell admission still requires same-card comparison.' if state['accepted'] else 'Reject;cooperative scheduling remains selected.'))
    else:
        add('Verified WG128 ping-pong focused unit5/5 andsanitizers1/1/5/5 clean;whole-model recheck remains pending. This is the second prescribed WG row width,not an added scheduling mechanism.')
    add('Verified focused body/pipeline/split/rowstats/attention/pool/store/multicast/Repeat/PP suites pass 5/5 with isolated sanitizer memcheck 1/1, initcheck 1/1, racecheck 5/5, synccheck 5/5. Historical full-model checks covered eight B1/B16 cells × five numerical processes, but 40 historical cohorts lacked cross-process byte hashes; their CSV status is partial. All Phase Z winners require the upgraded five-process byte-hash gate.')
    add('Verified boundary trace3 preserves production writes and only four markers; sentinel window sums differ from nontrace by 0.23%, 3.91%, 3.15%, 1.14% (RN18 B1, BERT B1, BERT B16, VGG B16), within 5%. Full trace1 is attribution only. All TileMega timing uses nontrace libraries.')
    add('## Model and in-situ selection')
    add('Verified 516 F4 coefficients from 560 calibration rows including attention recheck; five fresh GPU1 processes per row. Existing native/F3 constant bytes are unchanged. Main loop 146, bandwidth 32, epilogue 123, split 166, gather 70, swap 12, small/attention 10 plus one recheck. Standalone split 66/166 instances spill under the 168-register entry limit; their metadata and measured times are retained.')
    add('Verified CPU formula/coordinate ctest gates pass; 20/20 solves, 80/80 candidate codegens, and 20/20 rank0 native builds pass with zero production spill. Seven fields alter predictions (mc via diagnostic counterfactual only). All 20 program near-tie audits fail 0.5% separation; CP3 is unmet.')
    table(['P-MODEL sentinel','GEMM median error %','GEMM p90 error %','First total error %'],[
        ['RN18 B1','42.36','46.71','41.24'],['RN50 B16','52.20','63.17','53.03'],
        ['VGG16 B16','47.34','50.87','37.24'],['BERT B1','55.52','61.33','54.53'],
        ['BERT B16','49.82','63.89','51.25']])
    add('Verified worst-constant recheck: B1 Rows16 actual 96-task attention is 1734.34995 ns/task (was 1341.05); B1 total error remains 53.80%. Second P-MODEL fails. DT-MODEL uses six separated legal candidates and six variants, three fresh 20-warmup/50-window processes each. Groups with identical shape, chain and neighbors share choices; group windows are summed. Tuned/rank0 P-PAIR then nontrace P-SEL ABCDDCBA determines the winner.')
    add('Verified VGG16 B16 combined trace variant4 FC1 S28 Slots8 spills 32 B; replace only that stage with the already valid S28 Slots6 choice. Six variants contain five distinct FC1 choices; other groups retain six. No spilling program is timed. The known accepted VGG seed was absent from model top-six: post-selection same-card P-PAIR 1.345568→1.261584 ms (−6.24%, threshold 0.5%) accepts its exact rebuilt coordinates; P-DNN/ledger repeated.')
    add('## Phase Z results')
    add('Verified measured rows use fixed G22 cards: RN18/BERT GPU3, RN50 GPU0, VGG GPU2; P-CORR 5/5 byte-identical; P-DNN 50 warmups + 200 events × three fresh processes. Ratio=TileMega/fixed DNN3 target. Pending rows carry no timing inference. P90 pools 600 event samples; process range uses the three medians. DNN4_final.csv contains full coordinate counts, identities, clocks and raw receipts.')
    values = []
    for model in MODELS:
        for batch in BATCHES:
            r = measured.get((model,batch));target=float(targets[model,batch]['baseline_ms'])
            if r is None:
                values.append([model,batch,'pending',f'{target:.6f}','—','—'])
                continue
            coords = Counter(json.loads(r['coordinate_summary']))
            dominant = coords.most_common(1)[0][0]
            values.append([model,batch,f"{float(r['median_ms']):.6f}",f'{target:.6f}',
                           f"{float(r['ratio']):.6f}",r['winner']+';'+dominant])
    table(['Model','B','TileMega ms','Target ms','Ratio','Winner;most frequent coordinate'],values)
    add('Verified aggregates cover only completed rows when fewer than 20; they are not full-model/batch claims until all members complete.')
    table(['Group','Measured cells','Geometric ratio','Geometric speedup','Wins'],
          [[r['kind']+':'+r['group'],r['cells'],f"{float(r['geometric_ratio']):.6f}",
            f"{float(r['geometric_speedup']):.6f}",r['wins']] for r in rows('DNN4_ratios.csv')])
    add('## Unwon cells: three largest P-TRACE1 stages')
    ledger = rows('DNN4_ledger.csv')
    values = []
    for r in final:
        if float(r['ratio']) < 1:
            continue
        stages = sorted([s for s in ledger if s['model']==r['model'] and s['batch']==r['batch']],
                        key=lambda s:float(s['stage_ns']),reverse=True)[:3]
        descriptions = []
        for stage in stages:
            segments = sorted([(k.removesuffix('_max_cta_ns'),float(v)) for k,v in stage.items()
                               if k.endswith('_max_cta_ns') and k not in
                               ('slot_wait_max_cta_ns','side_wait_max_cta_ns')],
                              key=lambda x:x[1],reverse=True)[:2]
            descriptions.append(f"s{stage['stage']} {stage['body']} {float(stage['stage_ns'])/1000:.3f}us ("
                                +','.join(f'{k}={v/1000:.3f}' for k,v in segments)+')')
        values.append([r['model']+' B'+r['batch'],*descriptions])
    table(['Cell','Largest','Second','Third'],values)
    add('## Anchors, deviations and limits')
    add('Verified measurement scheduling update: later cells reserve the GPU once per protocol, with unchanged six-sample preflight and monitoring. Children remain fresh and sequential: P-SEL ABCDDCBA, P-PAIR ABABAB, P-DNN three processes, P-TUNE three processes × 50 samples. Same-library validation differs by +0.165% in P-DNN and +0.182% in trace window sum. Original receipts and resume commands remain available; raw cohort state identifies the shared guard.')
    anchors = ROOT/'runs/dnn4/phase_z/anchors_v1'
    for model in MODELS:
        path = anchors/model/'result.json'
        if not path.exists():
            add(model+':P-ANCHOR pending.')
            continue
        state = json.loads(path.read_text())
        anchor = json.loads((Path(state['out'])/'anchor/result.json').read_text())
        add(f"Verified {model} GPU{anchor['gpu']} P-ANCHOR:{anchor.get('action','incomplete')};"
            +';'.join(f"B{r['batch']} {r['median_ms']:.6f}ms ({r['relative_difference']:+.2%})" for r in anchor['anchors'])+'.')
    add('Stated latest user direction delegates decisions and continued work. Verified deviations: 128x256 (and later 64x256 calibration) L2 footprints exceed 8 MiB; use <9 MiB while preserving K=2048/eight B tiles. S32 cannot satisfy S*C≤64 with v4 reads; uses C4/128 values, ascending arithmetic and zero production spill. CMake registers required CPU tests outside the initial path list.')
    add('Verified TMA horizontal partial boxes use generic tail stores to protect adjacent columns; residual reuse requires a WG read barrier. Sanitizer 2025.2.1 host shadow does not recognize TMA global-store initialization: poison-prefilled outputs plus FP32/canary checks retain an explicit initcheck coverage limit. Failed races, spills, fixtures and invalidated compiler-change attempts remain raw.')
    add('Verified GPU3 foreign occupancy causes guard rejection/retry; no foreign processes or settings are changed. Timing cores: GPU0=0–7, GPU2=104–111, GPU3=112–119; nvcc=136–143, all NUMA0. Start/end SM clocks are in CSV/raw receipts. Namespace-limited ownership qualification applies to all measured results. No baseline arm is added; fixed targets and precision/error thresholds remain unchanged.')
    add('Raw evidence: ignored runs/dnn4/{sync,phase_p,phase_k,phase_r,phase_e,phase_l,phase_s,phase_c,phase_x,phase_z}. Published CSVs: DNN4_cal, icache, constants, smoke, correctness, decisions, model, tune, solver, final, ratios, ledger; baselines_phase_z only if P-ANCHOR triggers resampling. Trace records are losslessly gzip-compressed with hash indexes; generated sources/binaries/logs stay outside git.')
    if len(lines)>250:
        raise ValueError(f'report exceeds250 lines:{len(lines)}')
    args.out.write_text('\n'.join(lines))
    print(json.dumps(dict(lines=len(lines),cells=len(final),wins=wins,report=str(args.out))))


if __name__ == '__main__':
    main()
