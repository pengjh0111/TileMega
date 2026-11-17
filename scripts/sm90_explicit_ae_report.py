#!/usr/bin/env python3
"""Publish all nine explicit-realization processes plus the frozen C/D data."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from sm90d1_common import ROOT, read, sha, write
from sm90_decode_mechanism_ae_report import csv_write, process_row
from sm90_explicit_ae_collect import ORDER


STEM='SM90_explicit_stage_AE_qwen3_1_7b_b1'


def plots(out,rows,ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for i,r in enumerate(rows):
        values=json.loads(r['three_process_tpot'])
        bars=ax.bar(i,r['tpot_ms'],width=.65,color=['#6e7785','#237ab8','#32a18d','#184f38','#ad7047'][i],
            hatch='//' if r['new_or_reused']=='reused' else None)
        ax.scatter([i-.08,i,i+.08],values,color='black',s=17,zorder=3)
        ax.bar_label(bars,fmt='%.4f',padding=7)
    ax.set_xticks(range(5),['A2\nS1 P0','B2\nS16 P0','C*\nS1 P1','D*\nS16 P1','E2\nS1 P0 graph'])
    ax.set_ylabel('Request-wall TPOT (ms); lower is faster')
    ax.set_title('Qwen3-1.7B / H800 / BF16 / BS1 / ISL64 OSL1024')
    ax.set_ylim(0,max(r['tpot_ms'] for r in rows)*1.16)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,-.015,'Dots: three processes. *C/D: reused from earlier sessions.',ha='center',fontsize=9)
    for ext in ('png','svg'):fig.savefig(out/f'{STEM}_tpot.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    bars=ax.bar(range(7),[r['value'] for r in ratios],color='#237ab8')
    ax.bar_label(bars,fmt='%.4f',padding=5)
    ax.set_xticks(range(7),[r['ratio'] for r in ratios])
    ax.axhline(1,color='#555555',linewidth=1,linestyle='--')
    ax.set_ylim(0,max(r['value'] for r in ratios)*1.16)
    ax.set_ylabel('TPOT ratio (x)')
    ax.set_title('Attention decomposition and execution realization (C/D reused)')
    ax.spines[['top','right']].set_visible(False)
    for ext in ('png','svg'):fig.savefig(out/f'{STEM}_ratios.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def report(run):
    state=read(run/'timing/state.json');spec=read(run/'spec.json')
    assert state['status']=='complete' and len(state['items'])==9
    assert tuple(i['arm'] for i in state['items'])==ORDER
    assert sha(run/'spec.json')==state['spec_sha256']
    assert sha(run/'consistency.json')==spec['consistency_sha256']
    processes=[]
    for i in state['items']:
        assert i['status']=='done';p=Path(i['result']);assert sha(p)==i['result_sha256']
        assert read(Path(i['attempts'][-1]['directory'])/'guard_result.json')['code']==0
        m=read(p);a=i['arm'];binary=spec['arms'][a]['decode_sha256']
        assert m['configuration']['decode_sha256']==binary
        assert sha(Path(spec['arms'][a]['decode']))==binary
        assert not any(m['configuration']['checks'].values())
        assert all(e['decode_loop_used'] for e in m['configuration']['executions'] if e['tokens']==1024)
        processes.append(process_row(a,i['slot'],p,binary,False,state['inventory']))
    previous=ROOT/'docs/results/SM90_decode_mechanism_AE_qwen3_1_7b_b1_processes.csv'
    with previous.open() as stream:
        for old in csv.DictReader(stream):
            if old['arm'] not in ('C','D'):continue
            p=Path(old['source']);assert sha(p)==old['source_sha256']
            processes.append(process_row(old['arm'],old['slot'],p,old['binary_sha256'],True,old['inventory']))
    assert len({r['pid'] for r in processes if r['new_or_reused']=='new'})==9
    rows=[]
    for a in ('A2','B2','C','D','E2'):
        group=[r for r in processes if r['arm']==a];assert len(group)==3
        values=[r['tpot_ms'] for r in group];t=statistics.median(values)
        if a in ('C','D'):
            assert t=={'C':1.7686245229777098,'D':1.6754311115689438}[a]
        rows.append(dict(arm=a,decomposition='S1' if a in ('A2','C','E2') else 'S16',
            realization='P0 explicit-stage' if a in ('A2','B2','E2') else 'P1 selected',
            execution='typed-stage CUDA Graph' if a=='E2' else 'megakernel',
            tpot_ms=t,three_process_tpot=json.dumps(values),
            process_range_pct=(max(values)-min(values))/t*100,
            new_or_reused=group[0]['new_or_reused'],binary_sha256=group[0]['binary_sha256']))
    t={r['arm']:r['tpot_ms'] for r in rows}
    comparisons=[('A2','B2','S1/S16 attention decomposition under P0'),
        ('C','D','S1/S16 attention decomposition under P1'),
        ('A2','C','execution realization under S1'),('B2','D','execution realization under S16'),
        ('A2','D','attention decomposition and realization combined'),
        ('E2','A2','matched explicit computation: execution organization, not pure launch latency'),
        ('E2','D','total baseline-to-selected gain')]
    ratios=[dict(ratio=a+'/'+b,value=t[a]/t[b],interpretation=label,
        cross_session=a in ('C','D') or b in ('C','D')) for a,b,label in comparisons]
    attempts=[dict(arm=i['arm'],**a) for i in state['items'] for a in i['attempts'] if a['code']]
    environment={k:len({r[k] for r in processes})==1 for k in
        ('versions','model_config_sha256','prompt_ids_sha256','checkpoint_sha256','sm_clocks','mem_clocks')}
    receipt=dict(evidence='static implementation audit and measured performance only',
        accepted_new_processes=9,reused_processes=6,C_remeasured=False,D_remeasured=False,
        rows=rows,ratios=ratios,spec=spec,consistency=read(run/'consistency.json'),
        state_sha256=sha(run/'timing/state.json'),rejected_attempts=attempts,
        reuse_source=str(previous),reuse_source_sha256=sha(previous),
        C_source_commit='7496665a83f857022f934c6474475be5f6862586',
        D_original_source_commit='4b9d1362e648e2ee2cc7f96da9422f97a6d36c45',
        environment_matches=environment,raw_root=str(run))
    out=ROOT/'docs/results'
    csv_write(out/f'{STEM}.csv',rows);csv_write(out/f'{STEM}_processes.csv',processes)
    csv_write(out/f'{STEM}_ratios.csv',ratios);write(out/f'{STEM}_provenance.json',receipt)
    plots(out,rows,ratios)
    text=('# SM90 explicit-stage realization ablation\n\n'
        'Verified performance scope: Qwen3-1.7B, H800, BS1, TP1, BF16, ISL/OSL 64/1024. '
        'Branch `sm90-explicit-stage-ae-v1` starts at `7496665a83f857022f934c6474475be5f6862586`. '
        'C=1.7686245229777098 ms is reused from that mechanism-AE result; '
        'D=1.6754311115689438 ms is reused from the original R3 main experiment. '
        'C and D were neither rebuilt nor remeasured. Their original process records and '
        'binary identities are retained in the companion CSV and provenance JSON.\n\n'
        'P0 is a prescribed explicit-stage realization, not a tuned operator-local optimum. '
        'P0/P1 changes fusion, preparation and completion together; it is not pure placement/owner/order '
        'search on a fixed task graph. E2/A2 measures execution organization with matched computation '
        'and task mapping; it is not pure launch latency.\n\n'
        'Static implementation audit (not numerical validation):\n\n'
        '| Mechanism | P0 implementation and source |\n|---|---|\n'
        '| InputNorm / gamma | `Sm90DenseExplicit.cuh::ExplicitRms` computes FP32 RMS statistics in the original MMA-fragment/WK reduction order. `Sm90DenseGemm.h` ExternalNorm disables in-GEMM square accumulation; `Sm90DenseEpilogue.h` consumes the inverse at the original output-scaling point. Existing gamma-folded weight recipes remain unchanged. No rounded normalized activation is introduced. |\n'
        '| Residual | O/Down GEMMs use kStore into a reused BF16 workspace, followed by `ExplicitOutput<false>`. This preserves BF16(accumulator) before residual addition and the final BF16 rounding. |\n'
        '| SwiGLU | Gate/Up GEMM retains original interleaved packing and writes BF16 gate/up once. `ExplicitOutput<true>` calls the original ServingEpilogueValue, including BF16 SiLU and product rounding. |\n'
        '| Attention prepare | `ExplicitAttentionPrepare` invokes the original Prepare once per KV group, then stores prepared Q in-place. The ExplicitStages branch in `Sm90DenseAttentionBody.cuh` only loads prepared Q/new KV; it does not normalize, rotate or append again. |\n'
        '| Attention mainloop / merge | Original MMA, TMA, online-softmax, task partition and partial/LSE format are retained. ExplicitStages disables last-arriver merge. S16 uses `ExplicitAttentionMerge` calling the original Merge; S1 writes context directly and has no merge stage. |\n'
        '| Head / embedding | `ExplicitHeadPartialFinish` publishes local per-CTA best/index without final reduction or embedding. `ExplicitArgmax` performs the original final warp reduction, then the next step starts with independent embedding. No full vocabulary logits are materialized. |\n'
        '| Boundaries / graphs | `ExplicitStageProgram` uses one typed list, per-stage aggregate counters, operand bank publication and cyclic/grid-stride task ownership. `ExplicitSequence` uses the same list and task mapping. Grid=132, threads=160, residency=1, cluster=1, and dynamic shared memory=217664 bytes remain fixed. |\n\n'
        'S16 has 12 stages per layer and 340 native stages per decode step; '
        'S1 has 11 stages per layer and 312 per step. Both include initial embedding and '
        'three terminal stages (head statistics, head partial, final argmax). '
        'The imported semantic CTG still contains 143 stages; the dedicated native emitter '
        'expands it explicitly. The authoritative execution list is the generated NativeProgram, '
        'not the old five-stage CTG count. E2 uses that same expanded list, including every '
        'independent semantic operation; graph capture/initialization occurs during engine construction '
        'before timed requests. A2/E2 coordinates and generated Program prefix are byte-identical.\n\n'
        'The specialized emit modes are `explicit-serving` and `explicit-kernel-sequence`, both '
        'with --tmexec-config and no solve. Coordinates retain geometry; the artifact manifest '
        '`realization=explicit_stages_qwen_b1_v1` and compiled type parameters describe the '
        'explicit merge/argmax overrides. P1 uses unchanged default template branches.\n\n'
        'Protocol: A2 B2 E2 / B2 E2 A2 / E2 A2 B2, one physical idle H800, three independent '
        'processes per new arm, one warmup plus three timed requests at each N1/N1024. '
        'TPOT=(median wall N1024 − median wall N1)/1023; arm statistic is the median of '
        'three processes. Range=(max−min)/median×100%. Historical checkpoint seed0, prompt, '
        'prefill binary, L1 decode-loop and guards are retained. No C1/C2, HF, token/logit '
        'comparison, sanitizer, stress, profiling, other workload, sweep or calibration was run.\n\n'
        '| Arm | Decomposition | Realization | Execution | TPOT ms | Three processes ms | Range % | New/reused |\n'
        '|---|---|---|---|---:|---|---:|---|\n')
    for r in rows:text+=(f'| {r["arm"]} | {r["decomposition"]} | {r["realization"]} | '
        f'{r["execution"]} | {r["tpot_ms"]:.9f} | {r["three_process_tpot"]} | '
        f'{r["process_range_pct"]:.6f} | {r["new_or_reused"]} |\n')
    text+='\n| Ratio | Value | Interpretation |\n|---|---:|---|\n'
    for r in ratios:text+=f'| {r["ratio"]} | {r["value"]:.9f} | {r["interpretation"]} |\n'
    text+=('\nVerified observed ordering, fastest first: `'+ ' < '.join(sorted(t,key=t.get))+
        '`. Every prescribed valid measurement is included; no speed-based filtering was used. '
        f'Guard/driver rejected attempts: {len(attempts)} (all records retained). '
        'All ratios involving C or D are cross-session comparisons. No statistical interaction '
        'or numerical-correctness claim is made from this performance-only collection.\n\n'
        'GPU: `'+state['inventory']+'`. New/historical environment matching: `'+json.dumps(environment,sort_keys=True)+
        '`. Clock sets, software versions, prompt/checkpoint hashes, raw request times and binary '
        'hashes are in the process CSV. Existing namespace-limited PID acceptance is disclosed '
        'there; unresolved host PID ownership is not claimed to be verified. No clock or power '
        'setting was changed.\n\n'
        f'![TPOT]({STEM}_tpot.svg)\n\n![Ratios]({STEM}_ratios.svg)\n\n'
        'Reproduction and source locations: `configs/serving/sm90_explicit_ae/README.md`. '
        f'All raw records, compile products and attempts: `{run}`. Earlier result files are unchanged.\n')
    (out/f'{STEM}.md').write_text(text);write(run/'final-summary.json',receipt)
    print(json.dumps(dict(times=t,ratios=ratios),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    report(p.parse_args().run.resolve())
