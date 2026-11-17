#!/usr/bin/env python3
"""Publish the new A–E matrix without using historical arm measurements."""
import argparse
import json
from pathlib import Path
import statistics
from sm90_local_global_local import ROOT, CONFIG, sha, write
from sm90_decode_mechanism_ae_report import process_row, csv_write
from sm90_local_global_collect import ORDER

STEM='SM90_local_global_AE_qwen3_1_7b_b1'

def plot(out,rows,ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for i,r in enumerate(rows):
        bar=ax.bar(i,r['tpot_ms'],color=['#737d8c','#277eb5','#36a58d','#215f46','#ba804b'][i],width=.65)
        ax.scatter([i-.08,i,i+.08],json.loads(r['three_process_tpot']),color='black',s=17,zorder=3)
        ax.bar_label(bar,fmt='%.4f',padding=7)
    ax.set_xticks(range(5),['A\nlocal/basic','B\nTM/basic','C\nlocal/planner','D_test\nTM/planner','E\nlocal/graph'])
    ax.set_ylabel('Request-wall TPOT (ms); lower is faster')
    ax.set_title('Qwen3-1.7B / H800 / BS1 / TP1 / BF16 / ISL64 OSL1024')
    ax.set_ylim(0,max(r['tpot_ms'] for r in rows)*1.15);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,-.01,'All five arms newly measured. D_ref is not substituted. Dots: independent processes.',ha='center',fontsize=9)
    for ext in ('png','svg'):fig.savefig(out/f'{STEM}_tpot.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    bars=ax.bar(range(len(ratios)),[r['value'] for r in ratios],color='#277eb5')
    ax.bar_label(bars,fmt='%.4f',padding=5);ax.set_xticks(range(len(ratios)),[r['ratio'] for r in ratios])
    ax.axhline(1,color='#555555',linestyle='--',linewidth=1)
    ax.set_ylim(0,max(r['value'] for r in ratios)*1.15);ax.set_ylabel('TPOT ratio (x)')
    ax.set_title('Frozen structure / planner / execution-organization comparisons')
    ax.spines[['top','right']].set_visible(False)
    for ext in ('png','svg'):fig.savefig(out/f'{STEM}_ratios.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main(run):
    state=json.loads((run/'timing-v1/state.json').read_text())
    spec=json.loads((run/'audit-v1/spec.json').read_text());audit=json.loads((run/'audit-v1/consistency.json').read_text())
    assert state['status']=='complete' and len(state['items'])==15
    assert tuple(i['arm'] for i in state['items'])==ORDER
    assert sha(run/'audit-v1/spec.json')==state['spec_sha256']
    assert sha(run/'audit-v1/consistency.json')==spec['consistency_sha256']
    processes=[]
    for item in state['items']:
        assert item['status']=='done';path=Path(item['result']);assert sha(path)==item['result_sha256']
        guard=json.loads((Path(item['attempts'][-1]['directory'])/'guard_result.json').read_text());assert guard['code']==0
        m=json.loads(path.read_text());arm=item['arm'];binary=spec['arms'][arm]['decode_sha256']
        assert m['configuration']['decode_sha256']==binary and sha(spec['arms'][arm]['decode'])==binary
        assert not any(m['configuration']['checks'].values())
        assert all(e['decode_loop_used'] for e in m['configuration']['executions'] if e['tokens']==1024)
        processes.append(process_row(arm,item['slot'],path,binary,False,state['inventory']))
    assert len({r['pid'] for r in processes})==15
    rows=[]
    for arm in 'ABCDE':
        group=[r for r in processes if r['arm']==arm];assert len(group)==3
        values=[r['tpot_ms'] for r in group];t=statistics.median(values)
        rows.append(dict(arm=arm,decomposition='S_local' if arm in 'ACE' else 'S_TM',
            plan='P_basic' if arm in 'ABE' else 'P_TM (Basic candidate retained on replay tie)',
            execution='stage kernels + CUDA Graph' if arm=='E' else 'static task megakernel',
            tpot_ms=t,three_process_tpot=json.dumps(values),process_range_pct=(max(values)-min(values))/t*100,
            new_or_reused='new (D_test)' if arm=='D' else 'new',binary_sha256=group[0]['binary_sha256'],
            raw_sources=json.dumps([r['source'] for r in group])))
    t={r['arm']:r['tpot_ms'] for r in rows}
    labels=[('A','B','decomposition and compatible implementation selection under Basic'),
        ('C','D','decomposition and compatible implementation selection under TileMega planner'),
        ('A','C','fixed S_local planning; selected task programs identical'),
        ('B','D','fixed S_TM planning; selected task programs identical'),
        ('A','D','combined structure and planning selection'),
        ('E','A','matched task program: megakernel execution organization, not pure launch latency'),
        ('E','D','full system versus stage-granular reference')]
    ratios=[dict(ratio=a+'/'+b,value=t[a]/t[b],interpretation=label,cross_session=False) for a,b,label in labels]
    environment={k:len({r[k] for r in processes})==1 for k in
        ('versions','model_config_sha256','prompt_ids_sha256','checkpoint_sha256','sm_clocks','mem_clocks')}
    assert all(environment.values()),environment
    local=json.loads((CONFIG/'S_local.json').read_text());tm=json.loads((CONFIG/'S_TM.json').read_text())
    attempts=[dict(arm=i['arm'],**a) for i in state['items'] for a in i['attempts'] if a['code']]
    receipt=dict(evidence='verified H800 performance and static program audit; numerical correctness not tested',
        accepted_new_processes=15,reused_matrix_processes=0,environment_matches=environment,
        inventory=state['inventory'],D_ref=tm['D_ref'],D_ref_artifact=tm['artifact'],
        D_test_required='planner and common executor changed; no historical substitution',
        acceptance=sorted({r['acceptance'] for r in processes}),
        raw_root=str(run),state_sha256=sha(run/'timing-v1/state.json'),rejected_attempts=attempts,
        local_structure_sha256=sha(CONFIG/'S_local.json'),tm_structure_sha256=sha(CONFIG/'S_TM.json'),
        local_selection_source=local['source_state'],spec=spec,audit=audit,rows=rows,ratios=ratios,
        frontier_candidates={name:{k:v for k,v in json.loads((run/'audit-v1'/(name+'.frontier-candidate.json')).read_text()).items()
            if k not in ('owner','queues','critical_priority')} for name in ('S_local','S_TM')})
    out=ROOT/'docs/results';csv_write(out/f'{STEM}.csv',rows);csv_write(out/f'{STEM}_processes.csv',processes)
    csv_write(out/f'{STEM}_ratios.csv',ratios);write(out/f'{STEM}_provenance.json',receipt);plot(out,rows,ratios)
    lines=['# SM90 operator-local × planner A–E','',
        'Verified: Qwen3-1.7B, H800, BS1, TP1, BF16, ISL64/OSL1024. All 15 matrix processes are new. '
        'D means D_test on the common static executor. Historical D_ref=1.6754311115689438 ms remains a separate, cross-session reference; it is never substituted.', '',
        '| Arm | Structure | Planner | Execution | TPOT ms | Three processes ms | Range % | Source |',
        '|---|---|---|---|---:|---|---:|---|']
    for r in rows:lines.append(f'| {r["arm"]} | {r["decomposition"]} | {r["plan"]} | {r["execution"]} | {r["tpot_ms"]:.9f} | '+
        ', '.join(f'{v:.9f}' for v in json.loads(r['three_process_tpot']))+f' | {r["process_range_pct"]:.4f} | {r["new_or_reused"]} |')
    lines += ['', '| Comparison | Ratio | Interpretation |','|---|---:|---|']
    for r in ratios:lines.append(f'| {r["ratio"]} | {r["value"]:.9f} | {r["interpretation"]} |')
    lines += ['', '## Attribution', '',
        'Verified planning outcome: both frontier candidates tie the Basic replay score. The declared tie rule retains Basic. '
        'The S_local frontier trial moved 672 task owners; the S_TM trial moved none. Neither trial replaced the retained Basic program. '
        'Thus A/C and B/D execute identical respective task programs. Their timing ratios measure repeatability here and do not demonstrate an executed owner/order optimization benefit.', '',
        'S_local was independently selected from complete isolated-operator measurements before network timing. '
        '2,383 candidates were enumerated; 196 survived the predeclared local-model screen and were measured. '
        'All valid samples are retained. Winners are best measured within this screened domain, not universal isolated optima. '
        'Attention independently selected S16/BK64/slots2/LA. Other selected parameters, local scores, complete candidate identities, input/cache protocol and raw paths are in '
        f'[{STEM}/local_candidates.csv]({STEM}/local_candidates.csv) and the frozen S_local.json.', '',
        'S differs in decomposition and compatible implementation, including the fixed local fusion boundaries. '
        'Local uses explicit norm statistics, Q/K preparation, residual and SwiGLU, and separate embedding; S_TM retains its original fused contracts. '
        'The head keeps bounded argmax completion in both structures; no full vocabulary logits are materialized. '
        'The S_local/ S_TM differences cannot be attributed solely to tile size. E/A changes whole-program execution organization with matched computation and mapping; it is not pure launch latency.', '',
        '## Implementation and checks', '',
        '* `sm90_local_global_local.py` and `sm90_local_global_operator.cuh`: predeclared joint candidate screen, native-body complete-operator timing, 256 MiB L2 eviction outside every sample, 3 warmups/15 samples, five weighted attention lengths.',
        '* `Sm90DenseTaskCoordinate.cuh`, `Sm90DenseGemm.h`, `Sm90DenseAttentionBody.cuh`: explicit task coordinates consumed by producer addresses, consumer output coordinates and task counts. Defaults retain cyclic behavior.',
        '* `sm90_local_global_plan.py`: common Basic RR/topological rule; common critical-frontier/earliest-finish/handoff search and replay with Basic retained as a candidate.',
        '* `Sm90DenseStaticPlan.cuh`: common GPU-consumed static entries, per-track order and acquire/release event program for A–D; one compute task per ordinary entry. Head runs its assigned coordinate list once per track and finishes once.',
        '* `Sm90DenseStaticSequence.cuh`: E uses A’s exact per-unit coordinate lists; kernel completion replaces cross-unit aggregate completion. Capture and initialization are outside timing.',
        '* `sm90_local_global_emit.py`: explicit model-specific native ABI emitter; repacks each weight recipe, rebuilds all tensor-map boxes including an independent GateUp activation map, and resizes attention partial/LSE bindings.',
        '* `sm90_local_global_audit.py`: unique task coverage, acyclic data/order graph, every logical dependency enforced by emitted waits/order, correct event producer counts, matched paired signatures, complete KV coverage and compiled coordinate/queue symbols.', '',
        'These are static implementation checks and runtime exception checks. No C1/C2, HF, token/logit comparison, sanitizer, profiling or extra workload was run. '
        'Storage lifetime checks apply to the selected aggregate plans; no general arbitrary-plan race-free claim is made.', '',
        '## Measurement and provenance', '',
        'One physical H800: `'+state['inventory']+'`. Interleaving: ABCDE / BCDEA / CDEAB. '
        'Each independent process uses N1 and N1024, one warmup and three measured requests each. '
        'TPOT=(median wall_N1024−median wall_N1)/1023; each arm is the median of its three processes. '
        'The checkpoint, prompt, prefill binary and software versions match across all arms. The same seeded synthetic checkpoint protocol as the main experiment is retained.', '',
        f'{len(attempts)} end-to-end guard-rejected attempts were retained. No valid result was removed or selected by expected ordering. '
        'The inherited guard reports `provisional_container_pid_namespace`: container-visible owner checks pass, but host PID ownership is not independently proven. '
        'Idle power thresholds were remapped by physical UUID without retuning thresholds or changing clocks/power limits. '
        'All accepted local preflight samples also pass the correctly remapped threshold.', '',
        f'Raw measurements, failed attempts, generated sources, compiler logs and binaries: `{run}`. '
        'Companion process CSV contains all timed wall samples, process IDs, raw measurement hashes and binary identities. '
        'Full A–D plans are archived under `configs/serving/sm90_local_global_ae/qwen3_1_7b_b1/plans/`; raw plan files and compiled source linkage remain under runs.', '',
        'Reproduction commands and implementation limits: `configs/serving/sm90_local_global_ae/qwen3_1_7b_b1/EXECUTION.md`.', '',
        f'![TPOT]({STEM}_tpot.png)', '',f'![Ratios]({STEM}_ratios.png)','']
    (out/f'{STEM}.md').write_text('\n'.join(lines));write(run/'final-summary.json',dict(rows=rows,ratios=ratios))
    print(json.dumps(dict(rows=rows,ratios=ratios),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
