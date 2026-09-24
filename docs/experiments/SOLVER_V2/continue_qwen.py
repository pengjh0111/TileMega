#!/usr/bin/env python3
"""Wait for each fresh Qwen control independently, then run its four arms."""
import json,pathlib,subprocess,time
R=pathlib.Path(__file__).resolve().parents[3];E=pathlib.Path(__file__).resolve().parent
pending={1,4,16,64}
while pending:
    for seq in sorted(pending):
        control=E/'legacy_r8_domain'/f'qwen3_s{seq}'
        solve=next((control/name for name in ('solve.command.json','solve.json') if (control/name).exists()),control/'solve.command.json')
        measurement=control/'selected.cu.measurement.json'
        if solve.exists() and json.loads(solve.read_text())['exit_code']:
            (control/'downstream_blocked.json').write_text(json.dumps(dict(reason='legacy solve failed; inspect raw solve.log')))
            pending.remove(seq);continue
        if not measurement.exists():continue
        if not json.loads(measurement.read_text()).get('winner'):
            (control/'downstream_blocked.json').write_text(json.dumps(dict(reason='legacy control has no internally valid measured candidate')))
            pending.remove(seq);continue
        inp=pathlib.Path('/root/r7_work/qwen3') if seq==4 else pathlib.Path('/root/r9_work')/f'qwen3_s{seq}'
        bridge=inp/'auto.cu.export.json' if seq==4 else E/'admission'/f'qwen3_s{seq}_fxorder_failed'/'selected.cu.export.json'
        for k in ('4','8','16','W'):
            out=E/'matrix'/f'qwen3_s{seq}'/f'skeleton-k{k}';out.mkdir(parents=True,exist_ok=True)
            if (out/'launch.command.json').exists():continue
            cmd=['python3',str(E/'run_matrix.py'),'--bridge',str(bridge),'--fixture',str(inp/'fixture'),'--seed',str(control/'selected.mlir'),'--seq',str(seq),'--k-base',k,'--search-jobs','3','--out',str(out),'--compiler','/root/r9_work/skeleton_bulk_tool/tools/tilemega-compile']
            with (out/'runner.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (out/'launch.command.json').write_text(json.dumps(dict(command=cmd,pid=p.pid),indent=2))
            print('START',seq,k,p.pid,flush=True)
        cmd=[str(R/'build-portable/tools/tilemega-skeleton-audit'),str(control/'selected.mlir'),str(R/'docs/experiments/COSTMODEL/event_fit/target.json'),str(seq),'3',str(control/'eft_queue.tsv'),str(R/'docs/experiments/SIMULATOR/hop_ns.tsv')]
        with (control/'oracle_eft_audit.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (control/'oracle_eft_audit.command.json').write_text(json.dumps(dict(command=cmd,pid=p.pid),indent=2))
        pending.remove(seq)
    if pending:time.sleep(15)
