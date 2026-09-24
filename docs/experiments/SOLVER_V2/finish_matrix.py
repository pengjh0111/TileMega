#!/usr/bin/env python3
"""Wait for raw matrix production, then archive edges and run every R9 gate.

This survives a disconnected client. It never commits, pushes, changes the
search domain, or substitutes missing results. Failed upstream cells are
reported as terminal failures and the verifier still runs all gates.
"""
import argparse,json,pathlib,subprocess,time
E=pathlib.Path(__file__).resolve().parent

def terminal(directory,oracle_required=False):
    for name in ('solve.command.json','solve.json','measure.command.json','measure.json'):
        p=directory/name
        if p.exists() and json.loads(p.read_text()).get('exit_code',0):return True
    # The legacy runner records measure.json; skeleton records measure.command.json.
    measured=(directory/'selected.cu.measurement.json').exists() and any((directory/name).exists() for name in ('measure.json','measure.command.json'))
    return measured and (not oracle_required or (directory/'oracle_audit.command.json').exists())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--poll-seconds',type=float,default=30);a=ap.parse_args()
    if a.poll_seconds<=0:raise ValueError('poll interval must be positive')
    while True:
        pending=[];done=[]
        for model in ('llama','qwen3'):
            for seq in (1,4,16,64):
                control=E/'legacy_r8_domain'/f'{model}_s{seq}'
                if terminal(control):done.append(str(control.relative_to(E)))
                else:pending.append(str(control.relative_to(E)))
                blocked=(control/'downstream_blocked.json').exists()
                for k in ('4','8','16','W'):
                    arm=E/'matrix'/f'{model}_s{seq}'/f'skeleton-k{k}'
                    if blocked or terminal(arm,True):done.append(str(arm.relative_to(E)))
                    else:pending.append(str(arm.relative_to(E)))
        reference_pending=[str(p.relative_to(E)) for p in (E/'reference'/f'{m}_s{s}' for m in ('gqa2','mha4') for s in (4,128)) if not terminal(p,True)]
        state=dict(observed_ns=time.time_ns(),terminal=done,pending=pending,reference_pending=reference_pending)
        temporary=E/'matrix_progress.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(E/'matrix_progress.json')
        print('MATRIX terminal='+str(len(done))+'/40 pending='+str(len(pending)),flush=True)
        if not pending and not reference_pending:break
        time.sleep(a.poll_seconds)
    complete=[p.parent for p in (E/'matrix').glob('*/*/selected.cu.timing.tsv')]
    if complete:subprocess.run(['python3',str(E/'archive_edges.py'),*map(str,complete)],check=True)
    with (E/'verify_after_matrix.log').open('w') as log:
        p=subprocess.run(['python3',str(E/'verify.py')],stdout=log,stderr=subprocess.STDOUT)
    print('VERIFY exit='+str(p.returncode)+' log='+str(E/'verify_after_matrix.log'),flush=True)

if __name__=='__main__':main()
