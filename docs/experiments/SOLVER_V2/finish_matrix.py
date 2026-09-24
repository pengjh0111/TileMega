#!/usr/bin/env python3
"""Wait for raw matrix production, then archive edges and run every R9 gate.

This survives a disconnected client. It never commits, pushes, changes the
search domain, or substitutes missing results. Failed upstream cells are
reported as terminal failures and the verifier still runs all gates.
"""
import argparse,json,pathlib,subprocess,time
E=pathlib.Path(__file__).resolve().parent

def search_progress(directory):
    path=directory/'selected.cu.search.tsv'
    result=dict(evaluated=0,completed_coordinates=0,domains=[])
    if path.exists():
        for line in path.read_text().splitlines():
            fields=line.split('\t')
            if fields[0]=='EVALUATE':result['evaluated']+=1
            elif fields[0]=='COORDINATE':result['completed_coordinates']+=1
            elif fields[0]=='DOMAIN' and len(fields)==3:result['domains'].append(int(fields[2]))
    return result

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
        pending=[];done=[];searches={}
        for model in ('llama','qwen3'):
            for seq in (1,4,16,64):
                control=E/'legacy_r8_domain'/f'{model}_s{seq}'
                if terminal(control):done.append(str(control.relative_to(E)))
                else:pending.append(str(control.relative_to(E)))
                blocked=(control/'downstream_blocked.json').exists()
                for k in ('4','8','16','W'):
                    arm=E/'matrix'/f'{model}_s{seq}'/f'skeleton-k{k}'
                    if (arm/'solver_admission.json').exists():searches[str(arm.relative_to(E))]=search_progress(arm)
                    if blocked or terminal(arm,True):done.append(str(arm.relative_to(E)))
                    else:pending.append(str(arm.relative_to(E)))
        reference_pending=[str(p.relative_to(E)) for p in (E/'reference'/f'{m}_s{s}' for m in ('gqa2','mha4') for s in (4,128)) if not terminal(p,True)]
        state=dict(observed_ns=time.time_ns(),terminal=done,pending=pending,reference_pending=reference_pending,searches=searches)
        temporary=E/'matrix_progress.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(E/'matrix_progress.json')
        print('MATRIX terminal='+str(len(done))+'/40 pending='+str(len(pending)),flush=True)
        if not pending and not reference_pending:break
        time.sleep(a.poll_seconds)
    complete=[p.parent for p in (E/'matrix').glob('*/*/selected.cu.timing.tsv')]
    if complete:
        edges=subprocess.run(['python3',str(E/'archive_edges.py'),*map(str,complete)])
        print('EDGE_ARCHIVE exit='+str(edges.returncode),flush=True)
    resources=subprocess.run(['python3',str(E/'archive_resources.py')])
    print('RESOURCE_ARCHIVE exit='+str(resources.returncode),flush=True)
    winners=subprocess.run(['python3',str(E/'audit_winners.py')])
    print('WINNER_ORACLE exit='+str(winners.returncode),flush=True)
    tables=subprocess.run(['python3',str(E/'report_tables.py')])
    print('REPORT_TABLES exit='+str(tables.returncode),flush=True)
    with (E/'verify_after_matrix.log').open('w') as log:
        p=subprocess.run(['python3',str(E/'verify.py')],stdout=log,stderr=subprocess.STDOUT)
    print('VERIFY exit='+str(p.returncode)+' log='+str(E/'verify_after_matrix.log'),flush=True)

if __name__=='__main__':main()
