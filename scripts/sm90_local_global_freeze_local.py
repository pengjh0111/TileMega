#!/usr/bin/env python3
"""Freeze isolated winners before any end-to-end matrix measurement."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from sm90_local_global_local import CONFIG, ROOT, SHAPES, sha, write

def main(state_path):
    state=json.loads(state_path.read_text())
    if state['status']!='complete':raise ValueError('no guard-accepted complete collection')
    domain=json.loads((CONFIG/'local_domain.json').read_text())
    assert state['domain_sha256']==sha(CONFIG/'local_domain.json')
    directory=Path(state['accepted']);measured={}
    for file in sorted(directory.glob('*.jsonl')):
        for line in file.read_text().splitlines():
            row=json.loads(line);row.update(raw_source=str(file),raw_sha256=sha(file))
            if len(row['samples_us'])!=15 or min(row['samples_us'])<=0:raise ValueError('invalid timing sample')
            measured.setdefault(row['candidate'],[]).append(row)
    assert set(measured)=={p['id'] for p in domain['candidates'] if p['shortlisted']}
    scored=[];by_family={}
    for p in domain['candidates']:
        row=dict(p,measured_us=None,winner=False)
        if p['shortlisted']:
            samples=measured[p['id']]
            if p['family']=='attention':
                intervals=domain['attention_intervals'];assert {r['past'] for r in samples}=={r['past'] for r in intervals}
                times={r['past']:statistics.median(r['samples_us']) for r in samples}
                row['measured_us']=sum(times[i['past']]*i['steps'] for i in intervals)/1023
            else:
                assert len(samples)==1 and samples[0]['past']==-1
                row['measured_us']=statistics.median(samples[0]['samples_us'])
            assert all(r['shared_bytes']==p['shared_bytes'] for r in samples)
            row['records']=samples;by_family.setdefault(p['family'],[]).append(row)
        scored.append(row)
    winners={name:min(rows,key=lambda r:(r['measured_us'],r['id'])) for name,rows in by_family.items()}
    assert set(winners)==set(SHAPES)|{'attention'}
    for row in winners.values():row['winner']=True
    coordinates=dict(schema=5,model='dense',gemms={},attention={},residency=1,cluster=1,
        swiglu_u=8,weight_stream=False,layer_loop=True,emb='stage',group_wait=False)
    for family in SHAPES:
        p=winners[family]
        coordinates['gemms'][family]=dict(tile_n=p['tn'],warps_n=p['wn'],warps_k=p['wk'],
            slot_bytes=2*p['tn']*p['tk'],slots=p['slots'],mma='mma_sync',register_stages=p['register_stages'],multicast=False)
    p=winners['attention'];coordinates['attention']=dict(chunks=p['chunks'],block_keys=p['block_keys'],
        slots=p['slots'],new_row_work=0,merge=p['merge'])
    path=CONFIG/'S_local.json'
    if path.exists():raise ValueError('winner is frozen; do not reselect with network results')
    receipt=dict(schema=1,structure='S_local',coordinates=coordinates,
        boundary_spec='README.md',boundary_spec_sha256=sha(CONFIG/'README.md'),
        domain_sha256=sha(CONFIG/'local_domain.json'),source_state=str(state_path),source_state_sha256=sha(state_path),
        accepted_guard=str(directory.parent/'guard_result.json'),accepted_guard_sha256=sha(directory.parent/'guard_result.json'),
        selected={name:row['id'] for name,row in winners.items()},
        isolated_us={name:row['measured_us'] for name,row in winners.items()},
        measured_candidates=len(measured),whole_program_measurements_at_freeze=0,
        selection='complete-operator measured median; weighted five-length attention; no D tie preference',
        shared_frontend_choices='packed QKV and GateUp; U8; gamma folded weights; FP32 RMS statistics then original output scaling',
        fixed_realization='separate norms/preparation/residual/SwiGLU/embedding; head internal final argmax; attention completion selected locally',
        provenance=str(directory.parents[2]/'local-v1/compiled-provenance.json'),winners=winners)
    write(path,receipt)
    output=ROOT/'docs/results/SM90_local_global_AE_qwen3_1_7b_b1';output.mkdir(parents=True,exist_ok=True)
    fields=['family','id','screen_group','tn','tk','wn','wk','slots','register_stages','chunks','block_keys','merge',
            'shared_bytes','predicted_ns','prediction_source','shortlisted','measured_us','winner','raw_sources']
    with (output/'local_candidates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in scored:
            writer.writerow(dict(row,raw_sources=';'.join(sorted({r['raw_source'] for r in row.get('records',[])}))))
    write(output/'local_selection.json',dict(S_local_sha256=sha(path),winners=receipt['selected'],
        isolated_us=receipt['isolated_us'],valid_samples=sum(len(r['samples_us']) for rr in measured.values() for r in rr),
        rejected_guard_attempts=sum(a.get('code')==75 for a in state['attempts']),
        input_protocol=domain['inputs'],cache_protocol=domain['cache'],inventory=state['inventory']))
    print(json.dumps(dict(selected=receipt['selected'],isolated_us=receipt['isolated_us'],coordinates=coordinates),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    main(p.parse_args().state.resolve())
