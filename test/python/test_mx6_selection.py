"""Independent exhaustive ranking and exact MX-6 measurement selection rules."""
import itertools
import math
import pytest
from tilemega.moe.selection_v3 import STAGES,attention_mean,k_best,select_measured


def test_k_best_matches_exhaustive_including_ties():
    lists={name:[dict(ns=x)for x in values]for name,values in zip(STAGES,
        ([2,3,3],[1,4],[1,2],[2,3],[7,9],[4,5],[10,11]))}
    layers=16;factors=[layers]*6+[1]
    expected=sorted((sum(lists[name][r]['ns']*factors[i]for i,(name,r)in enumerate(zip(STAGES,ranks))),ranks)
        for ranks in itertools.product(*[range(len(lists[name]))for name in STAGES]))
    actual=[(row['predicted_ns'],tuple(row['ranks'][name]for name in STAGES))for row in k_best(lists,layers)]
    assert actual==expected


def test_attention_mean_averages_exact_decode_lengths():
    assert attention_mean([65,256,512,768,1088])==576
    assert attention_mean([17]*5)==17
    with pytest.raises(ValueError):attention_mean([1]*4)
    with pytest.raises(ValueError):attention_mean([1,2,float('nan'),4,5])


@pytest.mark.parametrize('fastest,selected',[(99.01,0),(99.,2),(98.99,2)])
def test_one_percent_rule_is_inclusive(fastest,selected):
    result=select_measured([100,101,102,103],[[99,101],[100,102],[fastest-.1,fastest+.1],[102,104]])
    assert result['selected']==selected
    assert result['measured_ns'][2]==fastest


def test_exact_four_and_two_fresh_measurements_required():
    with pytest.raises(ValueError):select_measured([1]*3,[[1,1]]*3)
    with pytest.raises(ValueError):select_measured([1]*4,[[1,1],[1],[1,1],[1,1]])
    with pytest.raises(ValueError):select_measured([1]*4,[[1,math.inf]]*4)


def test_known_unqualified_costs_cannot_reach_plan_selection(tmp_path):
    import hashlib
    import json
    from tilemega.moe.solve_v3 import check_qualification
    path=tmp_path/'costs.json';path.write_text('{}')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar=path.with_suffix('.qualification.json')
    sidecar.write_text(json.dumps(dict(cost_table_sha256=digest,solver_eligible=False,reason='observed protocol violation')))
    with pytest.raises(ValueError,match='not solver eligible'):
        check_qualification(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='qualification hash differs'):
        check_qualification(path)


def test_excluded_mode_and_noisy_p90_change_ranking():
    from tilemega.moe.selection_v3 import stage_lists,STAGES,LENGTHS
    resources=dict(resident=1,stack=0,static_smem=0,MTreg=100)
    rows=[]
    def row(stage,cost,**key):
        return dict(key=dict(model='example',b=1,stage=stage,**key),median_ns=cost,p90_ns=cost,
                    processes=3,resources=resources.copy())
    for name in STAGES:
        if name=='attention':
            rows.extend(row(name,10,length=n,prefetch=0,split=1)for n in LENGTHS)
        elif name=='down':
            rows.append(row(name,1,combine_mode='unit_owner'))
            rows.append(row(name,100,combine_mode='owner_tile'))
            noisy=row(name,90,combine_mode='owner16');noisy.update(noisy=True,p90_ns=120,processes=5);rows.append(noisy)
        else:rows.append(row(name,10))
    choices=stage_lists(rows,'example',1,['unit_owner'])
    assert [r['key']['combine_mode']for r in choices['down']]==['owner_tile','owner16']
    assert [r['ns']for r in choices['down']]==[100,120]


def test_eight_candidate_remedy_keeps_the_one_percent_rule():
    predicted=list(range(100,108));pairs=[[100,100]]*7+[[98.5,99.5]]
    assert select_measured(predicted,pairs)['selected']==7


def test_disqualified_mode_does_not_renumber_the_kernel_enum():
    import runpy
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    bench=runpy.run_path(str(root/'experiments/mx6/build_costs.py'))
    assert 'unit_owner' not in bench['MODES']
    key=dict(stage='down',b=1,tn=32,wn=8,d=2,experts=64,hidden=2048,intermediate=1024,combine_mode='unit_la')
    _,definitions,_=bench['definitions'](key,2)
    assert definitions['COMBINE']==3


def test_parallel_resource_completion_keeps_prediction_order(tmp_path,monkeypatch):
    import argparse
    import json
    import time
    from tilemega.moe import solve_v3
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'complete_scope':True,'rows':[]}))
    choices=[dict(candidate_rank=i,predicted_ns=100+i,stages={})for i in range(1,12)]
    monkeypatch.setattr(solve_v3,'check_qualification',lambda p:{'excluded_combine_modes':['unit_owner']})
    monkeypatch.setattr(solve_v3,'stage_lists',lambda *a:{})
    monkeypatch.setattr(solve_v3,'k_best',lambda *a:iter(choices))
    monkeypatch.setattr(solve_v3,'pin',lambda choice,sha,purpose='resource_probe':
        {'mx_moe_v3':{'purpose':purpose,'selection':{}}})
    finished=[];queried=[]
    cell=solve_v3.Cell.__new__(solve_v3.Cell)
    cell.a=argparse.Namespace(model_id='qwen3_30b_a3b_l24',batch=1,candidates=4,build_workers=4)
    cell.out=tmp_path/'selection';cell.out.mkdir()
    cell.cell=tmp_path/'selected';cell.cell.mkdir()
    cell.costs=catalog;cell.control=tmp_path/'control'
    def build(pin,folder,query=True):
        if folder==cell.cell:return True,[]
        rank=int(folder.name.removeprefix('rank'))
        time.sleep(.04 if rank%2 else .001)
        finished.append(rank)
        return rank>=3,[]
    def query(folder,label):
        queried.append(int(folder.name.removeprefix('rank')))
        return True
    cell.build=build;cell.query=query
    cell.correctness=lambda *a:{'pass':True}
    cell.measure=lambda *a:{'fresh_process_ns':[100,100]}
    result=cell.run()
    qualified=json.loads((cell.out/'QUALIFIED.json').read_text())['qualified']
    assert finished.index(2)<finished.index(1)
    assert [r['choice']['candidate_rank']for r in qualified]==[3,4,5,6]
    assert queried==[3,4,5,6]
    assert result['choice']['candidate_rank']==3
