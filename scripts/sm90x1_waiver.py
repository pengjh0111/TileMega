"""Explicit per-cell user waivers; never manufacture a correctness receipt."""
import math

from sm90d1_common import ROOT, read, write

POLICY = ROOT/'configs/bench/sm90x1_parallel_policy.json'


def authorization(cell):
    policy = read(POLICY) if POLICY.exists() else {}
    entry = policy.get('correctness_waivers', {}).get(cell['cell'])
    return entry if entry and entry.get('authorized_by_user') and entry.get('skip_remaining_pcorr') else None


def record(cell, directory):
    entry = authorization(cell)
    if not entry:
        raise ValueError('no explicit user correctness waiver for '+cell['cell'])
    write(directory/'correctness-waiver.json', dict(cell=cell['cell'], authorization=entry,
          pcorr_passed=False, performance_eligible=False,
          evidence='stated user waiver; incomplete correctness attempts retained'))


def passed(directory):
    result = directory/'result.json'
    if not result.exists() or not (directory/'unit.json').exists():
        return False
    data = read(result)
    return bool(data.get('pass') and data.get('c1_pass') and data.get('c2_pass'))


def aggregates(cells, ratios, choices, waived_cells):
    output=[]
    for batch in (1,16,'overall'):
        selected=[c for c in cells if batch=='overall' or c['batch']==batch]
        names={c['cell'] for c in selected};data=[r for r in ratios if r['cell'] in names]
        complete=len(data)==len(selected);waived=sorted(names & set(waived_cells))
        common=dict(batch=batch,correctness_waived_cells=';'.join(waived),
                    performance_eligible=not waived,
                    status='complete_under_user_waiver' if complete and waived else 'complete' if complete else 'pending')
        for metric in ('X','Xprime'):
            values=[r[metric] for r in data]
            output.append(dict(common,metric=metric,count=len(values),denominator=len(selected),
                               value=math.exp(sum(map(math.log,values))/len(values)) if complete else '',
                               min=min(values) if complete else '',max=max(values) if complete else ''))
        for metric in ('Y','Yprime'):
            subset=[r for r in choices if r['cell'] in names]
            denominator=sum(5 if c['moe'] else 6 for c in selected);count=sum(r[metric] for r in subset)
            output.append(dict(common,metric=metric,count=count if complete else '',denominator=denominator,
                               value=count/denominator if complete else ''))
    return output
