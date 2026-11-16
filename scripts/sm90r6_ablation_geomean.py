#!/usr/bin/env python3
"""Summarize paired A0 ratios with explicit measured cohorts and missing cells."""
import argparse
import csv
import json
import math
from pathlib import Path
from sm90d1_common import sha
from sm90r6_ablate import ARMS
from sm90r6_merge_analysis import write


def summarize(path):
    rows=list(csv.DictReader(path.open()));cells={(r['model'],r['batch']) for r in rows}
    if len(rows)!=64 or len(cells)!=8 or len({(r['model'],r['batch'],r['arm']) for r in rows})!=64:
        raise ValueError('full 64-row matrix including explicit missing rows required')
    baseline={(r['model'],r['batch']):float(r['tpot_ms']) for r in rows if r['arm']=='A0'}
    result=[]
    for scope in ('all_available','eight_arm_protocol','seven_arm_protocol_exception'):
        cohort=[r for r in rows if scope=='all_available' or
                (r['measurement_protocol']=='P-ABL')==(scope=='eight_arm_protocol')]
        for arm in ARMS:
            subset=[r for r in cohort if r['arm']==arm];valid=[r for r in subset if r['measurement_status']=='verified']
            ratios=[float(r['tpot_ms'])/baseline[(r['model'],r['batch'])] for r in valid]
            if any(int(r['processes'])!=3 for r in valid) or any(not math.isfinite(v) or v<=0 for v in ratios):
                raise ValueError('invalid measured ratio or process count')
            gm=math.exp(sum(map(math.log,ratios))/len(ratios))
            result.append(dict(scope=scope,arm=arm,cells=len(valid),missing=len(subset)-len(valid),
                tpot_over_a0_geomean=gm,change_vs_a0_geomean_pct=(gm-1)*100,
                minimum_ratio=min(ratios),maximum_ratio=max(ratios),
                cell_ids=json.dumps([r['model']+'-B'+r['batch'] for r in valid]),
                missing_cell_ids=json.dumps([r['model']+'-B'+r['batch'] for r in subset if r not in valid]),
                interpretation='descriptive geometric mean of within-cell TPOT/A0 ratios; cohort sizes vary; no aggregate significance test',
                source=str(path),source_sha256=sha(path)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();write(a.out,summarize(a.csv))
