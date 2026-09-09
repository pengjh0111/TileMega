#!/usr/bin/env python3
"""Export a measured event profile to a separate, non-overwritten target file."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ('base','fit','raw','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('refusing to overwrite a target profile')
    if not (a.raw/'status.txt').read_text().startswith('REQUESTED PHASES COMPLETE'):
        raise RuntimeError('measurement is incomplete')
    correctness=list(csv.DictReader((a.raw/'correctness.tsv').open(),delimiter='\t'))
    if len(correctness)!=600 or any(row['pass']!='1' for row in correctness):
        raise RuntimeError('twelve-cell 50-process correctness gate is incomplete')
    fit=json.loads((a.fit/'coefficients.json').read_text())
    target=json.loads(a.base.read_text())
    def rate(value,unit):
        return dict(ns=value,reason='measured',unit=unit)
    notify,poll=fit['rates']['notify'],fit['rates']['poll']
    event=dict(source=str(a.raw/'attrib.tsv'),
        source_sha256=hashlib.sha256((a.raw/'attrib.tsv').read_bytes()).hexdigest(),
        method=fit['model']+'; '+fit['fit']+'; twelve cells, paired warmup=5/repeat=11',
        notify=rate(notify['task_refs'],'ns/runtime_task_ref'),
        poll=rate(poll['waits'],'ns/runtime_wait_entry'),
        notify_stage=rate(notify['variant_stages'],'ns/runtime_stage'),
        poll_stage=rate(poll['variant_stages'],'ns/runtime_stage'),
        notify_longest_worker=rate(notify['max_worker_task_refs'],'ns/max_worker_task_ref'),
        poll_longest_worker=rate(poll['max_worker_task_refs'],'ns/max_worker_task_ref'),
        fence=dict(ns=None,reason='not_calibrated',unit='ns/fence_free_producer'))
    target.setdefault('event_calibration_by_dtype',{})['bf16']=event
    a.out.write_text(json.dumps(target,indent=2)+'\n')


if __name__=='__main__':
    main()
