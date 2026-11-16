#!/usr/bin/env python3
"""Reject incomplete or corrupted dependency certificates for real F3 probes."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    target=json.loads(args.target.read_text());audit=json.loads(args.audit.read_text())
    results=[str(ROOT/r['result']) for r in target['sm90_f3_constants']['raw_sources']]
    cases=[('valid',audit,True)]
    bad=copy.deepcopy(audit);bad['rows'].pop(0);cases.append(('missing_probe',bad,False))
    bad=copy.deepcopy(audit);bad['rows'][0]['dependencies_sha256']='0'*64;cases.append(('raw_digest',bad,False))
    bad=copy.deepcopy(audit);name=next(iter(bad['rows'][0]['headers']));bad['rows'][0]['headers'][name]='0'*64
    cases.append(('changed_dependency',bad,False))
    bad=copy.deepcopy(audit);bad['target_sha256']='0'*64;cases.append(('target_digest',bad,False))
    rows=[]
    for name,value,expected in cases:
        certificate=args.out/(name+'.json');certificate.write_text(json.dumps(value)+'\n')
        out=args.out/(name+'-target.json')
        command=[sys.executable,str(ROOT/'scripts/dnn3_pack_constants.py'),'--results',*results,
                 '--target',str(args.target),'--dependency-audit',str(certificate),'--out',str(out)]
        with (args.out/(name+'.log')).open('w') as stream:
            r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        assert (r.returncode==0)==expected,name
        if expected:
            packed=json.loads(out.read_text())
            assert packed['sm90_native_constants']==target['sm90_native_constants']
            assert packed['sm90_f3_constants']['by_gpu']==target['sm90_f3_constants']['by_gpu']
        else:assert not out.exists()
        rows.append(dict(case=name,exit_code=r.returncode,expected_pass=expected))
    (args.out/'result.json').write_text(json.dumps(dict(evidence='verified',status='passed',rows=rows),indent=2)+'\n')
    print('F3 dependency audit: valid table retained; four corruptions rejected')


if __name__=='__main__':
    main()
