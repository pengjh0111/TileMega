#!/usr/bin/env python3
"""Losslessly archive completed diagnostic compiler logs; native inputs stay direct."""
from pathlib import Path
import argparse,gzip,hashlib,json

def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pack(log):
    log=Path(log).resolve();suffix='.attribution.so.ptxas.log'
    if not str(log).endswith(suffix):raise ValueError('only auxiliary attribution logs may be archived')
    normal=Path(str(log)[:-len(suffix)]);receipt=Path(str(normal)+'.attribution.json');identity=Path(str(normal)+'.identity.json')
    if not receipt.exists()or not identity.exists():return None
    proof=read(receipt);expected=proof['annotated_log'];assert Path(expected['path']).resolve()==log
    raw=log.read_bytes();assert sha(raw)==expected['sha256']and proof['annotated_device_text_bitwise_equal']and not proof['timing_binary_changed']
    primary=normal.read_bytes();primary_log=Path(str(normal)+'.ptxas.log').read_bytes();ident=read(identity);assert sha(primary)==ident['binary_sha256']and sha(primary_log)==ident['resources_sha256']
    target=Path(str(log)+'.gz');marker=Path(str(log)+'.archive.json');compressed=gzip.compress(raw,compresslevel=3,mtime=0)
    if target.exists():assert target.read_bytes()==compressed
    else:
        with target.open('xb')as f:f.write(compressed)
    assert gzip.decompress(target.read_bytes())==raw
    record=dict(kind='closed_auxiliary_attribution_log',schema_version=1,original=dict(path=str(log),sha256=sha(raw),bytes=len(raw)),archive=dict(path=str(target),sha256=sha(compressed),bytes=len(compressed)),native_binary=dict(path=str(normal),sha256=sha(primary)),native_required_log=dict(path=str(normal)+'.ptxas.log',sha256=sha(primary_log)),roundtrip_verified=True,reason='B11 storage; auxiliary diagnostic compiler log only; original timing binary, identity and required native MIR log remain direct')
    if marker.exists():assert read(marker)==record
    else:
        with marker.open('x')as f:json.dump(record,f,indent=2);f.write('\n')
    assert log.read_bytes()==raw and normal.read_bytes()==primary and Path(str(normal)+'.ptxas.log').read_bytes()==primary_log
    log.unlink();return record
def restore(log):
    log=Path(log).resolve();record=read(str(log)+'.archive.json');p=Path(record['archive']['path']);compressed=p.read_bytes();assert sha(compressed)==record['archive']['sha256'];raw=gzip.decompress(compressed);assert sha(raw)==record['original']['sha256']and len(raw)==record['original']['bytes']
    if log.exists():assert log.read_bytes()==raw
    else:
        with log.open('xb')as f:f.write(raw)
    print('ATTRIBUTION_LOG_RESTORE_VERIFIED',str(log),len(raw))
def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True);p=sub.add_parser('pack');p.add_argument('root',type=Path);p=sub.add_parser('restore');p.add_argument('log',type=Path);a=parser.parse_args()
    if a.action=='restore':restore(a.log);return
    count=saved=0
    for log in a.root.rglob('*.attribution.so.ptxas.log'):
        result=pack(log)
        if result:count+=1;saved+=result['original']['bytes']-result['archive']['bytes']
    print(json.dumps(dict(archived_logs=count,verified_saved_bytes=saved,native_inputs_unchanged=True)))
if __name__=='__main__':main()
