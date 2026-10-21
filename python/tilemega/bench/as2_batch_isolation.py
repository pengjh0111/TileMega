"""Audit batch-row zero against the same model/input's independent B1 request."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from tilemega.bench.as2_models import RUNS,controlled


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('b1-binding','batch-binding','b1-native','batch-native','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();one,many,first,full,out=(controlled(v,[RUNS]) for v in (a.b1_binding,a.batch_binding,a.b1_native,a.batch_native,a.out))
    if out.exists():raise FileExistsError(out)
    x=json.loads((one/'request.json').read_text());y=json.loads((many/'request.json').read_text())
    for key in ('model_id','config_sha256','weight_manifest_sha256','input_sha256','isl','capacity','geometry_variant'):
        if x[key]!=y[key]:raise ValueError('Batch comparison identity mismatch: '+key)
    if x['batch']!=1 or y['batch']!=16:raise ValueError('Require B1 and B16')
    rx,ry=(json.loads((root/'result.json').read_text()) for root in (first,full))
    if rx['osl']!=ry['osl'] or rx['repetitions']!=1 or ry['repetitions']!=1 or rx['warmup'] or ry['warmup']:
        raise ValueError('Require paired single diagnostic requests')
    for phase in ('prefill','decode'):
        s1=list(struct.iter_unpack('<17Q16I',(one/phase/'stages.bin').read_bytes()))
        sn=list(struct.iter_unpack('<17Q16I',(many/phase/'stages.bin').read_bytes()))
        if len(s1)!=len(sn) or any(u[17:19]+u[21:]!=v[17:19]+v[21:] for u,v in zip(s1,sn)):
            raise ValueError('Different Body/geometry/epsilon semantics')
    def input_record(identity,name):return next(r for r in identity['input_buffers'] if r['name']==name)
    for name in ('serving.rope_cos','serving.rope_sin'):
        if input_record(x,name)['sha256']!=input_record(y,name)['sha256']:raise ValueError('Different RoPE coefficients')
    tx,ty=(root/'inputs'/input_record(identity,'serving.tokens')['file'] for root,identity in ((one,x),(many,y)))
    if tx.read_bytes()!=ty.read_bytes()[:tx.stat().st_size]:raise ValueError('Different batch-row-zero input')
    row=[]
    def compare(name,p1,pn):
        a,b=p1.read_bytes(),pn.read_bytes()
        if len(b)!=16*len(a):raise ValueError('Unexpected batch-major payload extent: '+name)
        row.append(dict(name=name,bitwise=a==b[:len(a)],b1_bytes=len(a),
            b1_sha256=hashlib.sha256(a).hexdigest(),batch_row0_sha256=hashlib.sha256(b[:len(a)]).hexdigest()))
    p1=first/'request_0';pn=full/'request_0';compare('tokens',p1/'tokens.bin',pn/'tokens.bin')
    stages=list(struct.iter_unpack('<17Q16I',(one/'decode/stages.bin').read_bytes()))
    head=next(s for s in stages if s[17]==0 and s[18]==3)
    one_buffers={b['offset_bytes']:b for b in x['buffers']};name=one_buffers[head[16]]['name']
    id1=one_buffers[head[16]]['id'];idn=next(b['id'] for b in y['buffers'] if b['name']==name)
    compare('full_logits',p1/'decode'/f'{id1}.bin',pn/'decode'/f'{idn}.bin')
    for file in sorted((p1/'decode').glob('kv_*.bin')):compare(file.name,file,pn/'decode'/file.name)
    result=dict(schema='tilemega.as2.batch_isolation.v1',status='pass' if all(r['bitwise'] for r in row) else 'fail',
        scope='C2 batch-row zero only; not independent-reference model correctness or all-row isolation',
        model_id=x['model_id'],osl=rx['osl'],files_checked=len(row),failures=[r for r in row if not r['bitwise']],rows=row)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
