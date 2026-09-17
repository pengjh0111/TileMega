#!/usr/bin/env python3
"""Join disjoint exhaustive ISL certificates; never infer missing points."""
import csv,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent

def rows(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def write(p,rs):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]),delimiter='\t');w.writeheader();w.writerows(rs)
def main():
 out=HERE/'complete';out.mkdir(exist_ok=True);base=HERE/'gqa2_pieces';proofs=[r for r in rows(base/'proofs.tsv') if (r['family'],r['grid'])!=('wavefront','340')];samples=[r for r in rows(base/'samples.tsv') if (r['family'],r['grid'])!=('wavefront','340')]
 for r in samples:
  for field in ('template','native'):shutil.copyfile(base/r[field],out/r[field])
 for p in base.glob('*.isl'):
  if not p.name.startswith('wavefront_g340'):shutil.copyfile(p,out/p.name)
 shards=[]
 for start in range(1,129,16):
  root=HERE/'wave340_shards'/f's{start}_{start+15}';rs=rows(root/'proofs.tsv');covered=set()
  for r in rs:
   assert all(r[k]=='1' for k in ('total','bijective','dense','acyclic','resident','level_exact')),r
   covered.update(range(int(r['begin']),int(r['end'])+1))
  assert covered==set(range(start,start+16)),str(root)
  proofs+=rs;ss=rows(root/'samples.tsv');samples+=ss
  for r in ss:
   for field in ('template','native'):shutil.copyfile(root/r[field],out/r[field])
  shards.append(root)
 for part in ('tasks','dependencies','pi_sigma','rank'):
  prefix=None;bodies=[]
  for root in shards:
   text=(root/f'wavefront_g340.{part}.isl').read_text().strip();head,tail=text.split('{',1);body,suffix=tail.rsplit('}',1);assert not suffix.strip();assert prefix in (None,head);prefix=head;bodies.append(body.strip().rstrip(';'))
  (out/f'wavefront_g340.{part}.isl').write_text(prefix+'{ '+'; '.join(bodies)+' }\n')
 write(out/'proofs.tsv',proofs);write(out/'samples.tsv',samples)
 (out/'provenance.json').write_text(json.dumps(dict(base=str(base),exhaustive_shards=list(map(str,shards)),source_cg='docs/experiments/WRITEBACK/gqa2_resident_auto.mlir',scope='seq integer interval [1,128], grids 256 and 340; eight proof shards are not binary variants'),indent=2)+'\n')
 assert len(samples)==40
 print('CONSOLIDATED proofs='+str(len(proofs))+' endpoint_interior_comparisons='+str(len(samples)))
if __name__=='__main__':main()
