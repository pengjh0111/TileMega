#!/usr/bin/env python3
"""Check the C3(a) no-op control at W=1 without changing any timing sample."""
import csv,hashlib,json,subprocess
from pathlib import Path
from probe_audit import functions
HERE=Path(__file__).resolve().parent

def main():
 out=HERE/'bounded_w1_identity';out.mkdir(exist_ok=True);rows=[]
 for cell in sorted((HERE/'bounded_raw').iterdir()):
  blocks={}
  for arm in ('c2','c3'):
   binary=cell/'bin'/(arm+'__full');dest=out/cell.name;dest.mkdir(exist_ok=True)
   command=['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)]
   with (dest/(arm+'.sass')).open('w') as stream:subprocess.run(command,stdout=stream,check=True)
   (dest/(arm+'.json')).write_text(json.dumps(dict(command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
   blocks[arm]=functions((dest/(arm+'.sass')).read_text())
  assert blocks['c2'].keys()==blocks['c3'].keys()
  for name,body in blocks['c2'].items():
   rows.append(dict(cell=cell.name,function=name,identical=int(body==blocks['c3'][name]),c2_sha256=hashlib.sha256(body.encode()).hexdigest(),c3_sha256=hashlib.sha256(blocks['c3'][name].encode()).hexdigest()))
  print('W1_NOOP',cell.name,all(r['identical'] for r in rows if r['cell']==cell.name),flush=True)
 with (out/'identity.tsv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
