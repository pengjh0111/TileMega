#!/usr/bin/env python3
"""Pin a Llama B16 50-process protocol case to exact serving binaries."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefill-so',type=Path,required=True);p.add_argument('--decode-so',type=Path,required=True)
p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
config=json.loads(a.config.read_text());prefill=a.prefill_so.resolve();decode=a.decode_so.resolve()
manifest=json.loads(Path(str(decode)+'.plan.json').read_text())
if manifest.get('pg')!='pages':raise ValueError('protocol verification requires an explicit paged decode plan')
case=dict(model=str(Path(config['model']['path']).resolve()),batch=16,steps=64,
          prompt_ids=str(Path(config['workload']['prompts']).resolve()),
          reference_prefill=str(prefill),reference_decode=str(decode),
          prefill=str(prefill),decode=str(decode),
          binary_sha256=dict(prefill=hashlib.sha256(prefill.read_bytes()).hexdigest(),
                             decode=hashlib.sha256(decode.read_bytes()).hexdigest()),
          arms=[dict(label='L1_separate',prefill=str(prefill),decode=str(decode),
                     modes=['L1'],decode_loop=False),
                dict(label='L2_loop',prefill=str(prefill),decode=str(decode),modes=['L2']),
                dict(label='L2_separate',prefill=str(prefill),decode=str(decode),
                     modes=['L2'],decode_loop=False),
                dict(label='L2_loop_no_phase',prefill=str(prefill),decode=str(decode),
                     modes=['L2'],env={'TILEMEGA_KPHASE_MASK':'0'})])
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps([case],indent=2)+'\n')
print(a.out)
