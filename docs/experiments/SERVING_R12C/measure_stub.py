#!/usr/bin/env python3
"""Fixed-geometry compiler placeholder; this intentionally measures nothing."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a,ignored=p.parse_known_args()
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'measurements.json').write_text(json.dumps(dict(modes={'L2':{'mean_ms':0}},placeholder=True,ignored=ignored))+'\n')
print('fixed-geometry placeholder (no GPU measurement)')
