"""Measure what the executor does when the build page cannot hold the row.

`run.py`'s PAGE map lifts the real cells to 8192 because their 4096-wide scale
row is 8192 bytes; this probe builds the same cell at the 1024-byte default and
reads `E2E_PREFETCH` back, so the refusal is measured rather than inferred.
"""
import json,sys,pathlib
REPO=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure
sys.path.insert(0,str(REPO/'docs/experiments/PIPELINE'));import run as runner
cell=REPO/'docs/experiments/PIPELINE/raw/real_s4'
spec=dict(runner.specs('real',4)['prefetch'])
spec['extra']=[x for x in spec['extra'] if not x.startswith('PREFETCH_PAGE_BYTES')]
print(json.dumps(spec['extra']),flush=True)
if measure.build(cell,'real','prefetch_page1024',spec,'sm_89'):raise SystemExit('build')
for i in range(3):
    measure.run(cell,'real',4,'prefetch_page1024',cell/'page_probe',i,0,'probe')
print('PROBE done',flush=True)
