"""Enqueue the eight missing MoE reference trajectories; no timing runs."""
from pathlib import Path
import sys
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
spec=importlib.util.spec_from_file_location('mx8_queue',ROOT/'experiments/mx8/queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def prepare(cell_path):
    cell=q.read(cell_path);base=f'{cell["model"]}_B{cell["batch"]}'
    directory=ROOT/'runs/mx8/routing_build'/base
    binaries={}
    for phase in ('decode','prefill'):
        binary=directory/phase/'plan.so';binary.parent.mkdir(parents=True,exist_ok=True)
        if not Path(str(binary)+'.identity.json').exists():
            with (binary.parent/'BUILD.log').open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'experiments/mx8/build_routes.py'),
                    '--source',cell[phase]['path']+'.cu','--out',str(binary.parent)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        binaries[phase]=binary
    worker=ROOT/'experiments/mx8/capture_routes.py'
    q.submit(ROOT/'runs/mx8/queue','routes_'+base,dict(kind='routing',
        command=[sys.executable,str(worker),'--cell',str(cell_path),'--decode',str(binaries['decode']),
            '--prefill',str(binaries['prefill']),'--out','{out}/result'],
        inputs=[q.ref(p) for p in (cell_path,worker,binaries['decode'],binaries['prefill'],cell['prompt'])]+[cell['tokens']],
        priority=-1,needs_mib=40960,timeout_s=900,after=[]))
    print('queued routing',base,flush=True)


def main():
    cells=[p for p in (ROOT/'runs/mx8/cells').glob('*.json') if q.read(p)['moe'] and q.read(p)['batch']>1]
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(prepare,p):p for p in cells}
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:
                errors.append(dict(cell=str(futures[future]),error=str(error)))
                print('ROUTING BUILD FAILED',futures[future],error,flush=True)
    q.save(ROOT/'runs/mx8/ROUTING_QUEUE.json',dict(scopes=len(cells),errors=errors))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
