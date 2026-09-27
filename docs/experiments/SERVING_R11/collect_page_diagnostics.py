#!/usr/bin/env python3
"""Collect the four user-selected B=1/16 PG-1 chain diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
RUNTIME_PYTHON = Path(os.environ.get(
    'TILEMEGA_PYTHON', '/root/venvs/tilemega-torch213-cu126/bin/python'))


def run(argv: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + environment.get('PYTHONPATH', '')
    with log.open('w') as stream:
        status = subprocess.run(argv, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT).returncode
    if status:
        raise RuntimeError(f'{argv[0]} exited {status}; see {log}')


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if not RUNTIME_PYTHON.is_file():
        raise FileNotFoundError(f'serving Python environment: {RUNTIME_PYTHON}')
    cases=json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        model='llama' if 'llama' in case['model'] else 'qwen3'
        name=f'{model}_B{case["batch"]}'
        cell=args.out/name;cell.mkdir(parents=True,exist_ok=True)
        library=args.diagnostics/name/'trace.so'
        if not library.exists():raise FileNotFoundError(library)
        # Match the measured R10 winner to its saved CG; floor evaluation is
        # performed for every decode past value rather than interpolating three points.
        floor_steps=cell/'floor_steps.tsv'
        if not floor_steps.exists():
            source=Path('/root/r11_work/r10_control/plans')/f'{model}_decode_B{case["batch"]}'
            winner=(source/'plan.so.cu').read_bytes()
            matching=[rank for rank in (1,2,3) if
                      (source/f'plan.so.top{rank}.candidate.so.cu').is_file() and
                      (source/f'plan.so.top{rank}.candidate.so.cu').read_bytes()==winner]
            if len(matching)!=1:raise RuntimeError(f'cannot match winner CG: {name}')
            cg=source/f'plan.so.top{matching[0]}.mlir'
            tool=Path('/root/r11_work/build/tools/tilemega')
            target=ROOT/'docs/experiments/SERVING_R11/r10_control/calibration/target_serving.json'
            run([str(tool),'inspect','request-floor',str(cg),str(target),
                 str(case['batch']),'64','1086',str(cell/'floor.json'),str(floor_steps)],
                cell/'floor.log')
            (cell/'floor_source.json').write_text(json.dumps(dict(cg=str(cg),
                sha256=hashlib.sha256(cg.read_bytes()).hexdigest(),target=str(target)),indent=2)+'\n')
        common=['--model',case['model'],'--prefill-so',case['prefill'],
                '--decode-so',str(library),'--batch',str(case['batch'])]
        if not (cell/'chain/analysis.tsv').exists():
            run([str(RUNTIME_PYTHON),'-m','tilemega.serving.trace',*common,
                 '--past','575','--launches','32','--out',str(cell/'trace_v2')],cell/'trace.log')
            run([sys.executable,str(ROOT/'docs/experiments/TRACE_V2/analyze.py'),
                 str(cell/'trace_v2'),'--source',str(Path(case['decode']).parent/'plan.so.cu'),
                 '--window','1','--out',str(cell/'chain')],cell/'analyze.log')
        if not (cell/'request/page_trace.tsv').exists():
            run([str(RUNTIME_PYTHON),str(ROOT/'docs/experiments/SERVING_R11/measure_page_chain.py'),
                 *common,'--prompt-ids',case['prompt_ids'],'--steps','1024',
                 '--mode','L2','--out',str(cell/'request')],cell/'request.log')
        run([sys.executable,str(ROOT/'docs/experiments/SERVING_R11/analyze_page_chain.py'),
             '--page-trace',str(cell/'request/page_trace.tsv'),
             '--model',model,'--batch',str(case['batch']),
             '--floor-steps',str(floor_steps),
             '--chain-analysis',str(cell/'chain/analysis.tsv'),
             '--out',str(cell/'report')],cell/'page_chain.log')
        print(name, (cell/'report/page_chain_summary.json').read_text(),flush=True)


if __name__=='__main__':main()
