#!/usr/bin/env python3
"""Reanalyze the complete I1 capture matrix and bind its floors to H1."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from estimate_r16_mbu import dense_payload
from sm90r4_inputs import model_dir
from sm90r6_hw_bound import peaks

ROOT = Path(__file__).resolve().parents[1]
MATRIX = {(engine, model, batch) for engine in ('vllm','sglang','trtllm')
          for model in ('qwen3_1_7b','llama3_1_8b') for batch in (1,16)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(root):
    selected = {}
    for path in sorted(root.glob('*/protocol.json')):
        directory = path.parent
        guard = directory/'guard/guard_result.json'
        if not guard.exists() or json.loads(guard.read_text())['code'] != 0:
            continue
        protocol = json.loads(path.read_text())
        key = tuple(protocol[k] for k in ('engine','model','batch'))
        if key not in MATRIX or protocol['capture_steps'] != [257,320]:
            raise ValueError(f'unexpected capture protocol: {directory}')
        if key in selected:
            raise ValueError(f'multiple accepted captures for {key}; select an explicit root')
        selected[key] = directory
    if set(selected) != MATRIX:
        raise ValueError(f'incomplete I1 matrix: {sorted(MATRIX-set(selected))}')
    return selected


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peaks',type=Path)
    group.add_argument('--interim-bw',type=float)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--gemm-csv',type=Path,required=True)
    args=parser.parse_args()
    if args.peaks:
        bandwidth,_=peaks(args.peaks)
        peak_sha=sha(args.peaks)
    else:
        bandwidth=args.interim_bw;peak_sha='pending complete H1'
        for path in (args.csv,args.gemm_csv):
            if not path.resolve().is_relative_to(ROOT/'runs'):
                parser.error('interim normalization must stay under runs/')
    if bandwidth <= 0:
        parser.error('bandwidth must be positive')
    selected=select(args.root)
    rows=[];gemms=[]
    metrics=('kernels_per_step','step_us','busy_us','gap_us','floor_us',
             'in_kernel_excess_us','gap_share','floor_share','excess_share',
             'tpot_under_nsys_ms','full_run_tpot_ms','frozen_tpot_ms','overhead_pct')
    for (engine,model,batch),directory in sorted(selected.items()):
        measurement=json.loads((directory/f'job/measurements/B{batch}/measurements.json').read_text())
        config=json.loads((model_dir(model)/'config.json').read_text())
        payload=dense_payload(config,batch)
        prompt=ROOT/'configs/prompts'/('llama3_isl64.json' if model.startswith('llama') else 'qwen_isl64.json')
        if measurement['prompt_ids_sha256'] != sha(prompt):
            raise ValueError(f'capture uses a different frozen prompt: {directory}')
        command=[sys.executable,str(ROOT/'scripts/sm90r6_engine_trace.py'),
                 '--engine',engine,'--model',model,'--batch',str(batch),
                 '--gpu',str(measurement['gpu_index']),
                 '--bytes',str(payload['logical_bytes_per_decode']),
                 '--bw-gbps',str(bandwidth),'--analyze',str(directory)]
        with (directory/'final_analysis.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        summary=json.loads((directory/'summary.json').read_text())
        row=dict(engine=engine,model=model,batch=batch,gpu=measurement['gpu_index'],
                 evidence='verified trace; inferred logical-byte floor',steps=summary['steps'],
                 normalization='complete H1' if args.peaks else 'interim H1',
                 bw_gbps=bandwidth,peaks_sha256=peak_sha,
                 **{key:summary[key] for key in metrics})
        row['full_run_overhead_pct']=100*(summary['full_run_tpot_ms']/summary['frozen_tpot_ms']-1)
        for label in ('gemm','attention','norm','rope_kv','activation','allreduce','sampling','other'):
            row[label+'_us']=summary['classes'].get(label,0)
            row[label+'_busy_share']=summary['classes_share'].get(label,0)
        row.update(prompt_sha256=measurement['prompt_ids_sha256'],
                   checkpoint_sha256=measurement['checkpoint']['checkpoint_sha256'],
                   acceptance=measurement['acceptance'],analysis_sha256=summary['analysis_sha256'],
                   trace_sha256=sha(directory/'trace.nsys-rep'),raw_directory=str(directory))
        rows.append(row)
        with (directory/'gemm.csv').open() as stream:
            family_rows=list(csv.DictReader(stream))
        if {r['family'] for r in family_rows} != {'qkv','o','gate_up','down','lm_head'}:
            raise ValueError('I1 GEMM mapping is incomplete')
        for r in family_rows:
            r.update(evidence='verified duration; inferred logical-byte rate',
                     duration_definition=summary['gemm_duration'],raw_directory=str(directory))
            gemms.append(r)
    write_csv(args.csv,rows);write_csv(args.gemm_csv,gemms)
    print(json.dumps(dict(cells=len(rows),gemm_rows=len(gemms),bandwidth_gbps=bandwidth,
                          normalization='complete H1' if args.peaks else 'interim H1')))


if __name__=='__main__':
    main()
