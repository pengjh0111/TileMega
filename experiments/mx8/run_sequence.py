"""Qualify or time one AB4c arm; never remeasure the frozen main reference."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

# The orchestration script named queue.py must not shadow Python's queue.
sys.path = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
import torch

from tilemega.build.identity import sha, verify
from tilemega.serving.engine import ServingEngine


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--decode', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', choices=('qualify', 'measure'), required=True)
    parser.add_argument('--hf', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cell = json.loads(args.cell.read_text())
    identity = verify(args.decode)
    reference = json.loads(Path(cell['tokens']['path']).read_text())
    if isinstance(reference, dict):
        reference = reference['tokens']
    prompt = json.loads(Path(cell['prompt']).read_text())
    if len(prompt) >= cell['batch']:
        prompt = prompt[:cell['batch']]
    else:
        if len(prompt) != 1:
            raise ValueError('prompt cannot be expanded to requested batch')
        prompt *= cell['batch']
    prompts = torch.tensor(prompt, dtype=torch.int32)
    if sha(cell['tokens']['path']) != cell['tokens']['sha256']:
        raise ValueError('frozen token evidence changed')
    result = dict(task=args.task, PID=os.getpid(), physical_gpu=os.environ.get('TILEMEGA_DEVICE_INDEX'),
        cell_sha256=sha(args.cell), artifact_id=identity['artifact_id'], variant=identity['plan']['sequence_mode'],
        baseline_timing='frozen main experiment; no AB4b measurement', reference_tokens=len(reference[0]),
        runs=[], start=time.time())
    with ServingEngine(cell['model_dir'], cell['prefill']['path'], args.decode, cell['batch'],
            mode='L1', prefill_mode='L1', decode_loop=True, step_events=False) as engine:
        if args.task == 'qualify':
            a = engine.generate(prompts).tokens
            engine.decode_loop = False
            b = engine.generate(prompts).tokens
            exact = bool(torch.equal(a, b))
            ref = torch.tensor(reference, dtype=torch.int32)
            frozen_equal = bool(torch.equal(a[:, :ref.shape[1]], ref))
            result.update(single_loop_exact_1024=exact, frozen_reference_exact=frozen_equal,
                tokens_sha256=__import__('hashlib').sha256(a.numpy().tobytes()).hexdigest())
            write(args.out / 'tokens_1024.json', a.tolist())
            write(args.out / 'tokens_hf.json', a[:, :256].tolist())
            if not exact or not frozen_equal:
                write(args.out / 'FAILED.json', result)
                raise ValueError('AB4c generation differs from single-step or frozen main reference')
        else:
            for count in (1024, 1):
                for run in range(4):
                    generation = engine.generate(prompts, count)
                    if not torch.equal(generation.tokens[:, :min(count, len(reference[0]))],
                            torch.tensor(reference, dtype=torch.int32)[:, :count]):
                        raise ValueError('timed generation differs from frozen reference')
                    result['runs'].append(dict(count=count, run=run, warmup=run == 0,
                        e2e_ms=generation.e2e_ms, ttft_ms=generation.ttft_ms))
            full = [r for r in result['runs'] if r['count'] == 1024 and not r['warmup']]
            first = [r for r in result['runs'] if r['count'] == 1 and not r['warmup']]
            e2e = statistics.median(r['e2e_ms'] for r in full)
            first_ms = statistics.median(r['e2e_ms'] for r in first)
            result.update(e2e_ms=e2e, n1_ms=first_ms, tpot_ms=(e2e-first_ms)/1023,
                event_tpot_ms=statistics.median((r['e2e_ms']-r['ttft_ms'])/1023 for r in full),
                ttft_ms=statistics.median(r['ttft_ms'] for r in full),
                output_tokens_per_second=cell['batch']*1024/(e2e/1000),
                formula='(median(E2E_N1024)-median(E2E_N1))/1023')
    if args.hf:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        write(args.out / 'prompt.json', prompt)
        command = [sys.executable, '-m', 'tilemega.serving.hf_check', '--model', cell['model_dir'],
            '--prompt-ids', str(args.out / 'prompt.json'), '--generated', str(args.out / 'tokens_hf.json'),
            '--out', str(args.out / 'hf.json'), '--skip-free-greedy']
        if cell['moe']:
            command += ['--experts-implementation', 'eager']
        subprocess.run(command, check=True)
        hf = json.loads((args.out / 'hf.json').read_text())
        result['hf'] = hf
        if not hf['pass']:
            write(args.out / 'FAILED.json', result)
            raise ValueError('AB4c HF teacher-forcing check failed')
    result.update(end=time.time(), passed=True)
    write(args.out / 'RESULT.json', result)
    print(json.dumps(dict(pass_=True, result=str(args.out / 'RESULT.json'), task=args.task)))


if __name__ == '__main__':
    main()
