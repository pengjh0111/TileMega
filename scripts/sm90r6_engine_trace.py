#!/usr/bin/env python3
"""I-1 (prompt section 8.1): what a kernel boundary costs a production engine.

An engine's decode step is a sequence of kernels with gaps between them. This
traces that sequence with nsys and reduces it to appendix E's metrics, so the
cost TileMega avoids by staying inside one kernel is measured on the engines
rather than argued.

The baseline scripts are not modified (section 8.1 item 1). Two consequences,
both deliberate:
  * cudaProfilerStart/Stop cannot be injected from outside, so the whole
    generation is traced and the token count is reduced instead to keep the
    trace manageable. Step boundaries come from the kernel sequence's own
    periodicity either way (item 3), so nothing depends on the capture range.
  * nsys adds overhead; the TPOT measured under it is reported next to the
    frozen value so the reader can see how much (U9).
"""
import argparse, csv, json, os, re, statistics, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT

ENVS = {'vllm': '/workspace/llm-stack/envs/vllm/bin/python',
        'sglang': '/workspace/llm-stack/envs/sglang/bin/python',
        'trtllm': '/workspace/llm-stack/envs/trtllm/bin/python'}
OUT = ROOT / 'runs/sm90r6/i1'
NSYS = '/usr/local/bin/nsys'
# kernel name -> class, in the order appendix E lists them.
CLASSES = (('allreduce', r'(allreduce|all_reduce|nccl|ncclDevKernel)'),
           ('attention', r'(attn|attention|flash|paged|mha|mla)'),
           ('gemm', r'(gemm|cutlass|matmul|linear|sm90_xmma|wgmma|splitk|s16816|f16f16)'),
           ('norm', r'(rms_?norm|layer_?norm|norm_kernel)'),
           ('rope_kv', r'(rope|rotary|kv_cache|reshape_and_cache|cache_kernel)'),
           ('activation', r'(silu|gelu|swiglu|act_and_mul|elementwise)'),
           ('sampling', r'(argmax|sample|topk|top_k|softmax_kernel|logits)'))


def classify(name):
    lowered = name.lower()
    for label, pattern in CLASSES:
        if re.search(pattern, lowered):
            return label
    return 'other'


def trace(engine, model, batch, tokens, gpu, directory):
    """Run one guarded baseline under nsys; returns the sqlite export path."""
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / 'trace'
    baseline = [ENVS[engine], '-u',
                str(ROOT / f'python/tilemega/serving/{engine}_baseline.py'),
                '--model', str(directory / 'checkpoint'),
                '--prompt-ids', str(ROOT / 'configs/prompts/qwen_isl64.json'),
                '--batch', str(batch), '--max-tokens', str(tokens), '--warmup', '1',
                '--repeats', '1', '--round-index', '0', '--load-format', 'auto',
                '--policy', str(ROOT / 'configs/bench/measurement_policy.json'),
                '--out', str(directory / 'job/measurements')]
    checkpoint = [ENVS['vllm'], str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                  '--config', str(ROOT / f'configs/serving/sm90d1/inputs/{model}'),
                  '--out', str(directory / 'checkpoint'), '--seed', '0', '--max-gib', '64',
                  '--', *baseline]
    profiled = [NSYS, 'profile', '--trace=cuda', '--cuda-graph-trace=node',
                '--force-overwrite=true', '-o', str(report), '--'] + checkpoint
    node = '0' if gpu <= 3 else '1'
    command = [ENVS['vllm'], '-m', 'tilemega.bench.gpu_guard',
               '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'),
               '--out', str(directory / 'guard'), '--needs-free-mib', str(70 * 1024),
               '--timeout-s', '7200', '--', 'numactl', f'--cpunodebind={node}', *profiled]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       TILEMEGA_DEVICE_INDEX=str(gpu),
                       TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',
                       TILEMEGA_GPU_PID_MODE='namespace_limited',
                       PYTHONPATH=str(ROOT / 'python'), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8', VLLM_USE_V1='1')
    environment.pop('TILEMEGA_GPU_LOCK_HELD', None)
    with (directory / 'run.log').open('w') as log:
        code = subprocess.call(command, cwd=str(ROOT), env=environment,
                               stdout=log, stderr=subprocess.STDOUT)
    return code, report.with_suffix('.nsys-rep'), directory


def kernels(report, directory):
    """Kernel name, start and end in ns, from the nsys report."""
    sqlite = directory / 'trace.sqlite'
    subprocess.run([NSYS, 'export', '--type', 'sqlite', '--force-overwrite=true',
                    '-o', str(sqlite), str(report)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import sqlite3
    connection = sqlite3.connect(str(sqlite))
    # TRT-LLM's llmapi launcher re-execs under mpirun, and nsys follows only
    # the process it started, so the export can carry no kernel table at all.
    # That is a traceability limit of the engine's launcher, not an empty step.
    present = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='CUPTI_ACTIVITY_KIND_KERNEL'").fetchone()
    if not present:
        connection.close()
        return []
    rows = connection.execute(
        'SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k '
        'JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start').fetchall()
    connection.close()
    return [(name, int(start), int(end)) for name, start, end in rows]


def steps(trace_rows, step_ns, want=32):
    """Split the kernel stream into decode steps of the measured step time.

    The first version anchored on the most frequent kernel name, which is wrong
    for an engine that packs a step into one CUDA graph node: it reported one
    kernel per step and an 18 us step for a model whose DRAM floor alone is
    4.7 ms. The generation rate is the ground truth instead - a decode step is
    one token - so the stream is cut at that period, taken from the middle of
    the trace where the engine is in steady state.
    """
    if not trace_rows or step_ns <= 0:
        return []
    begin, end = trace_rows[0][1], trace_rows[-1][2]
    total = (end - begin) / step_ns
    if total < 4:
        return []
    # Anchor on the middle of the kernel sequence, not the middle of the wall
    # clock: the trace also covers model load and the untimed warmup, so the
    # time midpoint can fall in an idle gap where no kernel runs at all.
    middle = trace_rows[len(trace_rows) // 2][1]
    first = middle - (want / 2) * step_ns
    windows = []
    index = 0
    for step in range(want):
        low = first + step * step_ns
        high = low + step_ns
        if low < begin or high > end:
            continue
        while index < len(trace_rows) and trace_rows[index][1] < low:
            index += 1
        start_index = index
        cursor = index
        while cursor < len(trace_rows) and trace_rows[cursor][1] < high:
            cursor += 1
        if cursor > start_index:
            windows.append((start_index, cursor, low, high))
    return windows


def metrics(trace_rows, windows, floor_bytes, bw_gbps):
    """Appendix E, per step, reported as the median over the windows."""
    per_step = []
    for begin, end, low, high in windows:
        block = trace_rows[begin:end]
        if not block:
            continue
        step_ns = high - low
        # The union of the kernel intervals: overlap from programmatic
        # dependent launch is counted once, as appendix E requires.
        union, merged = 0, []
        for _, start, finish in sorted((r for r in block), key=lambda r: r[1]):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], finish)
            else:
                merged.append([start, finish])
        for start, finish in merged:
            union += finish - start
        by_class = {}
        for name, start, finish in block:
            by_class[classify(name)] = by_class.get(classify(name), 0) + (finish - start)
        floor_ns = floor_bytes / (bw_gbps * 1e9) * 1e9
        per_step.append(dict(kernels=len(block), step_us=step_ns / 1e3, busy_us=union / 1e3,
                             gap_us=(step_ns - union) / 1e3, floor_us=floor_ns / 1e3,
                             in_kernel_excess_us=(union - floor_ns) / 1e3,
                             classes={k: v / 1e3 for k, v in by_class.items()}))
    if not per_step:
        return None
    pick = lambda key: statistics.median(s[key] for s in per_step)
    classes = {}
    for label, _ in CLASSES + (('other', ''),):
        values = [s['classes'].get(label, 0.0) for s in per_step]
        if any(values):
            classes[label] = statistics.median(values)
    return dict(steps=len(per_step), kernels_per_step=pick('kernels'), step_us=pick('step_us'),
                busy_us=pick('busy_us'), gap_us=pick('gap_us'), floor_us=pick('floor_us'),
                in_kernel_excess_us=pick('in_kernel_excess_us'), classes=classes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True, choices=sorted(ENVS))
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--tokens', type=int, default=384)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--bytes', type=float, required=True, help='section 9.3 bytes_step')
    p.add_argument('--bw-gbps', type=float, default=3234.65, help='H1 measured maximum')
    p.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R6_kernel_boundary.csv')
    a = p.parse_args()
    directory = OUT / f'{a.engine}-{a.model}-B{a.batch}'
    code, report, directory = trace(a.engine, a.model, a.batch, a.tokens, a.gpu, directory)
    # nsys writes the report even when the traced process fails, and the
    # measurement protocol's own policy checks can fail a run whose kernels
    # are perfectly usable; the trace is what I-1 needs, so it is used when it
    # exists and the exit code is recorded beside the metrics.
    if not report.exists():
        print(json.dumps({'engine': a.engine, 'model': a.model, 'batch': a.batch,
                          'returncode': code, 'error': 'no trace'}), flush=True)
        return 1
    rows = kernels(report, directory)
    measured = directory / f'job/measurements/B{a.batch}/measurements.json'
    step_ns = (json.loads(measured.read_text())['tpot_seconds'] * 1e9
               if measured.exists() else 0)
    if not rows:
        print(json.dumps({'engine': a.engine, 'model': a.model, 'batch': a.batch,
                          'error': 'nsys captured no kernels: the engine re-execs under its '
                                   'own launcher, so the traced process is not the one that '
                                   'runs them', 'baseline_returncode': code}), flush=True)
        return 2
    windows = steps(rows, step_ns)
    result = metrics(rows, windows, a.bytes, a.bw_gbps)
    if result is None:
        print(json.dumps({'engine': a.engine, 'error': 'no usable step window',
                          'kernels': len(rows), 'step_ns': step_ns}), flush=True)
        return 1
    under_nsys = step_ns / 1e6 if step_ns else None
    row = dict(engine=a.engine, model=a.model, batch=a.batch, tp=1, baseline_returncode=code,
               kernels_per_step=f"{result['kernels_per_step']:.1f}",
               step_us=f"{result['step_us']:.2f}", busy_us=f"{result['busy_us']:.2f}",
               gap_us=f"{result['gap_us']:.2f}", floor_us=f"{result['floor_us']:.2f}",
               in_kernel_excess_us=f"{result['in_kernel_excess_us']:.2f}",
               gap_share=f"{result['gap_us'] / result['step_us']:.4f}",
               excess_share=f"{result['in_kernel_excess_us'] / result['step_us']:.4f}",
               floor_share=f"{result['floor_us'] / result['step_us']:.4f}",
               steps_used=result['steps'], total_kernels=len(rows),
               tpot_under_nsys_ms=f'{under_nsys:.4f}' if under_nsys else '',
               bytes_step=int(a.bytes), bw_gbps=a.bw_gbps,
               classes=json.dumps({k: round(v, 2) for k, v in result['classes'].items()}))
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if a.csv.exists():
        existing = [r for r in csv.DictReader(open(a.csv))
                    if (r['engine'], r['model'], r['batch']) != (a.engine, a.model, str(a.batch))]
    with a.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerows(existing + [row])
    print(json.dumps(row), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
