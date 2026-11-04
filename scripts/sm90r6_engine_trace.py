#!/usr/bin/env python3
"""I-1 (prompt section 8.1): what a kernel boundary costs a production engine.

An engine's decode step is a sequence of kernels with gaps between them. This
traces that sequence with nsys and reduces it to appendix E's metrics, so the
cost TileMega avoids by staying inside one kernel is measured on the engines
rather than argued.

The unchanged baseline runs with an inherited worker import hook. The actual
GPU worker brackets decode steps 257..320 with cudaProfilerStart/Stop. Only
that range is analyzed, and kernel sequence repetition determines boundaries.
"""
import argparse, csv, hashlib, json, math, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT
import sm90r5_dispatch as dispatch
from sm90r4_inputs import model_dir

ENVS = {'vllm': '/workspace/llm-stack/envs/vllm/bin/python',
        'sglang': '/workspace/llm-stack/envs/sglang/bin/python',
        'trtllm': '/workspace/llm-stack/envs/trtllm/bin/python'}
OUT = ROOT / 'runs/sm90r6/i1'
NSYS = '/usr/local/bin/nsys'
# kernel name -> class, in the order appendix E lists them.
CLASSES = (('allreduce', r'(allreduce|all_reduce|nccl|ncclDevKernel)'),
           ('rope_kv', r'(rope|rotary|kv_cache|reshape_and_cache|cache_kernel)'),
           ('attention', r'(attn|attention|flash|paged|mha|mla)'),
           ('gemm', r'(gemm|gemv|cutlass|matmul|linear|sm90_xmma|wgmma|splitk|s16816|f16f16|nvjet_)'),
           ('norm', r'(rms_?norm|layer_?norm|norm_kernel|triton_red_fused_.*mean_.*pow)'),
           ('activation', r'(silu|gelu|swiglu|act_and_mul)'),
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
    hooks = directory/'hooks'
    hooks.mkdir()
    hook_source = (ROOT/'scripts/sm90r6_trace_hooks/sitecustomize.py').read_bytes()
    (hooks/'sitecustomize.py').write_bytes(hook_source)
    (directory/'protocol.json').write_text(json.dumps(dict(
        schema=2, engine=engine, model=model, batch=batch, capture_steps=[257, 320],
        analysis_steps=[273, 304], hook_sha256=hashlib.sha256(hook_source).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256((ROOT/f'python/tilemega/serving/{engine}_baseline.py').read_bytes()).hexdigest()
    ), indent=2)+'\n')
    report = directory / 'trace'
    baseline = [ENVS[engine], '-u',
                str(ROOT / f'python/tilemega/serving/{engine}_baseline.py'),
                '--model', str(directory / 'checkpoint'),
                '--prompt-ids', str(ROOT / ('configs/prompts/llama3_isl64.json' if model.startswith('llama') else 'configs/prompts/qwen_isl64.json')),
                '--batch', str(batch), '--max-tokens', str(tokens), '--warmup', '1',
                '--repeats', '1', '--round-index', '0', '--load-format', 'auto',
                '--policy', str(ROOT / 'configs/bench/measurement_policy.json'),
                '--out', str(directory / 'job/measurements')]
    checkpoint = [ENVS['vllm'], str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                  '--config', str(ROOT / f'configs/serving/sm90d1/inputs/{model}'),
                  '--out', str(directory / 'checkpoint'), '--seed', '0', '--max-gib', '64',
                  '--', *baseline]
    profiled = [NSYS, 'profile', '--trace=cuda', '--cuda-graph-trace=node',
                '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--trace-fork-before-exec=true', '--wait=all', '--sample=none',
                '--cpuctxsw=none', '-o', str(report), '--'] + checkpoint
    node = '0' if gpu <= 3 else '1'
    command = [ENVS['vllm'], '-m', 'tilemega.bench.gpu_guard',
               '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'),
               '--out', str(directory / 'guard'), '--needs-free-mib', str(70 * 1024),
               '--timeout-s', '7200', '--', 'numactl', f'--cpunodebind={node}', *profiled]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       TILEMEGA_DEVICE_INDEX=str(gpu),
                       TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',
                       TILEMEGA_GPU_PID_MODE='namespace_limited',
                       PYTHONPATH=str(hooks)+os.pathsep+str(ROOT / 'python'),
                       TILEMEGA_I1_ENGINE=engine, TILEMEGA_I1_CAPTURE_DIR=str(directory / 'capture'),
                       TILEMEGA_TIMING='1', TILEMEGA_NUMA_NODE=node,
                       TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8', VLLM_USE_V1='1')
    for key in ('TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE',
                'TILEMEGA_NUMA_CONCURRENCY_VERIFIED'):
        environment.pop(key, None)
    (directory / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
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


def steps(trace_rows, want=32, captured=64):
    """Use the captured kernel sequence, never a TPOT-sized time window."""
    if not trace_rows or len(trace_rows) % captured:
        raise ValueError('capture is not a whole number of 64 kernel sequences')
    period = len(trace_rows)//captured
    names = [row[0] for row in trace_rows]
    template = names[:period]
    if any(names[i*period:(i+1)*period] != template for i in range(captured)):
        raise ValueError('kernel sequence does not repeat for all 64 captured decode steps')
    first = (captured-want)//2
    if first < 0 or first+want >= captured:
        raise ValueError('middle windows need the following step start')
    return [(i*period, (i+1)*period, trace_rows[i*period][1],
             trace_rows[(i+1)*period][1]) for i in range(first, first+want)]


def capture_complete(directory):
    complete = []
    for path in (directory/'capture').glob('worker-*.jsonl'):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        markers = [(e['event'], e['decode_step']) for e in events if e['event'] != 'installed']
        if markers:
            if markers != [('start', 257), ('stop', 321)]:
                raise ValueError(f'incomplete profiler range: {path}')
            complete.append(path)
    if len(complete) != 1:
        raise ValueError('I-1 TP1 requires exactly one complete worker profiler range')


def gemm_groups(block):
    groups = []
    for index, row in enumerate(block):
        name = row[0].lower()
        if classify(name) != 'gemm':
            continue
        if 'splitkreduce' in name:
            if (not groups or groups[-1][-1][0] != index-1 or
                    'splitk' not in groups[-1][0][1][0].lower()):
                raise ValueError('split-K reduction has no adjacent producer')
            groups[-1].append((index, row))
        else:
            groups.append([(index, row)])
    return groups


def gemm_metrics(trace_rows, windows, config, model, engine, batch):
    layers = config['num_hidden_layers']; h = config['hidden_size']
    hq = config['num_attention_heads']; hkv = config.get('num_key_value_heads', hq)
    d = config.get('head_dim', h//hq); inter = config['intermediate_size']
    order = ['qkv', 'o', 'gate_up', 'down']*layers + ['lm_head']
    weights = dict(qkv=2*h*(hq+2*hkv)*d, o=2*h*hq*d,
                   gate_up=4*h*inter, down=2*h*inter, lm_head=2*h*config['vocab_size'])
    rates = {family: [] for family in weights}
    for begin, end, _, _ in windows:
        gemms = gemm_groups(trace_rows[begin:end])
        if len(gemms) != len(order):
            raise ValueError(f'GEMM sequence requires validated operator mapping: '
                             f'{len(gemms)} kernels versus {len(order)} operators')
        for family, group in zip(order, gemms):
            # Include an operator's split-K reduction, excluding launch gaps.
            duration = sum(finish-start for _, (_, start, finish) in group)
            rates[family].append(weights[family]/duration)
    return [dict(engine=engine, model=model, batch=batch, family=family,
                 observations=len(values), weight_bytes=weights[family],
                 median_gbps=statistics.median(values),
                 p10_gbps=sorted(values)[math.ceil(.1*len(values))-1])
            for family, values in rates.items()]


def metrics(trace_rows, windows, floor_bytes, bw_gbps):
    """Appendix E, per step, reported as the median over the windows."""
    per_step = []
    for begin, end, low, high in windows:
        block = trace_rows[begin:end]
        if not block:
            continue
        step_ns = high - low
        if any(start < low or finish > high for _, start, finish in block):
            raise ValueError('kernel crosses a detected decode boundary')
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
    p.add_argument('--model', required=True, choices=['qwen3_1_7b', 'llama3_1_8b'])
    p.add_argument('--batch', type=int, required=True, choices=[1, 16])
    p.add_argument('--tokens', type=int, default=1024, choices=[1024])
    p.add_argument('--gpu', type=int, required=True, choices=[0, 1, 2, 3, 4, 5, 7])
    p.add_argument('--bytes', type=float, required=True, help='appendix D bytes_step')
    p.add_argument('--bw-gbps', type=float, required=True, help='H1 measured maximum')
    p.add_argument('--out', type=Path, default=OUT/'protocol_v2')
    p.add_argument('--analyze', type=Path, help='reanalyze an existing guarded capture')
    a = p.parse_args()
    gpu = a.gpu
    if a.analyze:
        directory = a.analyze.resolve()
        protocol = json.loads((directory/'protocol.json').read_text())
        if any(protocol[key] != getattr(a, key) for key in ('engine', 'model', 'batch')):
            raise ValueError('analysis arguments differ from capture identity')
        code = json.loads((directory/'guard/guard_result.json').read_text())['code']
        report = directory/'trace.nsys-rep'
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f'{a.engine}-{a.model}-B{a.batch}-', dir=a.out.resolve()))
        gpu, handle = dispatch.claim('trial', directory.name, only=a.gpu)
        try:
            code, report, directory = trace(a.engine, a.model, a.batch, a.tokens, gpu, directory)
        finally:
            handle.close()
        (directory/'result.json').write_text(json.dumps(dict(returncode=code))+'\n')
    if code:
        return code
    capture_complete(directory)
    rows = kernels(report, directory)
    windows = steps(rows)
    result = metrics(rows, windows, a.bytes, a.bw_gbps)
    config = json.loads((model_dir(a.model)/'config.json').read_text())
    gemms = gemm_metrics(rows, windows, config, a.model, a.engine, a.batch)
    frozen = [r for r in csv.DictReader((ROOT/'docs/results/R16_final_baselines.csv').open())
              if r['model'] == a.model and int(r['batch']) == a.batch and
              r['system'] == {'trtllm': 'tensorrt-llm'}.get(a.engine, a.engine)]
    if len(frozen) != 1:
        raise ValueError('I-1 needs one frozen TP1 reference')
    frozen_ms = float(frozen[0]['tpot_ms'])
    result.update(engine=a.engine, model=a.model, batch=a.batch, gpu=gpu,
                  tpot_under_nsys_ms=result['step_us']/1000, frozen_tpot_ms=frozen_ms,
                  overhead_pct=100*(result['step_us']/1000/frozen_ms-1),
                  gap_share=result['gap_us']/result['step_us'],
                  floor_share=result['floor_us']/result['step_us'],
                  excess_share=result['in_kernel_excess_us']/result['step_us'],
                  classes_share={k: v/result['busy_us'] for k,v in result['classes'].items()},
                  bytes_step=a.bytes, bw_gbps=a.bw_gbps, raw_directory=str(directory))
    (directory/'windows.json').write_text(json.dumps(windows)+'\n')
    mapping = [dict(operator_index=i, kernels=[dict(position=p, name=r[0]) for p,r in group])
               for i,group in enumerate(gemm_groups(rows[windows[0][0]:windows[0][1]]))]
    (directory/'gemm_mapping.json').write_text(json.dumps(mapping, indent=2)+'\n')
    result['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['gemm_duration'] = 'sum of multiply and adjacent split-K reduction kernel durations'
    (directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (directory/'gemm.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gemms[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(gemms)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
