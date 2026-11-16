"""Single-factor existing-config replay on a captured pretrained MoE input/route."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.serving.baseline_common import require_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--routing', type=Path, required=True)
    args = parser.parse_args()
    require_guard(1)
    import torch
    from vllm.model_executor.layers.fused_moe import override_config
    import vllm.model_executor.layers.fused_moe.fused_moe as fused
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    args.out.mkdir(parents=True, exist_ok=False)
    captured = torch.load(args.input, map_location='cpu', weights_only=True)
    identities = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                            sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
                  for key, value in captured.items()}
    tensors = {key: value.cuda() for key, value in captured.items()}
    m = tensors['hidden_states'].shape[0]
    diagnostics = json.loads(args.routing.read_text())
    configs = [row['config'] for row in diagnostics['gemm_samples'] if row['layer'] == 0]
    if not configs or any(row != configs[0] for row in configs):
        raise ValueError('replay requires one observed layer0 config,not a guessed/tuned tile')
    config = configs[0]
    bound = captured['topk_ids'].numel() * config['BLOCK_SIZE_M']
    n_e = torch.bincount(captured['topk_ids'].reshape(-1).long(), minlength=128)
    actual_pad = sum(((int(value) + config['BLOCK_SIZE_M']-1)//config['BLOCK_SIZE_M'])*config['BLOCK_SIZE_M']
                     for value in n_e)
    if actual_pad > bound:
        raise ValueError('route-derived safe padding bound violated')
    observed = [row['actual_padded_rows'] for row in diagnostics['gemm_samples']
                if row['layer'] == 0 and row['decode_iteration'] == 40]
    if not observed or any(value != actual_pad for value in observed):
        raise ValueError('offline route count does not match the observed GPU padding scalar')
    original = fused.invoke_fused_moe_kernel
    signature = inspect.signature(original)
    state = dict(crop=False)

    def invoke(*args, **kwargs):
        values = signature.bind(*args, **kwargs)
        ids = values.arguments['sorted_token_ids']
        length = min(ids.numel(), bound) if state['crop'] else ids.numel()
        values.arguments['sorted_token_ids'] = ids[:length]
        return original(*values.args, **values.kwargs)

    fused.invoke_fused_moe_kernel = invoke
    reference, rows = None, []
    for index in (0, 1, 1, 0):
        state['crop'] = index == 1
        with override_config(config):
            for _ in range(50):
                output = fused_experts(**tensors)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fused_experts(**tensors)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(100):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            actual = output.float().cpu()
            if reference is None:
                reference = actual
            error = (actual-reference).abs()
            cosine = torch.nn.functional.cosine_similarity(actual.reshape(1, -1), reference.reshape(1, -1)).item()
            rows.append(dict(variant=index, config=config, median_ms=statistics.median(samples),
                             min_ms=min(samples), max_ms=max(samples), max_abs_error=error.max().item(),
                             cosine=cosine, bitwise_equal=torch.equal(actual, reference), samples_ms=samples))
            if cosine < .9999:
                raise ValueError('existing-config replay changed output beyond diagnostic tolerance')
    result = dict(diagnostic_only=True, changed_factor='sorted-token launch upper bound only;'
                  'same original kernels/weights/input/route/tile/reduction',
                  route_assignments=captured['topk_ids'].numel(), actual_padded_rows=actual_pad,
                  static_safe_bound=bound,
                  input_identity=identities, order=[0, 1, 1, 0], results=rows,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='One actual pretrained layer/input/top-k route,50warmup+100CUDA-Graph replay event samples per position. '
                        'Repeated layer cache/scheduling differs from full model;not end-to-end performance or utilization. '
                        'Cropped view uses the safe static bound assignments*BLOCK_SIZE_M;'
                        'every routed assignment contributes at most one padded block. '
                        'Both controls create the same view operation before capture;Python dispatch is outside replay. '
                        'Original kernel source/tile/reduction retained;EM specialization/launch bound differs. '
                        'Probe only;no new kernel or production default changed.')
    (args.out / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(m=m, medians_ms=[row['median_ms'] for row in rows])), flush=True)


if __name__ == '__main__':
    main()
