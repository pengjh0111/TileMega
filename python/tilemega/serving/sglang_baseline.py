"""Independent SGLang baseline over frozen token IDs; imports no TileMega code.

R16 M-1 / appendix B: offline `sgl.Engine` with the radix cache off, CUDA
graphs covering B <= 16, token-ID input and greedy decoding that ignores EOS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--batch", choices=("1", "2", "4", "8", "16", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measurement_policy import TimingPolicy
    from workload import validate_prompt_ids
    from baseline_common import measure
    policy = TimingPolicy(args.policy)
    ids = json.loads(args.prompt_ids.read_text())
    batch_limit = 16 if args.batch == "all" else int(args.batch)
    prompt_len = validate_prompt_ids(ids, batch_limit)
    if args.max_tokens < 2:
        raise ValueError("max-tokens must be at least 2 (TPOT needs N > 1)")

    import sglang as sgl
    engine = sgl.Engine(model_path=str(args.model), dtype="bfloat16", disable_radix_cache=True,
                        skip_tokenizer_init=True, max_running_requests=16,
                        # SGLang 0.5.14 splits the R16 `cuda_graph_max_bs` per phase.
                        cuda_graph_max_bs_decode=16,
                        mem_fraction_static=args.mem_fraction, random_seed=0,
                        context_length=prompt_len + args.max_tokens + 64)
    server_args = getattr(engine, "server_args", None)
    meta = {"system": "sglang", "model": str(args.model), "sglang_version": sgl.__version__,
            "attention_backend": getattr(server_args, "attention_backend", None),
            "decode_attention_backend": getattr(server_args, "decode_attention_backend", None),
            "cuda_graph_max_bs_decode": getattr(server_args, "cuda_graph_max_bs_decode", None),
            "cuda_graph_bs_decode": getattr(server_args, "cuda_graph_bs_decode", None),
            "disable_cuda_graph": getattr(server_args, "disable_cuda_graph", None),
            "disable_decode_cuda_graph": getattr(server_args, "disable_decode_cuda_graph", None)}

    def generate(batch, count):
        outputs = engine.generate(input_ids=ids[:batch], sampling_params={
            "temperature": 0.0, "max_new_tokens": count, "ignore_eos": True})
        if len(outputs) != batch:
            raise AssertionError("SGLang returned a different number of requests")
        return [list(row["output_ids"]) for row in outputs]

    try:
        batches = (1, 2, 4, 8, 16) if args.batch == "all" else (int(args.batch),)
        measure(args.out, batches, args.max_tokens, args.warmup, args.repeats, policy, generate, meta)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
