"""Independent torch.compile baseline over frozen token IDs; imports no TileMega code.

R16 M-1 / appendix B: HF transformers with a static KV cache, the forward
compiled with mode="reduce-overhead" and fullgraph=True, greedy generation of
exactly N tokens. If fullgraph capture fails, the error is recorded and the
run continues with fullgraph=False (marked in the results). Compilation and
graph capture happen in the warmup runs.

transformers 5.x: the compile goes through `generation_config.compile_config`
instead of replacing `model.forward`. Its StaticLayer allocates the KV cache
lazily inside the first forward; a manually compiled forward captures that
allocation in the CUDA graph and the next replay overwrites the cache
(measured: "accessing tensor output of CUDAGraphs that has been overwritten").
generate() runs the prefill eagerly and compiles the decode forward.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
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
    parser.add_argument("--mode", default="reduce-overhead")
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

    import torch
    import transformers
    # MetaX torch routes F.scaled_dot_product_attention through a Python
    # wrapper that queries the SDP backend flags; Dynamo cannot trace those
    # bool-returning calls and fullgraph capture fails. The flags do not change
    # during a run, so they are declared constant: kernel selection is the
    # same as eager. Recorded in the results as `sdp_flags_constant`.
    # The wrapper's flash branch then calls the flash_attn C extension, which
    # Dynamo cannot trace either; `torch.disable_flash_attn` (a MetaX switch)
    # keeps SDPA on the native ATen op. Recorded as `metax_flash_attn_disabled`.
    sdp_flags_constant = False
    metax_flash_attn_disabled = False
    if hasattr(torch, "disable_flash_attn"):
        for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
            setattr(torch.backends.cuda, name,
                    torch._dynamo.assume_constant_result(getattr(torch.backends.cuda, name)))
        sdp_flags_constant = True
        torch.disable_flash_attn = True
        metax_flash_attn_disabled = True
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16).cuda().eval()
    model.generation_config.cache_implementation = "static"
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    from transformers import CompileConfig
    meta = {"system": "torch.compile", "model": str(args.model), "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "mode": args.mode, "fullgraph": True,
            "compile_path": "generation_config.compile_config (decode forward; prefill eager)",
            "sdp_flags_constant": sdp_flags_constant,
            "metax_flash_attn_disabled": metax_flash_attn_disabled}

    def compile_forward(fullgraph):
        torch._dynamo.reset()
        if hasattr(model, "_compiled_call"):
            del model._compiled_call
        model.generation_config.compile_config = CompileConfig(fullgraph=fullgraph, mode=args.mode)

    compile_forward(True)

    def run(batch, count):
        # Contiguous on device: MACA torch can return non-contiguous strides
        # for a host-to-device copy (MX1_probe notes).
        input_ids = torch.tensor(ids[:batch], dtype=torch.long).cuda().contiguous()
        mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(input_ids, attention_mask=mask, do_sample=False,
                                 min_new_tokens=count, max_new_tokens=count)
        torch.cuda.synchronize()
        if out.shape[0] != batch or not torch.equal(out[:, :prompt_len].cpu(), input_ids.cpu()):
            raise AssertionError("torch.compile output does not match the frozen prompts")
        return out[:, prompt_len:].tolist()

    def generate(batch, count):
        try:
            tokens = run(batch, count)
            # What generate() actually used: it may lower fullgraph itself or
            # decline to compile; a silently eager run must not be labelled.
            config = model.generation_config.compile_config
            meta["fullgraph"] = bool(meta["fullgraph"] and config.fullgraph)
            meta["compiled_call_used"] = hasattr(model, "_compiled_call")
            return tokens
        except Exception:
            if not meta["fullgraph"]:
                raise
            meta["fullgraph"] = False
            meta["fullgraph_error"] = traceback.format_exc(limit=8)[-4000:]
            print("fullgraph=True failed; retrying with fullgraph=False", file=sys.stderr, flush=True)
            # The failed call left a half-initialized static cache behind.
            if hasattr(model, "_cache"):
                del model._cache
            compile_forward(False)
            return run(batch, count)

    batches = (1, 2, 4, 8, 16) if args.batch == "all" else (int(args.batch),)
    measure(args.out, batches, args.max_tokens, args.warmup, args.repeats, policy, generate, meta)


if __name__ == "__main__":
    main()
