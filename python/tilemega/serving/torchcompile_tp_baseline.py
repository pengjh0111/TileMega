"""HF torch.compile baseline at tensor parallel degree T (user request, beyond §5 B0.5).

transformers carries Qwen3's own `base_model_tp_plan` (colwise q/k/v/gate/up,
rowwise o/down — the same Megatron split TileMega implements), so TP comes from
`from_pretrained(..., tp_plan="auto")` under torchrun rather than from anything
written here. One rank per card of the claimed group; rank 0 times the request
under the R16 protocol and writes the measurement, the other ranks only
participate. Nothing about the engine is tuned: the single-card settings of
torchcompile_baseline.py are reused verbatim.

Launched by scripts/sm90r5_tp_baselines.py through torchrun; a direct run needs
RANK/WORLD_SIZE/LOCAL_RANK in the environment.
"""
import functools
import os
import sys
from pathlib import Path

from baseline_common import arguments, guarded_main, measure, require_guard

sys.path = [entry for entry in sys.path
            if Path(entry or ".").resolve() != Path(__file__).resolve().parent]


# Set by main() when the NCCL group has to be left to process exit; see below.
SKIP_NCCL_TEARDOWN = False


def main():
    global SKIP_NCCL_TEARDOWN
    args, ids = arguments("torch.compile")
    devices = require_guard(args.tp if args.tp > 1 else None)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.tp:
        raise ValueError(f"--tp {args.tp} but WORLD_SIZE {world}; launch with torchrun --nproc_per_node")
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, StaticCache
    try:
        from torch._dynamo.exc import FailOnRecompileLimitHit as RecompileLimit
    except ImportError:  # older torch raises something else; then no fallback fires
        class RecompileLimit(Exception):
            pass
    torch.manual_seed(0)
    # torchrun gives each rank one local device out of the guarded group.
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        **(dict(tp_plan="auto") if world > 1 else {}))
    if world == 1:
        model = model.cuda()
    model.eval()
    model.generation_config.cache_implementation = None
    model.generation_config.disable_compile = True
    eager_forward = model.forward
    # The mode torchcompile_baseline.py picks for this model, by the same rule:
    # reduce-overhead (a CUDA graph per decode step) unless the model is MoE.
    # Nothing is chosen for TP's sake; a setting that TP cannot run has to fail
    # and be reported, not be quietly swapped for a slower one.
    moe = any(key in model.config.to_dict() for key in ("num_experts", "num_local_experts"))
    mode = "max-autotune-no-cudagraphs" if moe else "reduce-overhead"

    def compile_forward(fullgraph):
        compiled = torch.compile(eager_forward, mode=mode, fullgraph=fullgraph)
        @functools.wraps(eager_forward)
        def forward(*values, **options):
            if mode == "reduce-overhead":
                torch.compiler.cudagraph_mark_step_begin()
            return compiled(*values, **options)
        return forward

    # transformers' tp_plan registers module-level backward hooks (its
    # replicated_with_grad_allreduce entries for q_norm/k_norm), which dynamo
    # rejects under fullgraph even at inference, so a sharded model compiles
    # with fullgraph off. The choice is made before any collective runs: a
    # mid-forward compile failure would leave the ranks at different collective
    # counts and hang the next all-reduce (observed as an NCCL timeout).
    fullgraph = world == 1
    model.forward = compile_forward(fullgraph)
    config = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "mode": mode, "fullgraph": fullgraph,
              "mode_source": "torchcompile_baseline.py's rule for this model (dense: reduce-overhead)",
              "fullgraph_reason": ("sharded: transformers tp_plan registers module-level backward hooks, "
                                   "which dynamo rejects under fullgraph") if world > 1 else "unsharded",
              "cache_implementation": "static", "seed": 0, "tp": world,
              "tp_plan": "auto (transformers base_model_tp_plan)",
              "tp_devices": devices, "launcher": "torchrun --nproc_per_node=T",
              "cuda_graphs": mode == "reduce-overhead", "hf_auto_compile": False,
              "cudagraph_mark_step_begin": ("each forward invocation"
                                            if mode == "reduce-overhead" else None),
              "prefix_caching": False,
              "speculative_decoding": False,
              "attention_backend": str(model.config._attn_implementation),
              "compile_preparation_counts": [1, 2] if args.max_tokens > 1 else [1],
              "compile_preparation_repeats": 3}
    eos = model.config.eos_token_id
    pad = (eos[0] if isinstance(eos, list) else eos) or 0
    caches, prepared = {}, set()

    def generate_body(rows, count):
        inputs = torch.tensor(rows, dtype=torch.long, device="cuda")
        mask = torch.ones_like(inputs)
        if len(rows) not in caches:
            cache = StaticCache(config=model.config, max_cache_len=len(ids[0]) + args.max_tokens)
            # Each rank holds num_key_value_heads/T of the cache: the tp_plan
            # shards k/v colwise, so a full-width cache fails to broadcast in
            # the index_put_ of the cache update.
            cache.early_initialization(
                batch_size=len(rows), num_heads=model.config.num_key_value_heads // world,
                head_dim=getattr(model.config, "head_dim", None) or
                         model.config.hidden_size // model.config.num_attention_heads,
                dtype=torch.bfloat16, device=inputs.device)
            caches.clear()
            caches[len(rows)] = cache
        cache = caches[len(rows)]

        def invoke(new_tokens):
            cache.reset()
            return model.generate(input_ids=inputs, attention_mask=mask, do_sample=False,
                                  min_new_tokens=new_tokens, max_new_tokens=new_tokens,
                                  use_cache=True, pad_token_id=pad, past_key_values=cache)

        def prepared_invoke():
            key = (len(rows), config["fullgraph"])
            if key not in prepared:
                try:
                    for prep in config["compile_preparation_counts"]:
                        for _ in range(config["compile_preparation_repeats"]):
                            invoke(prep)
                except RecompileLimit as error:
                    # fullgraph is the engine's own single-card setting, so it
                    # is attempted first and only given up when dynamo itself
                    # refuses: transformers 5.20.1 recompiles this model past
                    # the limit even unsharded. Recorded, never assumed. At
                    # world > 1 the choice is already fullgraph-off and must
                    # stay fixed, because changing it mid-forward would leave
                    # the ranks at different collective counts.
                    if world > 1:
                        raise
                    config.update(fullgraph=False, fullgraph_reason=(
                        f"unsharded: fullgraph attempted first and refused by dynamo — "
                        f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"))
                    torch._dynamo.reset()
                    model.forward = compile_forward(False)
                    prepared.clear()
                    return prepared_invoke()
                prepared.add(key)
            return invoke(count)

        with torch.inference_mode():
            # No fallback: every rank must take the same path (see above).
            output = prepared_invoke()
        # Every rank runs the same tokens; a divergence means the TP plan broke.
        if world > 1:
            first = output.clone()
            dist.broadcast(first, src=0)
            if not torch.equal(first, output):
                raise AssertionError(f"rank {rank} generated tokens differing from rank 0")
        if not torch.equal(output[:, :inputs.shape[1]], inputs):
            raise AssertionError("HF generation changed the frozen input prompt")
        return output[:, inputs.shape[1]:].tolist()

    # Rank 0 drives; the others take each (batch, count) from it. Replicating
    # measure()'s loop by hand is not safe: it orders counts (1, max_tokens)
    # and may retry a rejected round, and every mismatched token count issues a
    # different number of collectives, which hangs the next all-reduce.
    def generate(rows, count):
        if world > 1:
            dist.broadcast(torch.tensor([len(rows), count], dtype=torch.long, device="cuda"), src=0)
        return generate_body(rows, count)

    def mark(stage):
        # Teardown progress, so a process that stops making it is diagnosable
        # from the run log instead of needing a live stack.
        print(f"TP rank {rank}: {stage}", flush=True)

    if rank == 0:
        measure(args, ids, "torch.compile", generate, config, synchronize=torch.cuda.synchronize)
        mark("measurements written")
        if world > 1:
            dist.broadcast(torch.zeros(2, dtype=torch.long, device="cuda"), src=0)
            mark("stop sent")
    else:
        while True:
            item = torch.zeros(2, dtype=torch.long, device="cuda")
            dist.broadcast(item, src=0)
            batch, count = int(item[0]), int(item[1])
            if count == 0:
                break
            generate_body([list(row) for row in ids[:batch]], count)
        mark("stop received")
    if world > 1:
        # No teardown collective. `dist.destroy_process_group()` does not
        # return once decode steps have been captured in CUDA graphs: both
        # ranks reached it and stopped there, GPUs idle, with the measurement
        # already on disk (the marks above are what showed it). NCCL's atexit
        # teardown blocks the same way, so the interpreter is ended without it
        # in __main__. Nothing measured depends on this: the stop broadcast
        # already ordered the ranks and every file is written and closed.
        SKIP_NCCL_TEARDOWN = True
        mark("process group left to process exit (NCCL teardown hangs under cudagraphs)")


if __name__ == "__main__":
    guarded_main(main)
    sys.stdout.flush()
    sys.stderr.flush()
    if SKIP_NCCL_TEARDOWN:
        os._exit(0)
