"""Independent HF torch.compile baseline; supports config-only random models."""
import functools
import inspect
import json
from baseline_common import arguments, guarded_main, measure, require_guard
import sys
from pathlib import Path

# Standalone execution must not let serving/trace.py shadow stdlib trace.
sys.path = [entry for entry in sys.path
            if Path(entry or ".").resolve() != Path(__file__).resolve().parent]


def legacy_mask_device(model):
    """Supply the `device` that generate() stopped passing, for models still on
    the legacy 4D-mask helper.

    transformers 4.57.1 generation/utils.py calls the base model's
    _prepare_4d_causal_attention_mask_with_cache_position without `device`
    whenever that method exists, but models not yet migrated to the general
    mask API (olmoe) still declare it required, so a compileable cache raises
    TypeError. `device` only places the helper's torch.full/torch.arange, and
    cache_position -- which the call site does pass -- is already on it, so
    deriving it reproduces the mask the explicit-device call builds. Returns
    whether the model needed this.
    """
    import torch
    base = getattr(model, model.base_model_prefix, model)
    original = getattr(base, "_prepare_4d_causal_attention_mask_with_cache_position", None)
    if original is None:
        return False  # already on the general mask API (llama, qwen3, qwen3_moe, mixtral)
    try:
        inspect.signature(original).bind(None, sequence_length=1, target_length=1,
                                        dtype=torch.float32, cache_position=None,
                                        batch_size=1, config=None, past_key_values=None)
        return False  # accepts the call site's arguments as they are
    except TypeError:
        pass

    def patched(attention_mask, sequence_length, target_length, dtype,
                cache_position, batch_size, device=None, **kwargs):
        return original(attention_mask, sequence_length, target_length, dtype,
                        cache_position.device if device is None else device,
                        cache_position, batch_size, **kwargs)

    type(base)._prepare_4d_causal_attention_mask_with_cache_position = staticmethod(patched)
    return True


def main():
    args, ids = arguments("torch.compile")
    require_guard()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, StaticCache
    torch.manual_seed(0)
    if args.load_format == "dummy":
        hf_config = AutoConfig.from_pretrained(str(args.model), local_files_only=True)
        # Construct in BF16 directly so a 30B random model never has an FP32 copy.
        model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(args.model),
            torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.cuda().eval()
    model.generation_config.cache_implementation = None
    # We compile forward explicitly; HF's automatic __call__ compilation would
    # otherwise add a second compiler around the same model with StaticCache.
    model.generation_config.disable_compile = True
    eager_forward = model.forward
    legacy_mask = legacy_mask_device(model)
    # reduce-overhead captures a CUDA graph per decode step. A model whose MoE
    # routing selects tokens with a data-dependent nonzero forces a graph break,
    # so every capture comes back empty: OLMoE logged 21091 "CUDA Graph is
    # empty" warnings and spent 15 h on one batch. Such a model takes the mode
    # torchcompile_moe_baseline.py already uses for the same reason.
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

    model.forward = compile_forward(True)
    config = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "mode": mode,
              "fullgraph": True, "cache_implementation": "static", "seed": 0,
              "static_cache_initialization": "public early_initialization outside compiled region",
              "hf_auto_compile": False, "cuda_graphs": mode == "reduce-overhead",
              "cudagraph_mark_step_begin": "each forward invocation" if mode == "reduce-overhead" else None,
              "compile_preparation_counts": [1, 2] if args.max_tokens > 1 else [1],
              "compile_preparation_repeats": 3,
              "compile_preparation_location": "inside discarded warmup, before requested N1 timing",
              "prefix_caching": False, "speculative_decoding": False,
              "attention_backend": str(model.config._attn_implementation),
              "legacy_mask_device_supplied": legacy_mask,
              "moe_experts": "installed transformers implementation; grouped_mm unavailable in 4.57.1"}

    eos = model.config.eos_token_id
    pad = (eos[0] if isinstance(eos, list) else eos) or 0

    caches = {}
    prepared = set()

    def generate(rows, count):
        inputs = torch.tensor(rows, dtype=torch.long, device="cuda")
        mask = torch.ones_like(inputs)
        if len(rows) not in caches:
            cache = StaticCache(config=model.config, max_cache_len=len(ids[0])+args.max_tokens)
            cache.early_initialization(batch_size=len(rows), num_heads=model.config.num_key_value_heads,
                # MistralConfig carries head_dim=None rather than omitting it.
                head_dim=getattr(model.config, "head_dim", None) or model.config.hidden_size//model.config.num_attention_heads,
                dtype=torch.bfloat16, device=inputs.device)
            caches.clear()
            caches[len(rows)] = cache
        cache = caches[len(rows)]
        def invoke(new_tokens=count):
            cache.reset()
            return model.generate(input_ids=inputs, attention_mask=mask, do_sample=False,
                min_new_tokens=new_tokens, max_new_tokens=new_tokens, use_cache=True,
                pad_token_id=pad, past_key_values=cache)
        def prepared_invoke():
            key = (len(rows), config["fullgraph"])
            if key not in prepared:
                # Compile, capture and replay both prefill and one-token decode
                # before N=1 timing; a single prefill cannot warm the decode graph.
                for prep_count in config["compile_preparation_counts"]:
                    for _ in range(config["compile_preparation_repeats"]):
                        invoke(prep_count)
                prepared.add(key)
            return invoke()
        with torch.inference_mode():
            try:
                output = prepared_invoke()
            except Exception as error:
                if not config["fullgraph"]:
                    raise
                config["fullgraph_error"] = f"{type(error).__name__}: {error}"
                (args.out/"fullgraph_failure.json").write_text(json.dumps({
                    "error": config["fullgraph_error"]}, indent=2) + "\n")
                config["fullgraph"] = False
                torch._dynamo.reset()
                model.forward = compile_forward(False)
                output = prepared_invoke()
        if not torch.equal(output[:, :inputs.shape[1]], inputs):
            raise AssertionError("HF generation changed the input prompt")
        return output[:, inputs.shape[1]:].tolist()
    measure(args, ids, "torch.compile", generate, config, synchronize=torch.cuda.synchronize)


if __name__ == "__main__":
    guarded_main(main)
