"""Independent HF torch.compile baseline; supports config-only random models."""
import functools
import json
from baseline_common import arguments, guarded_main, measure, require_guard
import sys
from pathlib import Path

# Standalone execution must not let serving/trace.py shadow stdlib trace.
sys.path = [entry for entry in sys.path
            if Path(entry or ".").resolve() != Path(__file__).resolve().parent]


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

    def compile_forward(fullgraph):
        compiled = torch.compile(eager_forward, mode="reduce-overhead", fullgraph=fullgraph)
        @functools.wraps(eager_forward)
        def forward(*values, **options):
            torch.compiler.cudagraph_mark_step_begin()
            return compiled(*values, **options)
        return forward

    model.forward = compile_forward(True)
    config = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "mode": "reduce-overhead",
              "fullgraph": True, "cache_implementation": "static", "seed": 0,
              "static_cache_initialization": "public early_initialization outside compiled region",
              "hf_auto_compile": False, "cudagraph_mark_step_begin": "each forward invocation",
              "compile_preparation_counts": [1, 2] if args.max_tokens > 1 else [1],
              "compile_preparation_repeats": 3,
              "compile_preparation_location": "inside discarded warmup, before requested N1 timing",
              "prefix_caching": False, "speculative_decoding": False,
              "attention_backend": str(model.config._attn_implementation),
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
                head_dim=getattr(model.config, "head_dim", model.config.hidden_size//model.config.num_attention_heads),
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
