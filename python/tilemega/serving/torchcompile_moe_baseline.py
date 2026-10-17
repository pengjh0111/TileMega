"""Independent official HF grouped-MM MoE baseline with a shared BF16 checkpoint."""
import functools
import json
import sys
from pathlib import Path
from baseline_common import arguments, guarded_main, measure, require_guard

sys.path = [entry for entry in sys.path
            if Path(entry or ".").resolve() != Path(__file__).resolve().parent]


def main():
    args, ids = arguments("torch.compile MoE")
    require_guard()
    if args.load_format != "auto":
        raise ValueError("MoE comparison requires the shared checkpoint, not dummy weights")
    import torch
    from transformers import AutoModelForCausalLM, StaticCache
    torch.manual_seed(0)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, local_files_only=True,
        experts_implementation="grouped_mm", device_map={"": "cuda:0"},
        use_safetensors=True, output_loading_info=True)
    (args.out / "model-loading-info.json").write_text(json.dumps(loading, indent=2) + "\n")
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("official HF checkpoint conversion did not load every parameter")
    if len(model.model.layers) != 48 or any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("expected the complete 48-layer BF16 MoE model")
    model.eval()
    model.generation_config.cache_implementation = None
    model.generation_config.disable_compile = True
    eager_forward = model.forward
    # Official grouped_mm compilation excludes CUDA graphs. HF may publicly
    # switch to batched_mm for decode; observe that choice without overriding it.
    mode = "max-autotune-no-cudagraphs"
    config = dict(dtype="bfloat16", kv_cache_dtype="bfloat16", mode=mode,
        fullgraph=True, cache_implementation="static", seed=0,
        requested_experts_implementation="grouped_mm", experts_observed={},
        experts_switch_policy="official HF generate automatic decode switching; no override",
        cuda_graphs=False, hf_auto_compile=False, prefix_caching=False,
        speculative_decoding=False, quantization=None,
        attention_backend=str(model.config._attn_implementation),
        model_layers=len(model.model.layers),
        parameter_count=sum(p.numel() for p in model.parameters()),
        checkpoint_loading="official from_pretrained conversion; all loading errors rejected",
        static_cache_initialization="public early_initialization before compilation",
        compile_preparation_counts=[1, 2] if args.max_tokens > 1 else [1],
        compile_preparation_repeats=3,
        compile_preparation_location="discarded warmup before N1 timing")

    def compile_forward(fullgraph):
        compiled = torch.compile(eager_forward, mode=mode, fullgraph=fullgraph)
        @functools.wraps(eager_forward)
        def forward(*values, **options):
            # Inspect only the first warmup invocation of each input shape.
            inputs = options.get("input_ids", values[0] if values else None)
            stage = "prefill" if inputs is not None and inputs.shape[1] > 1 else "decode"
            if stage not in config["experts_observed"]:
                config["experts_observed"][stage] = model.get_experts_implementation()
            return compiled(*values, **options)
        return forward

    model.forward = compile_forward(True)
    eos = model.config.eos_token_id
    pad = (eos[0] if isinstance(eos, list) else eos) or 0
    caches, prepared = {}, set()

    def generate(rows, count):
        inputs = torch.tensor(rows, dtype=torch.long, device="cuda")
        mask = torch.ones_like(inputs)
        if len(rows) not in caches:
            cache = StaticCache(config=model.config, max_cache_len=len(ids[0]) + args.max_tokens)
            cache.early_initialization(batch_size=len(rows), num_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim, dtype=torch.bfloat16, device=inputs.device)
            caches.clear()
            caches[len(rows)] = cache
        cache = caches[len(rows)]
        def invoke(new_tokens):
            cache.reset()
            return model.generate(input_ids=inputs, attention_mask=mask, do_sample=False,
                min_new_tokens=new_tokens, max_new_tokens=new_tokens, use_cache=True,
                pad_token_id=pad, past_key_values=cache)
        def prepared_invoke():
            key = (len(rows), config["fullgraph"])
            if key not in prepared:
                for prep_count in config["compile_preparation_counts"]:
                    for _ in range(config["compile_preparation_repeats"]):
                        invoke(prep_count)
                prepared.add(key)
            return invoke(count)
        with torch.inference_mode():
            try:
                output = prepared_invoke()
            except Exception as error:
                if not config["fullgraph"]:
                    raise
                config["fullgraph_error"] = f"{type(error).__name__}: {error}"
                (args.out / "fullgraph_failure.json").write_text(json.dumps(
                    {"error": config["fullgraph_error"]}, indent=2) + "\n")
                config["fullgraph"] = False
                torch._dynamo.reset()
                model.forward = compile_forward(False)
                output = prepared_invoke()
        if not torch.equal(output[:, :inputs.shape[1]], inputs):
            raise AssertionError("HF generation changed the frozen input prompt")
        return output[:, inputs.shape[1]:].tolist()

    measure(args, ids, "torch.compile", generate, config, synchronize=torch.cuda.synchronize)


if __name__ == "__main__":
    guarded_main(main)
