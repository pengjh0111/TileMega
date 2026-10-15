"""Independent SGLang baseline using its offline Engine and frozen token IDs."""
from baseline_common import arguments, guarded_main, measure, require_guard


def main():
    args, ids = arguments("sglang")
    require_guard()
    import sglang as sgl
    options = dict(model_path=str(args.model), dtype="bfloat16", kv_cache_dtype="bf16",
                   load_format=args.load_format, disable_radix_cache=True,
                   skip_tokenizer_init=True, max_running_requests=16,
                   cuda_graph_max_bs=16, cuda_graph_bs=[1, 2, 4, 8, 16],
                   context_length=max(2048, len(ids[0])+args.max_tokens),
                   mem_fraction_static=0.85, random_seed=0)
    engine = sgl.Engine(**options)
    config = {**options, "speculative_decoding": False}
    config["attention_backend"] = getattr(engine.server_args, "attention_backend", None)
    config["decode_attention_backend"] = getattr(engine.server_args, "decode_attention_backend", None)

    def generate(rows, count):
        result = engine.generate(input_ids=rows, sampling_params={
            "temperature": 0.0, "max_new_tokens": count, "ignore_eos": True})
        if isinstance(result, dict):
            result = [result]
        if len(result) != len(rows):
            raise AssertionError("SGLang output batch changed")
        return [list(item["output_ids"]) for item in result]
    try:
        measure(args, ids, "sglang", generate, config)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    guarded_main(main)
