"""Independent vLLM baseline over frozen token IDs; imports no TileMega code."""
from baseline_common import arguments, guarded_main, measure, require_guard


def main():
    args, ids = arguments("vllm")
    require_guard()
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(args.model), dtype="bfloat16", kv_cache_dtype="auto",
              load_format=args.load_format, skip_tokenizer_init=True,
              max_model_len=max(2048, len(ids[0])+args.max_tokens), max_num_seqs=16,
              max_num_batched_tokens=2048, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, enforce_eager=False, seed=0)
    config = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16 (auto follows model)",
              "prefix_caching": False, "speculative_decoding": False,
              "cuda_graph": "V1 default compilation config", "seed": 0,
              "attention_backend": "resolved by vLLM; see engine stdout"}
    try:
        engine_config = llm.llm_engine.vllm_config
        config["engine_config"] = str(engine_config)
    except AttributeError:
        pass

    def generate(rows, count):
        result = llm.generate([{"prompt_token_ids": row} for row in rows],
            SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False), use_tqdm=False)
        if len(result) != len(rows) or any(list(item.prompt_token_ids) != rows[i] for i, item in enumerate(result)):
            raise AssertionError("vLLM output prompt order changed")
        return [list(item.outputs[0].token_ids) for item in result]
    measure(args, ids, "vllm", generate, config)


if __name__ == "__main__":
    guarded_main(main)
