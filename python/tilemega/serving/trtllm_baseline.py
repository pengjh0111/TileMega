"""Independent TensorRT-LLM 0.21 PyTorch backend baseline."""
import json
import os
from baseline_common import arguments, guarded_main, measure, require_guard
import sys
from pathlib import Path

# Standalone execution must not let serving/trace.py shadow stdlib trace.
sys.path = [entry for entry in sys.path
            if Path(entry or ".").resolve() != Path(__file__).resolve().parent]


def main():
    args, ids = arguments("tensorrt-llm")
    require_guard(args.tp)
    if args.tp > 1:
        # The MPI worker pool needs these in the environment before the import,
        # and must not be forced into the single-process path.
        os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
        os.environ.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    options = dict(backend="pytorch", dtype="bfloat16", load_format=args.load_format,
                   tensor_parallel_size=args.tp,
                   skip_tokenizer_init=True, max_batch_size=16,
                   max_seq_len=max(2048, len(ids[0])+args.max_tokens),
                   use_cuda_graph=True, cuda_graph_batch_sizes=[1, 2, 4, 8, 16])
    llm = LLM(model=str(args.model), **options,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85))
    config = {**options, "kv_cache_dtype": "bfloat16 (unquantized model dtype)",
              "enable_block_reuse": False, "speculative_decoding": False,
              "worker_use_single_process": os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS", "0") == "1",
              "free_gpu_memory_fraction": 0.85,
              "attention_backend": str(getattr(llm.args, "attn_backend", "TRTLLM"))}

    eos = json.loads((args.model/"config.json").read_text()).get("eos_token_id", 0)
    eos = eos[0] if isinstance(eos, list) else eos

    def generate(rows, count):
        result = llm.generate([{"prompt_token_ids": row} for row in rows],
                             SamplingParams(max_tokens=count, ignore_eos=True, top_k=1,
                                            end_id=eos or 0, pad_id=eos or 0, detokenize=False),
                             use_tqdm=False)
        if len(result) != len(rows):
            raise AssertionError("TensorRT-LLM output batch changed")
        for i, item in enumerate(result):
            if getattr(item, "prompt_token_ids", rows[i]) != rows[i]:
                raise AssertionError("TensorRT-LLM output prompt order changed")
        return [list(item.outputs[0].token_ids) for item in result]
    try:
        measure(args, ids, "tensorrt-llm", generate, config)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    guarded_main(main)
