"""Independent vLLM baseline over frozen token IDs; imports no TileMega code."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
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
    args = parser.parse_args()
    # This helper imports only the standard library; the vLLM environment
    # never imports tilemega or its torch-dependent serving engine.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measurement_policy import TimingPolicy
    policy = TimingPolicy(args.policy)
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt

    ids = json.loads(args.prompt_ids.read_text())
    from workload import validate_prompt_ids
    batch_limit = 16 if args.batch == 'all' else int(args.batch)
    prompt_len = validate_prompt_ids(ids, batch_limit)
    if args.max_tokens < 1:
        raise ValueError('max-tokens must be positive')
    try:
        llm = LLM(
            model=str(args.model), dtype="bfloat16", max_model_len=max(2048, prompt_len+args.max_tokens),
            max_num_seqs=16, max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85, enable_prefix_caching=False,
            enforce_eager=False, seed=0,
        )
    except BaseException:
        # Use only standard-library NVML queries: engine initialization may
        # fail before vLLM has exposed a usable object.
        if _external_allocation():
            raise SystemExit(75)
        raise
    batches = (1, 2, 4, 8, 16) if args.batch == "all" else (int(args.batch),)
    for batch in batches:
        out = args.out / f"B{batch}"
        out.mkdir(parents=True, exist_ok=True)
        prompts = [TokensPrompt(prompt_token_ids=row) for row in ids[:batch]]
        rows = []
        guard_path = out / "guard.jsonl"
        for count in (args.max_tokens, 1):
            sampling = SamplingParams(
                temperature=0.0, max_tokens=count, ignore_eos=True,
                detokenize=False,
            )
            for run in range(args.warmup + args.repeats):
                for attempt in range(3):
                    if policy.options.get("guard", True) and not _wait_for_exclusive_gpu(
                            guard_path, f"N{count}-run{run}-before"):
                        raise SystemExit(75)
                    if not policy.observe(guard_path, f"N{count}-run{run}-attempt{attempt}-before"):
                        continue
                    start = time.perf_counter()
                    result = llm.generate(prompts, sampling, use_tqdm=False)
                    elapsed = time.perf_counter() - start
                    exclusive = not policy.options.get("guard", True) or _wait_for_exclusive_gpu(
                        guard_path, f"N{count}-run{run}-after", fail_fast=True)
                    power_ok = policy.observe(guard_path, f"N{count}-run{run}-attempt{attempt}-after")
                    if exclusive and power_ok:
                        break
                else:
                    raise SystemExit(75)
                if any(list(x.prompt_token_ids) != ids[i] for i, x in enumerate(result)):
                    raise AssertionError("vLLM output order does not match the frozen prompts")
                tokens = [list(x.outputs[0].token_ids) for x in result]
                if len(tokens) != batch or any(len(row) != count for row in tokens):
                    raise AssertionError("vLLM did not generate the requested tokens")
                token_file = f"tokens_N{count}_run{run}.json"
                row = {"N": count, "run": run, "warmup": run < args.warmup,
                       "wall_seconds": elapsed, "tokens_file": token_file}
                rows.append(row)
                (out / token_file).write_text(
                    json.dumps(tokens, separators=(",", ":")) + "\n")
        timed = lambda n: [r["wall_seconds"] for r in rows if r["N"] == n and not r["warmup"]]
        e2e = statistics.median(timed(args.max_tokens))
        ttft = statistics.median(timed(1))
        summary = {
            "model": str(args.model), "batch": batch,
            "vllm_version": vllm_version, "max_tokens": args.max_tokens,
            "e2e_seconds": e2e, "ttft_seconds": ttft,
            "tpot_seconds": (e2e - ttft) / (args.max_tokens - 1),
            "output_tokens_per_second": batch * args.max_tokens / e2e,
            "generation_runs": rows,
            "measurement_policy": policy.options,
        }
        (out / "measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "generation_runs"}))


def _external_allocation():
    """True when memory on the device belongs to someone else (mx-smi)."""
    from baseline_common import _mx_smi, own_pids
    owners, visible, above_idle = _mx_smi().occupancy(os.environ.get("TILEMEGA_DEVICE_INDEX", "0"))
    return bool(set(owners) - own_pids()) or above_idle - visible > 1024


def _wait_for_exclusive_gpu(path: Path, label: str, fail_fast: bool = False) -> bool:
    from baseline_common import exclusive_gpu
    return exclusive_gpu(path, label, fail_fast)


if __name__ == "__main__":
    main()
