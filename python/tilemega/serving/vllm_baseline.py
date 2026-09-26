"""Independent vLLM baseline over frozen token IDs; imports no TileMega code."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--batch", choices=("1", "2", "4", "8", "16", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt

    ids = json.loads(args.prompt_ids.read_text())
    assert len(ids) == 16 and all(len(row) == 64 for row in ids)
    llm = LLM(
        model=str(args.model), dtype="bfloat16", max_model_len=2048,
        max_num_seqs=16, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85, enable_prefix_caching=False,
        enforce_eager=False, seed=0,
    )
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
            for run in range(4):
                for attempt in range(3):
                    if not _wait_for_exclusive_gpu(
                            guard_path, f"N{count}-run{run}-before"):
                        raise RuntimeError("GPU remained occupied for 30 minutes")
                    start = time.perf_counter()
                    result = llm.generate(prompts, sampling, use_tqdm=False)
                    elapsed = time.perf_counter() - start
                    exclusive = _wait_for_exclusive_gpu(
                        guard_path, f"N{count}-run{run}-after", fail_fast=True)
                    if exclusive:
                        break
                else:
                    raise RuntimeError("GPU was contaminated in all three attempts")
                if any(list(x.prompt_token_ids) != ids[i] for i, x in enumerate(result)):
                    raise AssertionError("vLLM output order does not match the frozen prompts")
                tokens = [list(x.outputs[0].token_ids) for x in result]
                if len(tokens) != batch or any(len(row) != count for row in tokens):
                    raise AssertionError("vLLM did not generate the requested tokens")
                token_file = f"tokens_N{count}_run{run}.json"
                row = {"N": count, "run": run, "warmup": run == 0,
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
        }
        (out / "measurements.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "generation_runs"}))


def _wait_for_exclusive_gpu(path: Path, label: str, fail_fast: bool = False) -> bool:
    deadline = time.monotonic() if fail_fast else time.monotonic() + 30 * 60
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True)
        rows = [line.split(",") for line in output.splitlines() if line.strip()]
        pids = {int(row[0].strip()) for row in rows}
        visible_mib = sum(int(row[-1].strip()) for row in rows)
        used_mib = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0].strip())
        hidden_mib = max(0, used_mib - visible_mib)
        # vLLM runs its GPU engine in a child process; that process is part of
        # this measurement, whereas another user's PID invalidates the run.
        own = {os.getpid()}
        while True:
            descendants = set()
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    stat = (entry / "stat").read_text().split()
                    if int(stat[3]) in own:
                        descendants.add(int(entry.name))
                except (OSError, ValueError):
                    continue
            if descendants <= own:
                break
            own.update(descendants)
        exclusive = bool(pids) and pids <= own and hidden_mib <= 256
        with path.open("a") as stream:
            stream.write(json.dumps({"label": label, "pids": sorted(pids),
                                     "visible_mib": visible_mib,
                                     "used_mib": used_mib,
                                     "hidden_mib": hidden_mib,
                                     "exclusive": exclusive, "time": time.time()}) + "\n")
        if exclusive or time.monotonic() >= deadline:
            return exclusive
        time.sleep(10)


if __name__ == "__main__":
    main()
