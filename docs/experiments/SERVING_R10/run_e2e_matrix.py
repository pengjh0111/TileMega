#!/usr/bin/env python3
"""Run the ten complete R10 requests and retain raw per-engine evidence."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLANS = Path("/root/r10_work/plans")
TORCH = "/root/venvs/tilemega-torch213-cu126/bin/python"
VLLM = "/root/venv_vllm/bin/python"


def command(argv: list[str], out: Path, *, pythonpath: bool = True,
            allow_failure: bool = False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = str(ROOT / "python")
    start = time.perf_counter()
    lock_path = Path("/root/r10_work/serving_gpu.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (out / "stdout.txt").open("w") as stdout, \
             (out / "stderr.txt").open("w") as stderr:
            result = subprocess.run(argv, cwd=ROOT, env=env,
                                    stdout=stdout, stderr=stderr)
    (out / "command.json").write_text(json.dumps({
        "argv": argv, "seconds": time.perf_counter() - start,
        "returncode": result.returncode,
    }, indent=2) + "\n")
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed ({result.returncode}): {out}")
    return result.returncode


def copy_greedy(cache: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for source in cache.glob("hf_free_greedy_b*.json"):
        target = dest / source.name
        if not target.exists():
            shutil.copyfile(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("llama", "qwen3"),
                        default=["llama", "qwen3"])
    parser.add_argument("--batches", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    out_root = HERE / "ev1"
    failed = False
    for model in args.models:
        checkpoint = Path("/root/models") / (
            "llama3_2_1b" if model == "llama" else "qwen3_1_7b")
        ids = HERE / "prompts" / f"{model}_ids.json"
        greedy_cache = out_root / model / "hf_free_cache"
        greedy_cache.mkdir(parents=True, exist_ok=True)
        # Measure every TileMega batch first, then keep a single vLLM LLM
        # instance alive while it measures all five batches for this model.
        # This is the baseline-session contract from §4.1(c).
        for batch in args.batches:
            cell = out_root / model / f"B{batch}"
            prefill = PLANS / f"{model}_prefill_B{batch}" / "plan.so"
            decode = PLANS / f"{model}_decode_B{batch}" / "plan.so"
            if not all(path.exists() and Path(str(path) + ".plan.json").exists()
                       for path in (prefill, decode)):
                print(f"{model} B{batch}: plans not complete", flush=True)
                failed = True
                continue
            common = ["--model", str(checkpoint), "--prefill-so", str(prefill),
                      "--decode-so", str(decode), "--prompt-ids", str(ids),
                      "--batch", str(batch)]
            try:
                tilemega = cell / "tilemega"
                if not (tilemega / "measurements.json").exists():
                    command([TORCH, "-m", "tilemega.serving.measure", *common,
                             "--out", str(tilemega)], cell / "tilemega_command")
                mode = cell / "mode_check"
                if not (mode / "mode_check.json").exists():
                    command([TORCH, "-m", "tilemega.serving.check_modes", *common,
                             "--out", str(mode)], cell / "mode_command")
            except Exception as error:
                print(f"{model} B{batch}: {error}", flush=True)
                failed = True
        vllm_session = out_root / model / "vllm_session"
        if not all((vllm_session / f"B{batch}" /
                    "measurements.json").exists() for batch in args.batches):
            try:
                command([VLLM, "-m", "tilemega.serving.vllm_baseline",
                         "--model", str(checkpoint), "--prompt-ids", str(ids),
                         "--batch", "all", "--out", str(vllm_session)],
                        out_root / model / "vllm_command")
            except Exception as error:
                print(f"{model} vLLM session: {error}", flush=True)
                failed = True
                continue
        for batch in args.batches:
            cell = out_root / model / f"B{batch}"
            prefill = PLANS / f"{model}_prefill_B{batch}" / "plan.so"
            decode = PLANS / f"{model}_decode_B{batch}" / "plan.so"
            if not all(path.exists() and Path(str(path) + ".plan.json").exists()
                       for path in (prefill, decode)):
                print(f"{model} B{batch}: plans not complete", flush=True)
                failed = True
                continue
            try:
                tilemega = cell / "tilemega"
                mode = cell / "mode_check"
                if not (tilemega / "measurements.json").exists() or not (
                        mode / "mode_check.json").exists():
                    raise RuntimeError("TileMega measurement or mode check missing")
                repeated = [json.loads((tilemega /
                    f"tokens_N1024_run{i}.json").read_text()) for i in (1, 2, 3)]
                timed_equal = repeated[0] == repeated[1] == repeated[2]
                baseline = vllm_session / f"B{batch}"
                vllm_hf = cell / "vllm_hf"
                copy_greedy(greedy_cache, vllm_hf)
                if not (vllm_hf / "check.json").exists():
                    command([TORCH, "-m", "tilemega.serving.hf_check",
                             "--model", str(checkpoint), "--prompt-ids", str(ids),
                             "--generated", str(baseline / "tokens_N1024_run1.json"),
                             "--out", str(vllm_hf / "check.json")],
                            cell / "vllm_hf_command", allow_failure=True)
                copy_greedy(vllm_hf, greedy_cache)
                tm_hf = cell / "tilemega_hf"
                copy_greedy(greedy_cache, tm_hf)
                if not (tm_hf / "check.json").exists():
                    command([TORCH, "-m", "tilemega.serving.hf_check",
                             "--model", str(checkpoint), "--prompt-ids", str(ids),
                             "--generated", str(tilemega / "tokens_N1024_run1.json"),
                             "--vllm-metrics", str(vllm_hf / "check.json"),
                             "--out", str(tm_hf / "check.json")],
                            cell / "tilemega_hf_command", allow_failure=True)
                copy_greedy(tm_hf, greedy_cache)
                tm = json.loads((tilemega / "measurements.json").read_text())
                vl = json.loads((baseline / "measurements.json").read_text())
                tm_hf_report = json.loads((tm_hf / "check.json").read_text())
                mode_report = json.loads((mode / "mode_check.json").read_text())
                report = {"model": model, "batch": batch,
                          "tilemega_e2e_seconds": tm["e2e_seconds"],
                          "vllm_e2e_seconds": vl["e2e_seconds"],
                          "throughput_ratio": vl["e2e_seconds"] / tm["e2e_seconds"],
                          "tilemega_hf_pass": tm_hf_report["pass"],
                          "mode_equal": mode_report["pass"],
                          "timed_tokens_equal": timed_equal,
                          "evidence": str(cell.relative_to(ROOT))}
                (cell / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report), flush=True)
                failed |= not (report["tilemega_hf_pass"] and
                               report["mode_equal"] and timed_equal)
            except Exception as error:
                print(f"{model} B{batch}: {error}", flush=True)
                failed = True
                continue
    for script in ("floor_report.py", "collect_flow_evidence.py",
                   "step_ratio_report.py", "build_report_tables.py",
                   "gate_report.py"):
        try:
            subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT,
                           check=True)
        except subprocess.CalledProcessError as error:
            print(f"report generation failed: {script}: {error}", flush=True)
            failed = True
    try:
        subprocess.run([sys.executable, str(HERE / "bench_collective.py")],
                       cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        # This auxiliary mainloop comparison is a report item, not an EV-1
        # correctness gate; keep the ten-cell result status independent.
        print(f"collective microbenchmark failed: {error}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
