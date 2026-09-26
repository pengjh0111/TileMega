#!/usr/bin/env python3
"""Build and measure old/new BF16 mainloops on an otherwise idle GPU."""
from __future__ import annotations

import json
import fcntl
import csv
import os
from pathlib import Path
import subprocess
import statistics


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "collective_bench"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True).splitlines()[0].strip()
    major, minor = map(int, query.split("."))
    arch = f"sm_{major}{minor}"
    arch_id = 100 * major + 10 * minor
    nvcc = "/usr/local/cuda/bin/nvcc"
    binary = OUT / "bench_collective"
    build = [nvcc, "-std=c++17", "-O2", f"-arch={arch}",
             f"-DTILEMEGA_BENCH_ARCH_ID={arch_id}",
             "-Iinclude", "-Ithird_party/cutlass/include",
             str(HERE / "bench_collective.cu"), "-o", str(binary)]
    (OUT / "build_command.json").write_text(json.dumps(build, indent=2) + "\n")
    with (OUT / "build.stdout").open("w") as stdout, \
         (OUT / "build.stderr").open("w") as stderr:
        subprocess.run(build, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    lock_path = Path("/root/r10_work/serving_gpu.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gpu_query = ["nvidia-smi", "--query-compute-apps=pid,process_name",
                     "--format=csv,noheader"]
        before = subprocess.check_output(gpu_query, text=True).strip()
        if before:
            raise RuntimeError(f"GPU is occupied before benchmark: {before}")
        samples: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
        for run in range(3):
            path = OUT / f"run{run}.tsv"
            with path.open("w") as stdout, \
                 (OUT / f"run{run}.stderr").open("w") as stderr:
                subprocess.run([str(binary)], cwd=ROOT, stdout=stdout,
                               stderr=stderr, check=True, env=os.environ.copy())
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    key = tuple(row[name] for name in
                                ("collective", "M", "K", "ctas"))
                    samples.setdefault(key, []).append(row)
        with (OUT / "result.tsv").open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("collective", "M", "K", "ctas", "median_us",
                             "weight_gbps", "runs", "min_gbps", "max_gbps"))
            for key, values in samples.items():
                rates = [float(row["weight_gbps"]) for row in values]
                times = [float(row["median_us"]) for row in values]
                writer.writerow((*key, statistics.median(times),
                                 statistics.median(rates), len(values),
                                 min(rates), max(rates)))
        after = subprocess.check_output(gpu_query, text=True).strip()
        (OUT / "guard.json").write_text(json.dumps({
            "before": before, "after": after,
        }, indent=2) + "\n")
        if after:
            raise RuntimeError(f"GPU is occupied after benchmark: {after}")
    print(f"wrote {OUT / 'result.tsv'}")


if __name__ == "__main__":
    main()
