#!/usr/bin/env python3
"""Populate the serving resource cache before a coordinate-descent run.

Only the SM80-class legal shape set and the target's dynamic shared-memory
limit determine which wrappers can compile.  The C++ solver still checks
each class's semantic legality and queries the cached exact resources.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]


def shapes(max_m: int, max_smem: int, stages_filter: int | None = None, dm: bool = False):
    for m in (16, 32, 64, 128):
        if m > max_m:
            continue
        for n in ((16, 32, 64, 128, 256) if dm else (32, 64, 128, 256)):
            if m * n > 16384:
                continue
            for k in ((16, 32, 64, 128) if dm else (64, 128)):
                per_stage = 2 * k * (m + n)
                stage_limit = max_smem // per_stage
                if dm:
                    stage_limit = min(8, stage_limit)
                for stages in range(2, stage_limit + 1):
                    if dm and max(per_stage * stages,
                                  4 * (2 if m == n == 16 else 1) * m * n,
                                  4 * m * n + 8 * m) > max_smem:
                        continue
                    if stages_filter is None or stages == stages_filter:
                        yield m, n, k, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-m", type=int, choices=(16, 32, 64, 128),
                        default=128)
    parser.add_argument("--dm", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stages", type=int,
                        help="probe only this executed pipeline depth")
    args = parser.parse_args()
    target = json.loads(args.target.read_text())
    arch = f"sm_{target['sm_major']}{target['sm_minor']}"
    limit = target["resources"]["max_dynamic_smem_per_cta"]
    configurations = list(shapes(args.max_m, limit, args.stages, args.dm))
    if not configurations:
        parser.error("no legal variant at the requested pipeline depth")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    def probe(shape):
        label = "x".join(map(str, shape))
        result = args.output / f"{label}.json"
        log = args.output / f"{label}.log"
        command = [sys.executable, str(Path(__file__).with_name("probe.py")),
                   "--cache", str(args.cache), "--output", str(result),
                   "--arch", arch, "--dtype", "bf16", "--serving",
                   "--tile", ",".join(map(str, shape))]
        if args.dm:
            command.append("--dm")
        with log.open("w") as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        return label, run.returncode, result

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed(pool.submit(probe, shape)
                                   for shape in configurations):
            label, returncode, path = future.result()
            records.append({"shape": label, "returncode": returncode,
                            "resource": str(path) if returncode == 0 else None})
            print(f"{label}: {'PASS' if returncode == 0 else 'FAIL'}", flush=True)
    summary = {"target": str(args.target), "arch": arch, "max_m": args.max_m,
               "stages_filter": args.stages,
               "jobs": args.jobs, "shape_count": len(configurations),
               "seconds": time.monotonic() - start,
               "failed": sum(item["returncode"] != 0 for item in records),
               "records": sorted(records, key=lambda item: item["shape"])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("shape_count", "seconds", "failed")}), flush=True)
    return bool(summary["failed"])


if __name__ == "__main__":
    sys.exit(main())
