#!/usr/bin/env python3
"""Run the two R10 G-6 full-domain pruning controls without GPU compilation."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "pruning_equivalence"
WORK = Path("/root/r10_work/pruning_equivalence")
TARGET = HERE / "calibration/target_serving.json"
LIMIT_SECONDS = 90 * 60


def winner(path: Path) -> dict[str, str]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    good = [row for row in rows if not row.get("error")]
    if not good:
        raise RuntimeError(f"no legal configurations in {path}")
    return min(good, key=lambda row: float(row["flow_ns"]))


def admitted(path: Path, key: str) -> tuple[bool, list[str]]:
    """Use the actual C++ pruned domain dump, not a Python copy of R-2/R-3."""
    members: dict[int, set[str]] = {}
    for line in path.read_text().splitlines():
        cells = line.split("\t")
        if cells[0] == "DOMAIN_MEMBER":
            members.setdefault(int(cells[1]), set()).add(cells[2])
    configs = key.split(";kappa=", 1)[0].split(";")
    missing = [f"class {i}: {config}" for i, config in enumerate(configs)
               if config not in members.get(i, set())]
    return bool(members) and not missing, missing


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    report = {}
    failed = False
    for model, batch in (("llama", 1), ("qwen3", 16)):
        rows = {}
        for prune, arm in ((True, "pruned"), (False, "unpruned")):
            output = WORK / f"{model}_{arm}.cu"
            command = [str(ROOT / "build-portable/tools/tilemega-compile"),
                f"/root/r10_work/export/{model}_decode/bridge.json",
                str(output), "--serving", "decode", "--batch", str(batch),
                "--past-range", "64:1086", "--capacity", "1088",
                "--solver", "skeleton", "--solve", str(TARGET),
                "--hop-curve", str(ROOT / "docs/experiments/SIMULATOR/hop_ns.tsv"),
                "--variant-cache", "/root/r9_work/variant_resources",
                "--emit", "serving", "--flow-search-only", "1",
                "--search-passes", "3", "--incremental-prepare", "1",
                "--serving-pruning", "1" if prune else "0"]
            start = time.perf_counter()
            with (OUT / f"{model}_{arm}.stdout").open("w") as stdout, \
                 (OUT / f"{model}_{arm}.stderr").open("w") as stderr:
                try:
                    run = subprocess.run(command, cwd=ROOT, stdout=stdout,
                        stderr=stderr, timeout=LIMIT_SECONDS)
                    code = run.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            duration = time.perf_counter() - start
            info = {"command": command, "elapsed_seconds": duration,
                    "returncode": code}
            (OUT / f"{model}_{arm}.result.json").write_text(
                json.dumps(info, indent=2) + "\n")
            for suffix in (".search.tsv", ".flow_ranked.tsv", ".timing.tsv"):
                raw = Path(str(output) + suffix)
                if raw.exists():
                    shutil.copyfile(raw, OUT / f"{model}_{arm}{suffix}")
            if code:
                print(f"{model} {arm}: returncode={code} after {duration:.1f}s",
                      flush=True)
                failed = True
                continue
            rows[arm] = winner(OUT / f"{model}_{arm}.flow_ranked.tsv")
            print(f"{model} {arm}: {float(rows[arm]['flow_ns']):.3f} ns"
                  f" after {duration:.1f}s", flush=True)
        if len(rows) == 2:
            ratio = float(rows["pruned"]["flow_ns"]) / float(rows["unpruned"]["flow_ns"])
            member, missing = admitted(OUT / f"{model}_pruned.search.tsv",
                                      rows["unpruned"]["key"])
            report[model] = {"pruned": rows["pruned"], "unpruned": rows["unpruned"],
                             "score_ratio": ratio, "score_gate": ratio <= 1.001,
                             "domain_membership_gate": member,
                             "missing_from_pruned_domain": missing}
            failed |= ratio > 1.001 or not member
        else:
            report[model] = {"error": "one or both arms incomplete"}
            failed = True
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
