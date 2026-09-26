#!/usr/bin/env python3
"""Summarize per-plan solver phases from the raw timing and process logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "report_tables" / "solver_phases.tsv"
PHASES = (
    "bridge_and_plan", "instantiate", "derive", "prepare_relations",
    "instantiate_and_derive", "resource_probe", "piece_pricing_and_release",
    "incremental_prepare", "flow", "skeleton", "materialize", "simulate",
    "megakernel_compile", "search_evaluations", "search_rounds",
    "cache_hit", "cache_miss", "price_cache_hit", "release_cache_hit",
    "incremental_space_hit",
)


def main() -> None:
    records = []
    for model in ("llama", "qwen3"):
        for phase in ("prefill", "decode"):
            for batch in (1, 2, 4, 8, 16):
                cell = f"{model}_{phase}_B{batch}"
                directory = HERE / "plans" / cell
                result_path = directory / "result.json"
                timing_path = directory / "plan.so.timing.tsv"
                if not result_path.exists() or not timing_path.exists():
                    continue
                result = json.loads(result_path.read_text())
                if result.get("returncode") != 0:
                    continue
                with timing_path.open() as stream:
                    timing = {row["phase"]: row for row in csv.DictReader(
                        stream, delimiter="\t")}
                row: dict[str, object] = {
                    "cell": cell, "wall_seconds": result["seconds"],
                    "budget_seconds": 600,
                    "budget_pass": result["seconds"] <= 600,
                    "evidence": str(timing_path.relative_to(HERE)),
                }
                for name in PHASES:
                    item = timing.get(name, {})
                    row[f"{name}_count"] = item.get("count", "")
                    row[f"{name}_seconds"] = (
                        float(item["total_ms"]) / 1000
                        if item.get("total_ms") else "")
                records.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    columns = list(records[0]) if records else ["cell", "wall_seconds",
        "budget_seconds", "budget_pass", "evidence"]
    with OUT.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"solver phase rows: {len(records)}/20")


if __name__ == "__main__":
    main()
