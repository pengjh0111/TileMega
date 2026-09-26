#!/usr/bin/env python3
"""Copy the measured winner's raw flow evidence and derive gap components."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
WORK = Path("/root/r10_work/plans")
OUT = HERE / "report_tables"


def tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(name: str, records: list[dict[str, object]]) -> None:
    if not records:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    gaps = []
    for model in ("llama", "qwen3"):
        for batch in (1, 2, 4, 8, 16):
            name = f"{model}_decode_B{batch}"
            raw = WORK / name / "plan.so"
            if not Path(str(raw) + ".plan.json").exists():
                continue
            measured_path = Path(str(raw) + ".top3_measured.tsv")
            if not measured_path.exists():
                continue
            measured_rows = tsv(measured_path)
            if not measured_rows:
                continue  # the background search has not timed its top three yet
            measured = min(measured_rows, key=lambda row: float(row["mean_ms"]))
            rank = int(measured["rank"])
            mode = measured["mode"]
            variant = "A" if mode == "L1" else "B"
            prefix = Path(str(raw) + f".m{rank}{variant}")
            flow_path = Path(str(prefix) + ".flow.tsv")
            chain_path = Path(str(prefix) + ".flow_chain.tsv")
            if not flow_path.exists() or not chain_path.exists():
                raise FileNotFoundError(f"winner flow missing: {name} rank {rank} {mode}")
            dest = HERE / "plans" / name
            dest.mkdir(parents=True, exist_ok=True)
            for suffix in ("flow.tsv", "flow_chain.tsv", "flow_parts.tsv",
                           "flow_spaces.tsv", "interval_sim.tsv",
                           "metrics.tsv", "omissions.tsv"):
                source = Path(str(prefix) + "." + suffix)
                if source.exists():
                    shutil.copyfile(source, dest / f"winner.{suffix}")
            flow = tsv(flow_path)[0]
            gap = float(flow["T"]) - float(flow["T_floor"])
            parts = sum(float(flow[key]) for key in
                        ("synchronization", "fixed", "contention", "chain_delay"))
            if not math.isclose(gap, parts, rel_tol=1e-9, abs_tol=1e-3):
                raise AssertionError(f"flow gap does not decompose: {name}")
            chain = tsv(chain_path)
            by_category: dict[str, tuple[int, float]] = {}
            for row in chain:
                category = row["category"]
                count, duration = by_category.get(category, (0, 0.0))
                by_category[category] = (count + 1, duration +
                    float(row["end_ns"]) - float(row["start_ns"]))
            attention_ns = sum(duration for category, (_, duration) in
                               by_category.items() if "attention" in category)
            protocol_ns = sum(float(row[key]) for row in chain
                              for key in ("wait_ns", "publication_ns", "hop_ns"))
            # The chain category total is a diagnostic, while the four
            # counterfactuals below are the model's exact additive partition.
            gaps.append({
                "model": model, "batch": batch, "rank": rank, "mode": mode,
                "T_ms": float(flow["T"]) / 1e6,
                "floor_ms": float(flow["T_floor"]) / 1e6,
                "synchronization_ms": float(flow["synchronization"]) / 1e6,
                "fixed_ms": float(flow["fixed"]) / 1e6,
                "contention_ms": float(flow["contention"]) / 1e6,
                "chain_delay_ms": float(flow["chain_delay"]) / 1e6,
                "pg_upper_bound_ms": float(flow["pg_upper_bound"]) / 1e6,
                "chain_depth": int(float(flow["chain_depth"])),
                "bubble_us": float(flow["bubble_ns"]) / 1000,
                "attention_chain_share": attention_ns / float(flow["T"]),
                "chain_protocol_constants_ms": protocol_ns / 1e6,
                "rmsnorm_links": by_category.get("rmsnorm", (0, 0.0))[0],
                "rmsnorm_chain_wall_ms": by_category.get("rmsnorm", (0, 0.0))[1] / 1e6,
                "evidence": str((dest / "winner.flow.tsv").relative_to(HERE)),
            })
    write("gaps.tsv", gaps)
    print(f"winner flow evidence: {len(gaps)}/10 decode plans")


if __name__ == "__main__":
    main()
