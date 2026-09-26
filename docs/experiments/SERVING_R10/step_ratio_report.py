#!/usr/bin/env python3
"""Join EV-1 CUDA-event step times with the CG floor at three past points."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
OUT = HERE / "report_tables"
STEP = {"p_lo": 1, "p_mid": 512, "p_hi": 1023}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> None:
    result = []
    for floor in rows(OUT / "floor_points.tsv"):
        model = floor["model"]
        batch = int(floor["batch"])
        label = floor["point"]
        path = HERE / "ev1" / model / f"B{batch}" / "tilemega" / "step_times.tsv"
        predicted = HERE / "plans" / f"{model}_decode_B{batch}" / \
            "winner.interval_sim.tsv"
        if not path.exists() or not predicted.exists():
            continue
        samples = [float(row["gpu_ms"]) for row in rows(path)
                   if int(row["step"]) == STEP[label]]
        if len(samples) != 3:
            raise AssertionError(f"expected three timed step events: {path} {label}")
        simulation = {int(row["past"]): row for row in rows(predicted)}
        past = int(floor["past"])
        step_ms = statistics.median(samples)
        floor_ms = float(floor["floor_ms"])
        result.append({
            "model": model, "batch": batch, "point": label, "past": past,
            "step": STEP[label], "tilemega_event_median_ms": step_ms,
            "floor_ms": floor_ms, "tilemega_step_over_floor": step_ms / floor_ms,
            "fluid_sim_ms": float(simulation[past]["makespan_ns"]) / 1e6,
            "event_over_fluid_sim": step_ms /
                (float(simulation[past]["makespan_ns"]) / 1e6),
            "step_evidence": str(path.relative_to(HERE)),
            "sim_evidence": str(predicted.relative_to(HERE)),
        })
    if not result:
        print("step ratios pending EV-1")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "step_ratios.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(f"step ratios: {len(result)}/30")


if __name__ == "__main__":
    main()
