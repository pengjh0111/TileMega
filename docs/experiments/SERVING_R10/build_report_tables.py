#!/usr/bin/env python3
"""Build review tables directly from R10 plan and full-request logs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
OUT = HERE / "report_tables"


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def emit(name: str, data: list[dict[str, object]], columns: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in data:
            writer.writerow({key: row.get(key, "") for key in columns})


def tpot_stats(measurement: dict, elapsed_key: str,
               runs_key: str) -> tuple[float, float, float]:
    samples = measurement[runs_key]
    full = {row["run"]: row[elapsed_key] for row in samples
            if row["N"] == 1024 and not row["warmup"]}
    first = {row["run"]: row[elapsed_key] for row in samples
             if row["N"] == 1 and not row["warmup"]}
    if sorted(full) != [1, 2, 3] or sorted(first) != [1, 2, 3]:
        raise AssertionError("expected three timed full and first-token runs")
    values = sorted((full[run] - first[run]) * 1000 / 1023
                    for run in (1, 2, 3))
    return statistics.mean(values), statistics.median(values), values[2]


def plan_table() -> list[dict[str, object]]:
    result = []
    for model in ("llama", "qwen3"):
        for phase in ("prefill", "decode"):
            for batch in (1, 2, 4, 8, 16):
                cell = f"{model}_{phase}_B{batch}"
                directory = HERE / "plans" / cell
                manifest = directory / "plan.so.plan.json"
                report: dict[str, object] = {"cell": cell, "status": "pending"}
                required = [manifest, directory / "result.json",
                            directory / "plan.so.timing.tsv",
                            directory / "plan.so.top3_measured.tsv",
                            directory / "plan.so.resources.tsv",
                            directory / "fp64_audit.json"]
                if not all(path.exists() for path in required):
                    result.append(report)
                    continue
                plan = json.loads(manifest.read_text())
                solving = json.loads((directory / "result.json").read_text())
                timing = {row["phase"]: row for row in rows(
                    directory / "plan.so.timing.tsv")}
                tiles = {(g["tile_m"], g["tile_n"], g["tile_k"],
                          g["stages"]) for g in plan["gemms"]}
                measured = rows(directory / "plan.so.top3_measured.tsv")
                best = min(measured, key=lambda row: float(row["mean_ms"]))
                resources = {row["rank"]: row for row in rows(
                    directory / "plan.so.resources.tsv")}
                chosen = resources.get(best["rank"], {})
                audit = json.loads((directory / "fp64_audit.json").read_text())
                report.update({
                    "status": "complete" if solving["returncode"] == 0 else "failed",
                    "solve_seconds": solving["seconds"],
                    "mode": plan["mode"], "measured_mode": best["mode"],
                    "measured_ms": best["mean_ms"], "grid": plan["grid"],
                    "residency": plan["residency"], "kappa": plan["kappa"],
                    "ec": plan["attention_kv_block"],
                    "rq": plan["attention_query_rows"],
                    "variant_count": len(tiles),
                    "qkv": str(tuple(plan["gemms"][0][key] for key in
                                      ("tile_m", "tile_n", "tile_k", "stages", "split_k"))),
                    "o": str(tuple(plan["gemms"][1][key] for key in
                                    ("tile_m", "tile_n", "tile_k", "stages", "split_k"))),
                    "gate_up": str(tuple(plan["gemms"][2][key] for key in
                                          ("tile_m", "tile_n", "tile_k", "stages", "split_k"))),
                    "down": str(tuple(plan["gemms"][3][key] for key in
                                       ("tile_m", "tile_n", "tile_k", "stages", "split_k"))),
                    "lm_head": str(tuple(plan["gemms"][-1][key] for key in
                                          ("tile_m", "tile_n", "tile_k", "stages", "split_k"))),
                    "estimated_rmax": chosen.get("estimated", ""),
                    "actual_rmax": chosen.get("actual", ""),
                    "flow_ns": chosen.get("flow_ns", ""),
                    "simulated_ns": chosen.get("simulated_ns", ""),
                    "search_evaluations": timing.get("search_evaluations", {}).get("count", ""),
                    "price_ms": timing.get("piece_pricing_and_release", {}).get("total_ms", ""),
                    "compile_ms": timing.get("megakernel_compile", {}).get("total_ms", ""),
                    "fp64_count": audit["fp64_total"],
                    "evidence": str(directory.relative_to(HERE)),
                })
                result.append(report)
    return result


def request_table() -> list[dict[str, object]]:
    result = []
    floors = {(row["model"], int(row["batch"])): row for row in
              rows(OUT / "request_floors.tsv")}
    for model in ("llama", "qwen3"):
        for batch in (1, 2, 4, 8, 16):
            directory = HERE / "ev1" / model / f"B{batch}"
            measured = directory / "tilemega" / "measurements.json"
            baseline_path = HERE / "ev1" / model / "vllm_session" / f"B{batch}" / \
                "measurements.json"
            hf_path = directory / "tilemega_hf" / "check.json"
            mode_path = directory / "mode_check" / "mode_check.json"
            report: dict[str, object] = {"model": model, "batch": batch,
                                         "status": "pending"}
            if not all(path.exists() for path in
                       (measured, baseline_path, hf_path, mode_path)):
                result.append(report)
                continue
            tm = json.loads(measured.read_text())
            baseline = json.loads(baseline_path.read_text())
            hf = json.loads(hf_path.read_text())
            mode = json.loads(mode_path.read_text())
            tm_tpot = tpot_stats(tm, "e2e_seconds", "runs")
            vl_tpot = tpot_stats(baseline, "wall_seconds", "generation_runs")
            values = [float(row["gpu_ms"]) for row in rows(
                directory / "tilemega" / "step_times.tsv") if int(row["step"]) > 0]
            values.sort()
            report.update({
                "status": "complete", "tilemega_ttft_s": tm["ttft_seconds"],
                "vllm_ttft_s": baseline["ttft_seconds"],
                "tilemega_e2e_s": tm["e2e_seconds"],
                "vllm_e2e_s": baseline["e2e_seconds"],
                "tilemega_tpot_ms": tm["tpot_seconds"] * 1000,
                "vllm_tpot_ms": baseline["tpot_seconds"] * 1000,
                "tilemega_tpot_mean_ms": tm_tpot[0],
                "tilemega_tpot_p50_ms": tm_tpot[1],
                "tilemega_tpot_p90_ms": tm_tpot[2],
                "vllm_tpot_mean_ms": vl_tpot[0],
                "vllm_tpot_p50_ms": vl_tpot[1],
                "vllm_tpot_p90_ms": vl_tpot[2],
                "device_step_mean_ms": statistics.mean(values),
                "device_step_p50_ms": statistics.median(values),
                "device_step_p90_ms": values[math.ceil(.9 * len(values)) - 1],
                "tilemega_tokens_per_s": tm["output_tokens_per_second"],
                "vllm_tokens_per_s": baseline["output_tokens_per_second"],
                "throughput_ratio": baseline["e2e_seconds"] / tm["e2e_seconds"],
                "c1_pass": hf["pass"], "gap_le_half": hf["gap_le_0_5_ratio"],
                "max_gap": hf["max_gap"], "c2_pass": mode["pass"],
                "evidence": str(directory.relative_to(HERE)),
            })
            floor = floors.get((model, batch))
            if floor is not None:
                floor_seconds = float(floor["request_floor_s"])
                report.update({
                    "prefill_floor_s": floor["prefill_floor_s"],
                    "decode_floor_s": floor["decode_floor_s"],
                    "request_floor_s": floor_seconds,
                    "tilemega_e2e_over_floor": tm["e2e_seconds"] / floor_seconds,
                    "vllm_e2e_over_floor": baseline["e2e_seconds"] / floor_seconds,
                })
            result.append(report)
    return result


def main() -> None:
    plans = plan_table()
    requests = request_table()
    plan_columns = ["cell", "status", "solve_seconds", "mode", "measured_mode",
                    "measured_ms", "grid", "residency", "kappa", "ec", "rq",
                    "variant_count", "qkv", "o", "gate_up", "down", "lm_head",
                    "estimated_rmax", "actual_rmax", "flow_ns", "simulated_ns",
                    "search_evaluations", "price_ms", "compile_ms", "fp64_count",
                    "evidence"]
    request_columns = ["model", "batch", "status", "tilemega_ttft_s",
                       "vllm_ttft_s", "tilemega_e2e_s", "vllm_e2e_s",
                       "tilemega_tpot_ms", "vllm_tpot_ms", "device_step_mean_ms",
                       "tilemega_tpot_mean_ms", "tilemega_tpot_p50_ms",
                       "tilemega_tpot_p90_ms", "vllm_tpot_mean_ms",
                       "vllm_tpot_p50_ms", "vllm_tpot_p90_ms",
                       "device_step_p50_ms", "device_step_p90_ms",
                       "tilemega_tokens_per_s", "vllm_tokens_per_s",
                       "throughput_ratio", "c1_pass", "gap_le_half", "max_gap",
                       "c2_pass", "prefill_floor_s", "decode_floor_s",
                       "request_floor_s", "tilemega_e2e_over_floor",
                       "vllm_e2e_over_floor", "evidence"]
    emit("plans.tsv", plans, plan_columns)
    emit("requests.tsv", requests, request_columns)
    complete = [float(row["throughput_ratio"]) for row in requests
                if row["status"] == "complete"]
    (OUT / "status.json").write_text(json.dumps({
        "plans_complete": sum(row["status"] == "complete" for row in plans),
        "requests_complete": len(complete),
        "throughput_geomean": (math.exp(sum(map(math.log, complete)) / len(complete))
                               if len(complete) == 10 else None),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
