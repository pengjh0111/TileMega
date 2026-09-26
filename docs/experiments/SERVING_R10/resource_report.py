#!/usr/bin/env python3
"""Extract selected-kernel resources and top-three rankings from raw plans."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
OUT = HERE / "report_tables"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def emit(name: str, data: list[dict], columns: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)


def kernel_resources(log: str, name: str) -> tuple[int, int, int]:
    entry = re.search(r"Compiling entry function '[^']*" + name +
                      r"[^']*'[^\n]*\n(.*?)(?=ptxas info\s+: Compiling entry|\Z)",
                      log, re.DOTALL)
    if entry is None:
        raise ValueError(f"ptxas log has no {name} entry")
    registers = re.search(r"Used (\d+) registers", entry.group(1))
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",
                       entry.group(1))
    if registers is None or spills is None:
        raise ValueError(f"ptxas log has incomplete {name} resources")
    return int(registers.group(1)), int(spills.group(1)), int(spills.group(2))


def task_smem_bytes(plan: dict, generated: str) -> int:
    # These are the compile-time storage forms checked by static_assert in
    # ModelHarness.cuh and FusedAttentionTaskBody.h. The legacy attention union
    # has a 4096-float default unless the generated source overrides it.
    extent = re.search(r"#define TILEMEGA_ATTENTION_SCRATCH_EXTENT (\d+)", generated)
    legacy_attention = 4 * (int(extent.group(1)) if extent else 4096)
    head = re.search(r"#define TILEMEGA_SERVING_HEAD_DIM (\d+)", generated)
    if head is None:
        raise ValueError("generated serving source has no head dimension")
    head_dim = int(head.group(1))
    serving_attention = 416 * head_dim + 6272
    gemm = max(max(2 * g["stages"] * g["tile_k"] *
                       (g["tile_m"] + g["tile_n"]),
                       4 * g["tile_m"] * g["tile_n"])
               for g in plan["gemms"])
    return max(gemm, serving_attention, legacy_attention, 4 * 128)


def main() -> None:
    resources: list[dict] = []
    rankings: list[dict] = []
    for model in ("llama", "qwen3"):
        for phase in ("prefill", "decode"):
            for batch in (1, 2, 4, 8, 16):
                cell = f"{model}_{phase}_B{batch}"
                directory = HERE / "plans" / cell
                manifest = directory / "plan.so.plan.json"
                if not manifest.exists():
                    continue
                plan = json.loads(manifest.read_text())
                log_path = directory / "plan.so.ptxas.log"
                generated_path = Path("/root/r10_work/plans") / cell / "plan.so.cu"
                if log_path.exists() and generated_path.exists():
                    log = log_path.read_text()
                    l1 = kernel_resources(log, "tilemega_l1_kernel")
                    l2 = kernel_resources(log, "tilemega_l2_kernel")
                    resources.append({
                        "cell": cell, "task_smem_bytes": task_smem_bytes(
                            plan, generated_path.read_text()),
                        "l1_registers": l1[0], "l2_registers": l2[0],
                        "l1_spill_store_bytes": l1[1],
                        "l1_spill_load_bytes": l1[2],
                        "l2_spill_store_bytes": l2[1],
                        "l2_spill_load_bytes": l2[2],
                        "residency": plan["residency"],
                        "evidence": str(log_path.relative_to(HERE)),
                    })
                measured = directory / "plan.so.top3_measured.tsv"
                estimates = directory / "plan.so.resources.tsv"
                if measured.exists() and estimates.exists():
                    by_rank = {int(row["rank"]): row for row in read_tsv(estimates)}
                    for row in read_tsv(measured):
                        predicted = by_rank[int(row["rank"])]
                        rankings.append({
                            "cell": cell, "rank": row["rank"], "mode": row["mode"],
                            "flow_ms": float(predicted["flow_ns"]) / 1e6,
                            "fluid_ms": float(predicted["simulated_ns"]) / 1e6,
                            "measured_ms": row["mean_ms"],
                            "estimated_rmax": predicted["estimated"],
                            "actual_rmax": predicted["actual"],
                            "re_solved": predicted["re_solved"],
                            "evidence": str(measured.relative_to(HERE)),
                        })
    emit("kernel_resources.tsv", resources, [
        "cell", "task_smem_bytes", "l1_registers", "l2_registers",
        "l1_spill_store_bytes", "l1_spill_load_bytes",
        "l2_spill_store_bytes", "l2_spill_load_bytes", "residency", "evidence"])
    emit("top3_rankings.tsv", rankings, [
        "cell", "rank", "mode", "flow_ms", "fluid_ms", "measured_ms",
        "estimated_rmax", "actual_rmax", "re_solved", "evidence"])
    print(f"resources {len(resources)}/20; top-three mode rows {len(rankings)}")


if __name__ == "__main__":
    main()
