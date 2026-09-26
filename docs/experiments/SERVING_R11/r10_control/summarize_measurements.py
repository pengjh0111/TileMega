#!/usr/bin/env python3
"""Close the four current-source R10 controls from raw paired request logs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = (("llama", 1), ("llama", 16), ("qwen3", 1), ("qwen3", 16))


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarize() -> list[dict]:
    rows = []
    for model, batch in CELLS:
        cell = HERE / "ev1" / f"{model}_B{batch}"
        tm = read(cell / "tilemega/measurements.json")
        vl = read(cell / f"vllm/B{batch}/measurements.json")
        tm_hf = read(cell / "tilemega_hf/check.json")
        vl_hf = read(cell / "vllm_hf/check.json")
        mode = read(cell / "mode_check/mode_check.json")
        if not (tm["timed_tokens_identical"] if "timed_tokens_identical" in tm else
                all(r["tokens"] == next(x["tokens"] for x in tm["runs"]
                    if x["N"] == 1024 and not x["warmup"])
                    for r in tm["runs"] if r["N"] == 1024 and not r["warmup"])):
            raise AssertionError(f"{model} B{batch}: timed TileMega tokens differ")
        if not mode["pass"] or mode["mismatches"]:
            raise AssertionError(f"{model} B{batch}: L1/L2 tokens differ")
        for label, check in (("TileMega", tm_hf), ("vLLM", vl_hf)):
            if not check["pass"] or check["positions"] != batch * 1024:
                raise AssertionError(f"{model} B{batch}: {label} C-1 failed")
        rows.append(dict(model=model, batch=batch, source="4bf26fb85ada63fccea101b323f3d7a09351eca2",
            tilemega_ttft_ms=tm["ttft_seconds"] * 1000,
            vllm_ttft_ms=vl["ttft_seconds"] * 1000,
            tilemega_tpot_ms=tm["tpot_seconds"] * 1000,
            vllm_tpot_ms=vl["tpot_seconds"] * 1000,
            tilemega_e2e_s=tm["e2e_seconds"], vllm_e2e_s=vl["e2e_seconds"],
            tilemega_tok_s=tm["output_tokens_per_second"],
            vllm_tok_s=vl["output_tokens_per_second"],
            throughput_ratio=tm["output_tokens_per_second"] / vl["output_tokens_per_second"],
            tilemega_gap_le_0_5=tm_hf["gap_le_0_5_ratio"],
            tilemega_max_gap=tm_hf["max_gap"],
            vllm_gap_le_0_5=vl_hf["gap_le_0_5_ratio"],
            vllm_max_gap=vl_hf["max_gap"],
            c1_pass=True, c2_pass=True,
            raw_path=str(cell.relative_to(HERE))))
    return rows


def main() -> None:
    rows = summarize()
    with (HERE / "paired_final.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    geomean = math.exp(sum(math.log(r["throughput_ratio"]) for r in rows) / len(rows))
    result = {"source": rows[0]["source"], "cells": len(rows),
              "geometric_mean_throughput_ratio": geomean,
              "four_cell_research_gate": geomean >= 1.0,
              "original_ten_cell_gate": "unmeasured: six cells not requested in R10-C",
              "all_c1_c2": all(r["c1_pass"] and r["c2_pass"] for r in rows)}
    (HERE / "paired_final.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
