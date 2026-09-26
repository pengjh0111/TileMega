#!/usr/bin/env python3
"""Evaluate the CG's symbolic serving floor over a complete request."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
WORK = Path("/root/r10_work/plans")
OUT = HERE / "report_tables"
FIELD = re.compile(r'tmexec\.dram_floor = \{compute_ns = "([^"]+)".*?'
                   r'dram_ns = "([^"]+)"')
ROLES = re.compile(r'tilemega\.dimension_roles = '
                   r'\{batch = "([^"]*)", past = "([^"]*)", seq = "([^"]*)"\}')


def selected_mlir(model: str, phase: str, batch: int) -> Path:
    base = WORK / f"{model}_{phase}_B{batch}" / "plan.so"
    with Path(str(base) + ".top3_measured.tsv").open() as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    best = min(records, key=lambda row: float(row["mean_ms"]))
    return Path(str(base) + f".top{best['rank']}.mlir")


def parse(path: Path) -> tuple[dict[str, str], str, str, float]:
    source = path.read_text()
    roles = ROLES.search(source)
    fields = FIELD.search(source)
    solved = re.search(r'floor_value_ns = ([0-9.]+) : f64', source)
    if not (roles and fields and solved):
        raise ValueError(f"CG floor or dimension roles missing in {path}")
    return (dict(zip(("batch", "past", "seq"), roles.groups())),
            fields.group(1), fields.group(2), float(solved.group(1)))


def evaluate(piecewise: str, role_map: dict[str, str], batch: int,
             past: int, seq: int) -> float:
    variables = {symbol: {"batch": batch, "past": past, "seq": seq}[role]
                 for role, symbol in role_map.items() if symbol.startswith("s")}
    start = piecewise.index("{") + 1
    end = piecewise.rindex("}")
    for clause in piecewise[start:end].split(";"):
        if not clause.strip():
            continue
        expression, sep, condition = clause.strip().rpartition(" : ")
        if not sep:
            expression, condition = clause.strip(), "True"
        condition = re.sub(r"(?<![<>=!])=(?!=)", "==", condition)
        if eval(condition, {"__builtins__": {}}, variables):
            return float(eval(expression, {"__builtins__": {}}, variables))
    raise ValueError(f"no ISL piece covers B={batch}, past={past}: {piecewise}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    requests = []
    points = []
    expressions = []
    for model in ("llama", "qwen3"):
        for batch in (1, 2, 4, 8, 16):
            if not all((WORK / f"{model}_{phase}_B{batch}" /
                        "plan.so.plan.json").exists()
                       for phase in ("prefill", "decode")):
                continue
            prefill_path = selected_mlir(model, "prefill", batch)
            decode_path = selected_mlir(model, "decode", batch)
            pre_roles, pre_compute, pre_dram, pre_solved = parse(prefill_path)
            dec_roles, dec_compute, dec_dram, dec_solved = parse(decode_path)
            floor = lambda roles, compute, dram, past, seq: max(
                evaluate(compute, roles, batch, past, seq),
                evaluate(dram, roles, batch, past, seq))
            prefill_ns = floor(pre_roles, pre_compute, pre_dram, 0, 64)
            decode = [floor(dec_roles, dec_compute, dec_dram, past, 1)
                      for past in range(64, 1087)]
            if not math.isclose(prefill_ns, pre_solved, rel_tol=1e-10):
                raise AssertionError(f"{model} B{batch} prefill floor differs from CG")
            if not math.isclose(decode[575 - 64], dec_solved, rel_tol=1e-10):
                raise AssertionError(f"{model} B{batch} decode floor differs from CG")
            requests.append({
                "model": model, "batch": batch,
                "prefill_floor_s": prefill_ns / 1e9,
                "decode_floor_s": sum(decode) / 1e9,
                "request_floor_s": (prefill_ns + sum(decode)) / 1e9,
                "prefill_cg": str(prefill_path), "decode_cg": str(decode_path),
            })
            for label, past in (("p_lo", 64), ("p_mid", 575), ("p_hi", 1086)):
                points.append({"model": model, "batch": batch,
                               "point": label, "past": past,
                               "floor_ms": decode[past - 64] / 1e6})
            expressions.append({"model": model, "batch": batch,
                                "prefill_dram_ns": pre_dram,
                                "prefill_compute_ns": pre_compute,
                                "decode_dram_ns": dec_dram,
                                "decode_compute_ns": dec_compute})
    if not requests:
        raise RuntimeError("no paired serving plans are available")
    for name, values in (("request_floors.tsv", requests),
                         ("floor_points.tsv", points),
                         ("floor_expressions.tsv", expressions)):
        with (OUT / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps({"requests": len(requests), "points": len(points),
                      "cg_value_checks": 2 * len(requests)}))


if __name__ == "__main__":
    main()
