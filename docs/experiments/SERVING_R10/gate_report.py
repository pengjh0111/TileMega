#!/usr/bin/env python3
"""Recompute R10 G-1..G-12 from raw verification and experiment artifacts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
OUT = HERE / "report_tables"


def tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def json_or_none(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def gate(number: int, status: str, value: object, evidence: str) -> dict:
    return {"gate": f"G-{number}", "status": status,
            "value": value, "evidence": evidence}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verify = subprocess.run([sys.executable, str(HERE / "verify.py")],
                            capture_output=True, text=True)
    (OUT / "verify_output.txt").write_text(verify.stdout + verify.stderr)
    checks = re.findall(r"^K-\d+ (PASS|FAIL)$", verify.stdout, re.MULTILINE)
    result = [gate(1, "PASS" if len(checks) == 16 and all(
        item == "PASS" for item in checks) else "FAIL",
        f"{sum(item == 'PASS' for item in checks)}/16 code checks",
        "report_tables/verify_output.txt")]

    gemm = json_or_none(HERE / "task_body_tests/gemm_matrix/summary.json")
    prefill = json_or_none(HERE /
        "task_body_tests/prefill_attention/torch_comparison.json")
    decode_log = (HERE / "task_body_tests/attention_decode_matrix.log").read_text()
    ctests = (HERE / "ctest_partial_interrupted.txt").read_text()
    ctest_numbers = {int(x) for x in re.findall(
        r"^\s*(\d+)/80 Test\s+#\d+: .*Passed", ctests, re.MULTILINE)}
    ctest_numbers.update(range(33, 81)) if all(
        "100% tests passed" in (HERE / name).read_text()
        for name in ("ctest_cpu_33_54.txt", "ctest_gpu_55_65.txt",
                     "ctest_cpu_66_80.txt")) else None
    g2 = bool(gemm and gemm.get("pass") and gemm.get("cases") == 1344 and
              isinstance(prefill, list) and len(prefill) == 8 and
              all(row["violations"] == 0 for row in prefill) and
              "60/60 pass" in decode_log and len(ctest_numbers) == 80)
    result.append(gate(2, "PASS" if g2 else "FAIL",
        f"GEMM {gemm.get('cases') if gemm else '?'}; attention 60+"
        f"{len(prefill) if isinstance(prefill,list) else '?'}; CTest {len(ctest_numbers)}/80",
        "task_body_tests/; ctest_partial_interrupted.txt; ctest_cpu_*.txt"))

    requests = tsv(OUT / "requests.tsv")
    complete = [row for row in requests if row.get("status") == "complete"]
    c1 = len(complete) == 10 and all(row["c1_pass"] == "True" for row in complete)
    timed = [json_or_none(HERE / "ev1" / row["model"] /
            f"B{row['batch']}" / "result.json") for row in complete]
    c2 = len(complete) == 10 and all(row["c2_pass"] == "True" for row in
        complete) and all(item and item.get("timed_tokens_equal") for item in timed)
    result.append(gate(3, "PASS" if c1 else ("PENDING" if len(complete) < 10
        else "FAIL"), f"{sum(row.get('c1_pass') == 'True' for row in complete)}/10",
        "ev1/*/B*/tilemega_hf/check.json"))
    result.append(gate(4, "PASS" if c2 else ("PENDING" if len(complete) < 10
        else "FAIL"), f"mode {sum(row.get('c2_pass') == 'True' for row in complete)}/10; "
        f"timed {sum(bool(item and item.get('timed_tokens_equal')) for item in timed)}/10",
        "ev1/*/B*/mode_check/mode_check.json; ev1/*/B*/result.json"))

    audits = list((HERE / "plans").glob("*/fp64_audit.json"))
    counts = [json.loads(path.read_text())["fp64_total"] for path in audits]
    result.append(gate(5, "PASS" if len(counts) == 20 and max(counts) == 0
        else ("PENDING" if len(counts) < 20 else "FAIL"),
        f"{len(counts)}/20 audits; max FP64={max(counts) if counts else '?'}",
        "plans/*/fp64_audit.json"))

    incremental = json_or_none(HERE / "incremental_equivalence/report.json")
    pruning = json_or_none(HERE / "pruning_equivalence/report.json")
    inc_good = bool(incremental and all(row.get("pass") for row in
                                        incremental.values()))
    prune_good = bool(pruning and all(row.get("score_gate") and
        row.get("domain_membership_gate") for row in pruning.values()) and
        set(pruning) == {"llama", "qwen3"})
    result.append(gate(6, "PASS" if inc_good and prune_good else
        ("PENDING" if pruning is None else "FAIL"),
        f"incremental={inc_good}; pruning={prune_good}",
        "incremental_equivalence/report.json; pruning_equivalence/report.json"))

    plans = tsv(OUT / "plans.tsv")
    solved = [row for row in plans if row.get("status") == "complete"]
    over = [row for row in solved if float(row["solve_seconds"]) > 600]
    result.append(gate(7, "FAIL" if over else ("PASS" if len(solved) == 20
        else "PENDING"),
        f"{len(solved)}/20 plans; {len(over)} over 600 s; max "
        f"{max((float(row['solve_seconds']) for row in solved), default=0):.1f} s",
        "plans/*/result.json; report_tables/plans.tsv"))

    floor_audit = [row for row in tsv(HERE / "floor_audit/verify_output.tsv")
                   if row.get("model") in {"llama", "qwen3"}]
    floor_ok = len(floor_audit) == 4 and all(row["status"] == "PASS" and
        float(row["weight_error"]) <= .005 and float(row["KV_error"]) <= .005
        for row in floor_audit)
    result.append(gate(8, "PASS" if floor_ok else "FAIL",
        f"{sum(row['status'] == 'PASS' for row in floor_audit)}/4 endpoints",
        "floor_audit/verify_output.tsv"))

    ratios = [float(row["throughput_ratio"]) for row in complete]
    geometric = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None
    result.append(gate(9, "PENDING" if len(ratios) < 10 else
        ("PASS" if geometric is not None and geometric >= 1 else "FAIL"),
        f"{len(ratios)}/10; geomean={geometric if geometric is not None else '?'}",
        "report_tables/requests.tsv"))

    floors = tsv(OUT / "request_floors.tsv")
    steps = tsv(OUT / "step_ratios.tsv")
    result.append(gate(10, "PASS" if len(floors) == 10 and len(steps) == 30
        else "PENDING", f"request floors {len(floors)}/10; points {len(steps)}/30",
        "report_tables/request_floors.tsv; report_tables/step_ratios.tsv"))
    gaps = tsv(OUT / "gaps.tsv")
    result.append(gate(11, "PASS" if len(gaps) == 10 else "PENDING",
        f"counterfactual gaps {len(gaps)}/10", "report_tables/gaps.tsv"))
    arch = tsv(HERE / "arch_compile/results.tsv")
    result.append(gate(12, "PASS" if len(arch) == 3 and all(
        row["exit_code"] == "0" for row in arch) else "FAIL",
        ", ".join(f"{row['arch']}={row['exit_code']}" for row in arch),
        "arch_compile/results.tsv"))

    with (OUT / "gates.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    for row in result:
        print(f"{row['gate']} {row['status']}: {row['value']} [{row['evidence']}]")


if __name__ == "__main__":
    main()
