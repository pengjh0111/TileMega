#!/usr/bin/env python3
"""Compile separate trace builds and inspect four decode midpoint cells."""
from __future__ import annotations

import fcntl
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = Path("/root/r10_work")
TORCH = "/root/venvs/tilemega-torch213-cu126/bin/python"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def gemm_class(name: str, layer_count: int, per_layer: int,
               category: str) -> str:
    if category == "fused_attention":
        return "attention"
    if "gemm" not in category:
        return category
    original = int(name.split("serving.s", 1)[1].split(".", 1)[0])
    if original == per_layer * layer_count + 2:
        return "lm_head"
    if 1 <= original <= per_layer * layer_count:
        offsets = ({1: "qkv", 4: "o", 6: "gate_up", 7: "down"}
                   if per_layer == 8 else
                   {1: "qkv", 3: "o", 5: "gate_up", 6: "down"})
        return offsets.get((original - 1) % per_layer, category)
    return category


def bandwidth_rows(model: str, batch: int, output: Path) -> list[dict]:
    layer_count = int(json.loads((HERE / "model_sources.json").read_text())[
        model]["dimensions"]["num_hidden_layers"])
    plan = json.loads((HERE / "plans" / output.name /
                       "plan.so.plan.json").read_text())
    per_layer = 7 if plan["attention_kv_block"] >= plan["capacity"] else 8
    traces = tsv(output / "analysis" / f"{output.name}.task_spaces.tsv")
    parts = tsv(HERE / "plans" / output.name / "winner.flow_parts.tsv")
    by_stage: dict[int, dict] = {}
    for row in parts:
        stage = int(row["stage"])
        item = by_stage.setdefault(stage, {"name": row["name"],
                                           "category": row["category"],
                                           "bytes": 0.0})
        item["bytes"] += int(row["tasks"]) * float(
            row["no_producer_dram_bytes_per_task"])
    aggregate: dict[str, dict] = {}
    for row in traces:
        stage = int(row["stage"])
        item = by_stage.get(stage)
        if item is None:
            raise ValueError(f"trace stage {stage} has no selected flow piece")
        category = gemm_class(item["name"], layer_count, per_layer,
                              item["category"])
        span = int(row["last_end_ns"]) - int(row["first_start_ns"])
        if span <= 0:
            continue
        total = aggregate.setdefault(category, {"stages": 0, "tasks": 0,
                                                 "bytes": 0.0, "span": 0})
        total["stages"] += 1
        total["tasks"] += int(row["tasks"])
        total["bytes"] += item["bytes"]
        total["span"] += span
    return [{"model": model, "batch": batch, "category": category,
             "stages": values["stages"], "tasks": values["tasks"],
             "no_producer_read_bytes": round(values["bytes"]),
             "stage_span_ns": values["span"],
             "effective_gbps": values["bytes"] / values["span"],
             "measurement": "CG no-producer bytes / trace stage span",
             "trace_mode": "L2 diagnostic",
             "evidence": str(output / "analysis" /
                 f"{output.name}.task_spaces.tsv")}
            for category, values in sorted(aggregate.items())]


def chain_rows(model: str, batch: int, output: Path) -> list[dict]:
    layer_count = int(json.loads((HERE / "model_sources.json").read_text())[
        model]["dimensions"]["num_hidden_layers"])
    plan = json.loads((HERE / "plans" / output.name /
                       "plan.so.plan.json").read_text())
    per_layer = 7 if plan["attention_kv_block"] >= plan["capacity"] else 8
    parts = tsv(HERE / "plans" / output.name / "winner.flow_parts.tsv")
    stages = {int(row["stage"]): (row["name"], row["category"])
              for row in parts}
    path = output / "analysis" / f"{output.name}.chain_links.tsv"
    grouped: dict[str, dict[str, int]] = {}
    for row in tsv(path):
        stage = int(row["stage"])
        name, kind = stages[stage]
        category = gemm_class(name, layer_count, per_layer, kind)
        item = grouped.setdefault(category, {"links": 0, "wall_ns": 0,
            "task_ns": 0, "publication_ns": 0, "wait_hop_ns": 0,
            "barrier_ns": 0, "idle_ns": 0})
        item["links"] += 1
        for column, source in (("wall_ns", "wall_ns"),
                               ("task_ns", "task_fixed_plus_mainloop_ns"),
                               ("publication_ns", "publish_ns"),
                               ("wait_hop_ns", "wait_hop_ns"),
                               ("barrier_ns", "barrier_ns"),
                               ("idle_ns", "idle_ns")):
            item[column] += int(row[source])
    span = int(tsv(output / "analysis" / "analysis.tsv")[0]["kernel_span_ns"])
    return [{"model": model, "batch": batch, "category": category,
             **values, "kernel_span_ns": span,
             "wall_share": values["wall_ns"] / span,
             "trace_mode": "L2 diagnostic", "evidence": str(path)}
            for category, values in sorted(grouped.items())]


def trace_one(model: str, batch: int) -> dict:
    name = f"{model}_decode_B{batch}"
    original = WORK / "plans" / name / "plan.so"
    prefill = WORK / "plans" / f"{model}_prefill_B{batch}" / "plan.so"
    if not original.exists() or not prefill.exists():
        raise FileNotFoundError(f"serving plan missing: {name}")
    output = WORK / "serving_trace" / name
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "trace.so"
    command = shlex.split(Path(str(original) + ".build_command.txt").read_text())
    command.insert(1, "-DTILEMEGA_TRACE_V2=1")
    command[command.index("-o") + 1] = str(binary)
    if not binary.exists():
        with (output / "build.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"trace nvcc failed for {name}: {result.returncode}")
    (output / "build_command.json").write_text(json.dumps(command, indent=2) + "\n")
    audit = output / "fp64_audit.json"
    if not audit.exists():
        with (output / "fp64_audit.log").open("w") as log:
            result = subprocess.run([
                sys.executable, str(HERE / "audit_sass.py"), str(binary),
                "--out", str(audit)], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"trace FP64 audit failed for {name}")
    if json.loads(audit.read_text())["fp64_total"] != 0:
        raise RuntimeError(f"trace build contains FP64 instructions: {name}")
    checkpoint = WORK.parent / "models" / (
        "llama3_2_1b" if model == "llama" else "qwen3_1_7b")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    if not (output / "trace.json").exists():
        lock_path = WORK / "serving_gpu.lock"
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (output / "run.log").open("w") as log:
                result = subprocess.run([
                    TORCH, "-m", "tilemega.serving.trace",
                    "--model", str(checkpoint), "--prefill-so", str(prefill),
                    "--decode-so", str(binary), "--batch", str(batch),
                    "--past", "575", "--launches", "32", "--out", str(output),
                ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"serving trace failed for {name}: {result.returncode}")
    analysis = output / "analysis"
    if not (analysis / "analysis.tsv").exists():
        with (output / "analyze.log").open("w") as log:
            result = subprocess.run([
                sys.executable, str(ROOT / "docs/experiments/TRACE_V2/analyze.py"),
                str(output), "--out", str(analysis),
                "--source", str(Path(str(original) + ".cu")),
                "--window", "1",
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (output / "analysis_failed.txt").write_text(
                f"analyze.py exited {result.returncode}; see analyze.log\n")
    report = json.loads((output / "trace.json").read_text())
    report["cell"] = name
    report["trace_so_sha256"] = digest(binary)
    report["raw_trace"] = str(output)
    report["analyzed"] = (analysis / "analysis.tsv").exists()
    return report


def main() -> int:
    results = []
    rates = []
    chains = []
    for model in ("llama", "qwen3"):
        for batch in (1, 16):
            try:
                row = trace_one(model, batch)
            except Exception as error:
                row = {"cell": f"{model}_decode_B{batch}", "error": str(error)}
            results.append(row)
            if row.get("analyzed"):
                output = WORK / "serving_trace" / row["cell"]
                try:
                    rates.extend(bandwidth_rows(model, batch, output))
                    chains.extend(chain_rows(model, batch, output))
                except Exception as error:
                    row["summary_error"] = str(error)
            print(json.dumps(row), flush=True)
    (HERE / "trace_status.json").write_text(json.dumps(results, indent=2) + "\n")
    if rates:
        with (HERE / "trace_bandwidth.tsv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rates[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rates)
    if chains:
        with (HERE / "trace_chain.tsv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(chains[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(chains)
    return int(any("error" in row or "summary_error" in row for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
