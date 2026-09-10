#!/usr/bin/env python3
"""Independently reconcile symbolic CG counts with every archived GPU row."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    evidence = repo / "docs/experiments/COST_MODEL/attention_models"
    tool = repo / "build-portable/tools/tilemega-runtime-projection"
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.txt"
    status.write_text("RUNNING\n")
    try:
        frozen = sha(tool)
        rows = list(csv.DictReader((evidence / "correctness.tsv").open(), delimiter="\t"))
        counts = {}
        for model in ("gqa2", "mha4"):
            for chunk in (1, 2, 4, 8):
                if sha(tool) != frozen:
                    raise RuntimeError("projection tool changed during verification")
                command = [str(tool),str(repo / f"docs/experiments/SEQSCAN/raw/export/{model}.json"),
                           "256","128","1","3","128","128","16","3","1","element",
                           "1","0","0",str(chunk),str((640+chunk-1)//chunk)]
                run = subprocess.run(command,capture_output=True,text=True,timeout=900,
                                     env=dict(os.environ,TILEMEGA_ISL_AUDIT="1"))
                prefix = args.out / f"{model}_c{chunk}"
                prefix.with_suffix(".tsv").write_text(run.stdout)
                prefix.with_suffix(".txt").write_text(run.stderr)
                if run.returncode or "ISL_CONTEXT remaining=0" not in run.stderr:
                    raise RuntimeError(f"projection failed: {prefix}")
                for row in csv.DictReader(io.StringIO(run.stdout),delimiter="\t"):
                    counts[model,chunk,int(row["seq"])] = row
        checked = 0
        cells = {}
        for row in rows:
            key = row["model"],int(row["chunk"]),int(row["seq"])
            round_ = int(row["round"])
            cells.setdefault(key,set())
            if round_ in cells[key]:
                raise RuntimeError("duplicate fresh-process row")
            cells[key].add(round_)
            log = evidence / "logs" / f"{key[0]}_c{key[1]}_s{key[2]}_r{round_}.txt"
            if row["status"] != "PASS" or sha(log) != row["log_sha256"]:
                raise RuntimeError("archived process receipt changed")
            expected = counts[key]
            schedule = json.loads(row["schedule_json"])
            for measured,predicted in ((row["task_refs"],expected["task_refs"]),
                (row["waits"],expected["waits"]),
                (schedule["max_worker_task_refs"],expected["max_worker_tasks"])):
                if int(measured) != int(predicted):
                    raise RuntimeError(f"count mismatch {key}: GPU={measured} CG={predicted}")
                checked += 1
        required={(m,c,s) for m in ("gqa2","mha4") for c in (1,2,4,8) for s in (4,128)}
        if cells.keys()!=required or any(rounds!=set(range(50)) for rounds in cells.values()):
            raise RuntimeError("incomplete 800-process archive")
        (args.out / "result.json").write_text(json.dumps(dict(
            fresh_processes=len(rows),exact_count_comparisons=checked,
            tool_sha256=frozen,input_sha256=sha(evidence / "correctness.tsv"),
            scope="task_refs, waits, longest queue; not chunk pricing or GPU performance acceptance"),indent=2)+"\n")
        status.write_text(f"PASS processes={len(rows)} exact_counts={checked}/{checked}\n")
    except BaseException as error:
        status.write_text(f"STOP {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
