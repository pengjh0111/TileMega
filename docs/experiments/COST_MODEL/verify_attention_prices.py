#!/usr/bin/env python3
"""Compare full-model chunk prices with the frozen 800-process experiment."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="build-portable")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
archive = Path(__file__).resolve().parent / "attention_models"
output = args.out.resolve()
output.mkdir(exist_ok=False)
(output / "bin").mkdir()
tool = output / "bin/tilemega-attention-plan-price"
shutil.copy2(repo / args.build / "tools/tilemega-attention-plan-price", tool)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rows = list(csv.DictReader((archive / "correctness.tsv").open(), delimiter="\t"))
(output / "manifest.json").write_text(json.dumps(dict(binary_sha256=sha(tool),
    archive_sha256=sha(archive / "correctness.tsv")), indent=2) + "\n")
(output / "source.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=repo))
status = output / "status.txt"
status.write_text("RUNNING\n")
try:
    predictions = []
    for model in ("gqa2", "mha4"):
        for chunk in (1, 2, 4, 8):
            cells = [row for row in rows if row["model"] == model and int(row["chunk"]) == chunk]
            assert len(cells) == 100 and all(row["status"] == "PASS" for row in cells)
            registers = {int(row["reg"]) for row in cells}
            assert len(registers) == 1
            plan = json.loads((archive / f"plans/{model}_c{chunk}.json").read_text())
            choices = plan["variants"][0]["attention"]
            assert {choice["chunks"] for choice in choices} == {chunk}
            capacities = {choice["chunk_extent"] for choice in choices}
            assert len(capacities) == 1
            command = [str(tool), str(repo), model, str(chunk), str(capacities.pop()), str(registers.pop())]
            run = subprocess.run(command, text=True, capture_output=True, timeout=1800)
            (output / f"{model}_c{chunk}.tsv").write_text(run.stdout)
            (output / f"{model}_c{chunk}.txt").write_text(run.stderr)
            run.check_returncode()
            for prediction in csv.DictReader(io.StringIO(run.stdout), delimiter="\t"):
                selected = [row for row in cells if row["seq"] == prediction["seq"]]
                assert len(selected) == 50
                for row in selected:
                    for measured, predicted in (("ctas_per_sm", "ctas_per_sm"),
                        ("task_smem", "shared_bytes"), ("task_refs", "task_refs"), ("waits", "waits")):
                        if int(row[measured]) != int(prediction[predicted]):
                            raise RuntimeError(f"plan projection/resource mismatch: {model}/{chunk}/{row['seq']} {measured}")
                prediction["l1_ms"] = statistics.median(float(row["l1_ms"]) for row in selected)
                prediction["l2_ms"] = statistics.median(float(row["l2_ms"]) for row in selected)
                predictions.append(prediction)
            print(model, chunk, "plan/resource counts match", flush=True)
    with (output / "comparison.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)
    ranking = {}
    for model in ("gqa2", "mha4"):
        for seq in (4, 128):
            selected = [row for row in predictions if row["model"] == model and int(row["seq"]) == seq]
            ranking[f"{model}/{seq}"] = {field: [int(row["chunks"]) for row in sorted(selected, key=lambda row: float(row[field]))]
                                        for field in ("l1_ns", "l2_ns", "l1_ms", "l2_ms")}
    (output / "ranks.json").write_text(json.dumps(ranking, indent=2) + "\n")
    status.write_text("PASS 16 CPU price cells; 3200 resource/count comparisons; archived GPU medians only\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
