#!/usr/bin/env python3
"""Feed measured per-plan resources into the attention candidate solver."""
import csv
import io
import json
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
archive = Path(__file__).resolve().parent / "attention_models"
output = Path(__file__).resolve().parent / "attention_dp"
output.mkdir(exist_ok=False)
rows = list(csv.DictReader((archive / "correctness.tsv").open(), delimiter="\t"))
status = output / "status.txt"
status.write_text("RUNNING\n")
try:
    for model in ("gqa2", "mha4"):
        candidates = output / (model + "_candidates.tsv")
        with candidates.open("w") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["chunks", "extent", "registers", "tile_m", "tile_n", "tile_k", "stages", "split"])
            for chunk in (1, 2, 4, 8):
                plans = json.loads((archive / f"plans/{model}_c{chunk}.json").read_text())["variants"]
                assert len(plans) == 1
                plan = plans[0]
                shape = plan["uniform"]
                cells = [row for row in rows if row["model"] == model and int(row["chunk"]) == chunk]
                assert len(cells) == 100
                registers = {int(row["reg"]) for row in cells}
                extent = {choice["chunk_extent"] for choice in plan["attention"]}
                assert len(registers) == len(extent) == 1
                writer.writerow([chunk, extent.pop(), registers.pop(),
                    *[shape[key] for key in ("tile_m", "tile_n", "tile_k", "stages", "split_k")]])
        run = subprocess.run([str(repo / "build-portable/tools/tilemega-attention-dp"), str(repo), model, str(candidates)],
                             capture_output=True, text=True, timeout=1800)
        (output / (model + ".tsv")).write_text(run.stdout)
        (output / (model + ".txt")).write_text(run.stderr)
        run.check_returncode()
        alternatives = list(csv.DictReader(io.StringIO(run.stdout), delimiter="\t"))
        assert len(alternatives) == 8 and "exact_plan_minimum=PASS remaining=0" in run.stderr
        print(model, "attention DP selection passed", flush=True)
    status.write_text("PASS 2 models x 2 seq x 4 measured plans; exact minimum and direct-price bits\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
