#!/usr/bin/env python3
"""Finish a solved shortlist when an external timing interruption killed the driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "/root/venvs/tilemega-torch213-cu126/bin/python"
COMPILER = ROOT / "build-portable/tools/tilemega-compile"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="final /root/r10_work/.../plan.so")
    parser.add_argument("--model", required=True, choices=("llama", "qwen3"))
    parser.add_argument("--phase", required=True, choices=("prefill", "decode"))
    parser.add_argument("--batch", required=True, type=int)
    args = parser.parse_args()
    prefix = args.prefix
    checkpoint = Path("/root/models") / (
        "llama3_2_1b" if args.model == "llama" else "qwen3_1_7b")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    names = [Path(str(prefix) + f".top{i}") for i in (1, 2, 3)]
    for name in names:
        for path in (Path(str(name) + ".mlir"),
                     Path(str(name) + ".candidate.so")):
            if not path.is_file():
                raise FileNotFoundError(path)
    samples: dict[tuple[int, str], list[float]] = {}
    rows: list[dict] = []
    for round_number in range(3):
        for position in range(3):
            rank = (position + round_number) % 3 + 1
            name = names[rank - 1]
            out = Path(str(name) + f".recovery.r{round_number}")
            measured = out / "measurements.json"
            command = [PYTHON, "-m", "tilemega.serving.measure_candidate",
                       "--model", str(checkpoint), "--so",
                       str(name) + ".candidate.so", "--batch", str(args.batch),
                       "--past-mid", "575" if args.phase == "decode" else "0",
                       "--out", str(out)]
            if round_number % 2:
                command.append("--reverse-modes")
            if not measured.is_file():
                out.mkdir(parents=True, exist_ok=True)
                with (out / "stdout.txt").open("w") as stdout, \
                     (out / "stderr.txt").open("w") as stderr:
                    subprocess.run(command, env=env, cwd=ROOT, check=True,
                                   stdout=stdout, stderr=stderr)
                (out / "command.json").write_text(
                    json.dumps(command, indent=2) + "\n")
            result = json.loads(measured.read_text())
            for mode, data in result["modes"].items():
                mean_ms = float(data["mean_ms"])
                samples.setdefault((rank, mode), []).append(mean_ms)
                rows.append({"round": round_number, "rank": rank,
                             "mode": mode, "mean_ms": mean_ms,
                             "evidence": str(measured)})
    medians = {key: statistics.median(values) for key, values in samples.items()}
    if not all(len(values) == 3 for values in samples.values()):
        raise RuntimeError("the shortlist needs three timing rounds per mode")
    (rank, mode), fastest = min(medians.items(), key=lambda item: item[1])
    selected = names[rank - 1]
    command = [str(COMPILER), str(selected) + ".mlir", str(prefix),
               "--serving", args.phase, "--emit", "serving", "--batch",
               str(args.batch), "--past-range",
               "64:1086" if args.phase == "decode" else "0:0",
               "--capacity", "1088"]
    with Path(str(prefix) + ".recovery.build.stdout").open("w") as stdout, \
         Path(str(prefix) + ".recovery.build.stderr").open("w") as stderr:
        subprocess.run(command, cwd=ROOT, check=True, stdout=stdout,
                       stderr=stderr)
    source_match = digest(Path(str(prefix) + ".cu")) == digest(
        Path(str(selected) + ".candidate.so.cu"))
    if not source_match:
        raise RuntimeError("recovered binary source differs from measured candidate")
    manifest_path = Path(str(prefix) + ".plan.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = args.model
    manifest["mode"] = mode
    manifest["recovered_from_top3"] = rank
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"reason": "candidate guard entered before CUDA allocated a visible context",
              "rank": rank, "mode": mode, "median_ms": fastest,
              "samples": rows, "final_compile_command": command,
              "source_identical": source_match,
              "source_sha256": digest(Path(str(prefix) + ".cu")),
              "binary_sha256": digest(prefix)}
    Path(str(prefix) + ".recovery.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("rank", "mode", "median_ms", "source_identical")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
