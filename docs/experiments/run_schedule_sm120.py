#!/usr/bin/env python3
"""Run frozen BF16 schedule comparisons on a manually provisioned sm_120 host."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(text, label, required):
    lines = [line for line in text.splitlines() if line.startswith(label + " ")]
    if len(lines) != 1:
        raise ValueError(f"expected one {label}")
    values = dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))
    if not set(required) <= values.keys():
        raise ValueError(f"incomplete {label}: {set(required) - values.keys()}")
    return values


def validate(manifest, root, kind):
    if manifest.get("dtype") != "bf16" or manifest.get("arch") != "sm_120":
        raise ValueError("manifest must describe BF16 sm_120 binaries")
    cases = manifest["cases"]
    keys = set()
    states = {"unfused", "fused"} if kind == "fusion" else {"stage_major", "affine_balanced"}
    groups = {"gemm_elementwise", "gemm_rmsnorm"} if kind == "fusion" else {"placement"}
    for case in cases:
        key = (case["model"], case["seq"], case["group"], case["state"])
        if key in keys:
            raise ValueError("duplicate comparison cell")
        keys.add(key)
        for field in ("binary", "ptxas", "prediction", "source"):
            path = (root / case[field]).resolve()
            if digest(path) != case[field + "_sha256"]:
                raise ValueError(f"changed {field}: {path}")
        if not os.access(root / case["binary"], os.X_OK):
            raise ValueError("binary is not executable")
        if not (root / case["fixture"]).is_dir():
            raise ValueError("fixture directory is absent")
        prediction = json.loads((root / case["prediction"]).read_text())
        for field in ("total_ns", "event_ns", "ctas_per_sm"):
            value = prediction[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid prediction {field}")
        if prediction.get("state") != case["state"]:
            raise ValueError("prediction does not identify the measured state")
        log = (root / case["ptxas"]).read_text()
        if not re.search(r"Used \d+ registers", log) or not re.search(r"\d+ bytes spill stores", log):
            raise ValueError("ptxas resource/spill evidence is missing")
    expected = {(m, s, g, a) for m in ("gqa2", "mha4") for s in (4, 128)
                for g in groups for a in states}
    if keys != expected:
        raise ValueError(f"incomplete comparison matrix: missing={expected-keys}, extra={keys-expected}")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("fusion", "placement"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.txt"
    status.write_text("CHECKING\n")
    try:
        if args.rounds < 50:
            raise ValueError("correctness requires at least 50 fresh processes per cell")
        if any(key.startswith("TILEMEGA_") for key in os.environ):
            raise ValueError("remove inherited TILEMEGA overrides")
        # Check the actual selected device, not a caller-provided architecture label.
        device = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,compute_cap",
                                          "--format=csv,noheader"], text=True)
        devices = [row.strip().split(", ") for row in device.splitlines()]
        selected = os.environ.get("CUDA_VISIBLE_DEVICES")
        if selected:
            devices = [row for row in devices if selected in row[:2]]
        if len(devices) != 1 or devices[0][2] != "12.0":
            raise ValueError("select exactly one actual sm_120 GPU using CUDA_VISIBLE_DEVICES")
        (args.out / "device.txt").write_text(device)
        root = args.manifest.resolve().parent
        manifest = json.loads(args.manifest.read_text())
        cases = validate(manifest, root, args.kind)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        logs = args.out / "logs"
        logs.mkdir()
        fields = ["round", "execution_index", "model", "seq", "group", "state", "status",
                  "l05_ms", "l1_ms", "l2_ms", "prediction_json", "resource_json", "schedule_json",
                  "ptxas_sha256", "binary_sha256", "log_sha256"]
        with (args.out / "paired.tsv").open("x") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for round_ in range(args.rounds):
                # Rotate over the full product, including compilation state.
                shift = round_ % len(cases)
                for index, case in enumerate(cases[shift:] + cases[:shift]):
                    binary = root / case["binary"]
                    if digest(binary) != case["binary_sha256"]:
                        raise ValueError("binary changed during the experiment")
                    completed = subprocess.run([str(binary), str(root / case["fixture"])],
                        capture_output=True, text=True, timeout=180,
                        env=dict(os.environ, TILEMEGA_WARMUP="5", TILEMEGA_REPEAT="11"))
                    text = completed.stdout + completed.stderr
                    log = logs / f"r{round_:03d}_i{index:02d}.txt"
                    log.write_text(text)
                    if completed.returncode or "RESULT status=PASS" not in text:
                        raise RuntimeError(f"correctness failure: {log}")
                    times = record(text, "E2E_TIME", ("l05_ms", "l1_ms", "l2_ms"))
                    resource = record(text, "E2E_RESOURCE", ("reg", "block", "ctas_per_sm", "grid",
                        "task_smem", "occupancy_smem", "static_smem"))
                    schedule = record(text, "E2E_SCHEDULE", ("max_worker_span", "resident_limit",
                        "lifted_polls", "waiting_tasks", "task_refs", "waits", "i3_current"))
                    if schedule["i3_current"] != "pass" or int(resource["grid"]) > int(schedule["resident_limit"]):
                        raise RuntimeError("resident-only I3 constraint failed")
                    row = {key: case[key] for key in ("model", "seq", "group", "state",
                                                       "binary_sha256", "ptxas_sha256")}
                    row.update(round=round_, execution_index=index, status="PASS", log_sha256=digest(log),
                        resource_json=json.dumps(resource, sort_keys=True),
                        schedule_json=json.dumps(schedule, sort_keys=True),
                        prediction_json=(root / case["prediction"]).read_text().strip())
                    row.update({key: times[key] for key in ("l05_ms", "l1_ms", "l2_ms")})
                    writer.writerow(row)
                    stream.flush()
                status.write_text(f"RUNNING rounds={round_+1}/{args.rounds}\n")
        status.write_text(f"PASS correctness={args.rounds*len(cases)}/{args.rounds*len(cases)} "
                          "warmup=5 repeat=11 paired_analysis=pending\n")
    except BaseException as error:
        status.write_text(f"STOP {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
