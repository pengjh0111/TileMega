#!/usr/bin/env python3
"""Production attention chunk arms, state-rotated fresh-process validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(text, label):
    rows = [dict(re.findall(r"(\w+)=([^\s]+)", line))
            for line in text.splitlines() if line.startswith(label + " ")]
    if len(rows) != 1:
        raise RuntimeError(f"missing or ambiguous {label}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("build", "run"), required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    out = args.out.resolve()
    if any(key.startswith("TILEMEGA_") for key in os.environ):
        raise RuntimeError("remove inherited TILEMEGA overrides")
    arms = [(model, chunk) for model in ("gqa2", "mha4") for chunk in (1, 2, 4, 8)]
    if args.phase == "build":
        out.mkdir(parents=True, exist_ok=False)
        for directory in ("bin", "src", "plans", "build", "logs"):
            (out / directory).mkdir()
        (out / ".gitignore").write_text("bin/\n")
        (out / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=repo))
        def build(arm):
            model, chunk = arm
            tag = f"{model}_c{chunk}"
            # These table indices only address the test's plan choices; all
            # task/wait counts are independently checked from CG projection.
            anchor = repo / f"docs/experiments/EVENT_COST/element_variants/src/{model}_k1.cu"
            stages = [line for line in anchor.read_text().splitlines() if line.startswith("  {TaskKind::")]
            selected = [i for i, line in enumerate(stages) if "TaskKind::kAttention," in line]
            if not selected:
                raise RuntimeError("test model has no attention stages")
            plan = dict(schema="tilemega.runtime_variants.v1", variants=[dict(
                seq_begin=1, seq_end=128,
                uniform=dict(tile_m=128, tile_n=128, tile_k=16, stages=3, split_k=1),
                attention=[dict(stage=i, chunks=chunk, chunk_extent=(128+512+chunk-1)//chunk)
                           for i in selected])])
            plan_path = out / "plans" / (tag + ".json")
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            source, binary = out / "src" / (tag + ".cu"), out / "bin" / tag
            export = repo / f"docs/experiments/SEQSCAN/raw/export/{model}.json"
            compiler = repo / "build-portable/tools/tilemega-compile"
            with (out / "build" / (tag + "_codegen.txt")).open("w") as log:
                subprocess.run([str(compiler), str(export), str(source), "--variants", str(plan_path)],
                               stdout=log, stderr=log, check=True, timeout=600)
            command = ["/usr/local/cuda/bin/nvcc", "-std=c++17", "-O2", "-arch=native", "-lineinfo",
                       "-Xptxas=-v,--warn-on-spills", "--expt-relaxed-constexpr",
                       "-DTILEMEGA_FP32_PARTIALS=1", "-DTILEMEGA_CG_SPLIT_TASK_ORDER=1",
                       *[f"-I{repo / p}" for p in ("include", "third_party/cutlass/include",
                          "third_party/cutlass/tools/util/include", "third_party/cutlass/test")],
                       str(source), str(repo / "build-portable/libtilemega.a"),
                       "-L/usr/local/cuda/lib64", "-lcudart", "-o", str(binary)]
            with (out / "build" / (tag + "_ptxas.txt")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=log, check=True, timeout=600)
            return dict(tag=tag, command=command, binary_sha256=sha(binary), source_sha256=sha(source),
                        plan_sha256=sha(plan_path), export_sha256=sha(export), anchor_sha256=sha(anchor))
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(build, arms))
        (out / "manifest.json").write_text(json.dumps(dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            builds=rows), indent=2) + "\n")
        (out / "status.txt").write_text("BUILT\n")
        return
    manifest = json.loads((out / "manifest.json").read_text())
    for build in manifest["builds"]:
        if sha(out / "bin" / build["tag"]) != build["binary_sha256"]:
            raise RuntimeError("frozen attention binary changed")
    cases = [(model, chunk, seq) for seq in (4, 128) for model, chunk in arms]
    fields = ["round", "execution_index", "model", "chunk", "seq", "hash", "status",
              "l05_ms", "l1_ms", "l2_ms", "grid", "block", "ctas_per_sm", "task_smem",
              "reg", "occupancy_smem", "static_smem", "task_refs", "waits",
              "resource_json", "schedule_json", "log_sha256"]
    with (out / "correctness.tsv").open("x") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for round_ in range(50):
            shift = round_ % len(cases)
            for index, (model, chunk, seq) in enumerate(cases[shift:] + cases[:shift]):
                tag = f"{model}_c{chunk}"
                fixture = repo / f"docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3"
                result = subprocess.run([str(out / "bin" / tag), str(fixture)],
                    capture_output=True, text=True, timeout=120,
                    env=dict(os.environ, TILEMEGA_WARMUP="5", TILEMEGA_REPEAT="11"))
                text = result.stdout + result.stderr
                log = out / "logs" / f"{tag}_s{seq}_r{round_}.txt"
                log.write_text(text)
                if result.returncode or "RESULT status=PASS" not in text:
                    (out / "status.txt").write_text(f"STOP: {log.name}\n")
                    raise RuntimeError(f"attention correctness gate failed: {log}")
                hashes = record(text, "E2E_HASH")
                if len(set(hashes.values())) != 1:
                    raise RuntimeError("attention cross-level mismatch")
                if chunk == 1:
                    anchor = repo / f"docs/experiments/EVENT_COST/element_variants/logs/{model}_k1_s{seq}_p3_r0.txt"
                    if hashes != record(anchor.read_text(), "E2E_HASH"):
                        raise RuntimeError("chunk=1 hash differs from frozen pre-chunk binary")
                time, resource, schedule = [record(text, label) for label in
                    ("E2E_TIME", "E2E_RESOURCE", "E2E_SCHEDULE")]
                row = dict(round=round_, execution_index=index, model=model, chunk=chunk,
                           seq=seq, hash=hashes["l2"], status="PASS", log_sha256=sha(log),
                           resource_json=json.dumps(resource,sort_keys=True),
                           schedule_json=json.dumps(schedule,sort_keys=True))
                for key in fields:
                    if key not in row:
                        row[key] = time.get(key, resource.get(key, schedule.get(key)))
                        if row[key] is None:
                            raise RuntimeError(f"missing output field {key}")
                writer.writerow(row)
                stream.flush()
            (out / "status.txt").write_text(f"RUNNING {round_+1}/50 rounds\n")
    (out / "status.txt").write_text("PASS 800/800 fresh processes; projection and ranking pending\n")


if __name__ == "__main__":
    main()
