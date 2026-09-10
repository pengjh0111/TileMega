#!/usr/bin/env python3
"""CPU access/price evidence; no GPU timing or correctness claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    binary = repo / "build-portable/tools/tilemega-scalar-cost-probe"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists() and not args.resume:
        raise RuntimeError("refusing to replace an evidence run")
    metadata = {"binary": str(binary), "binary_sha256": sha(binary),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "cases": [], "purpose": "CPU TaskBody index emulation and symbolic scalar pricing"}
    if args.resume:
        metadata = json.loads(manifest.read_text())
        if sha(binary) != metadata["binary_sha256"]:
            raise RuntimeError("cannot resume with a changed probe")
    else:
        (out / "source.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=repo))
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    for dtype in ("bf16", "f32"):
        for model in ("gqa2", "mha4"):
            # Historical FP32 exports carry [1,8] domains, unlike SEQSCAN.
            # Keep those inputs unchanged; exercise long-context BF16 only.
            cases = ((1, 512), (4, 3), (128, 3), (512, 512)) if dtype == "bf16" else ((1, 3), (4, 3), (8, 8))
            for seq, past in cases:
                if sha(binary) != metadata["binary_sha256"]:
                    raise RuntimeError("probe changed during the run")
                stem = f"{dtype}_{model}_s{seq}_p{past}"
                command = [str(binary), str(repo), model, dtype, str(seq), str(past)]
                previous = [row for row in metadata["cases"] if row["command"] == command]
                if previous:
                    row = previous[-1]
                    if row["returncode"] or any(sha(out / row[key]) != row[key + "_sha256"] for key in ("stdout", "stderr")):
                        raise RuntimeError("existing evidence failed or changed; use a new output directory")
                    continue
                stdout, stderr = out / (stem + ".tsv"), out / (stem + ".txt")
                with stdout.open("xb") as output, stderr.open("xb") as errors:
                    result = subprocess.run(command, stdout=output, stderr=errors, timeout=1800)
                row = {"command": command, "returncode": result.returncode,
                       "stdout": stdout.name, "stdout_sha256": sha(stdout),
                       "stderr": stderr.name, "stderr_sha256": sha(stderr)}
                metadata["cases"].append(row)
                manifest.write_text(json.dumps(metadata, indent=2) + "\n")
                if result.returncode or "ISL_CONTEXT remaining=0" not in stderr.read_text():
                    raise RuntimeError(f"scalar verification failed: {stem}")
                print(stem, "PASS", flush=True)
    passed = sum(row["returncode"] == 0 for row in metadata["cases"])
    rejected = len(metadata["cases"]) - passed
    (out / "status.txt").write_text(f"CPU_CASES {passed}/14\nRETAINED_REJECTIONS {rejected}\n")


if __name__ == "__main__":
    main()
