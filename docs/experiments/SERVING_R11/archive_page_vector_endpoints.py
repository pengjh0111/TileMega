#!/usr/bin/env python3
"""Freeze the four same-geometry PG-off/PG-1 full-request controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from measure_page_vector_endpoints import CELLS, FIELDS, manifest

ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("/root/r11_work/page_vector_once"))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/experiments/SERVING_R11/page_vector_e2e")
    args = parser.parse_args()
    work, out = args.work, args.out
    rows = json.loads((work / "endpoint_e2e/results.json").read_text())
    if len(rows) != 2 * len(CELLS) or any(row["exit_code"] for row in rows):
        raise ValueError("all eight full-request controls have not completed")
    out.mkdir(parents=True, exist_ok=True)
    policy = Path("/root/r11_work/pg_ablation/measurement_policy.json")
    shutil.copy2(policy, out / "measurement_policy.json")
    summary = []
    raw = []
    for model, batch in CELLS:
        pair = [row for row in rows if row["model"] == model and row["batch"] == batch]
        if {row["label"] for row in pair} != {"off", "pages"}:
            raise ValueError(f"missing PG-off/page pair for {model} B{batch}")
        by_label = {row["label"]: row for row in pair}
        off, pages = (Path(by_label[name]["binary"]) for name in ("off", "pages"))
        a, b = manifest(off), manifest(pages)
        if any(a[field] != b[field] for field in FIELDS):
            raise ValueError(f"geometry or synchronization differs for {model} B{batch}")
        if a["pg"] != "off" or b["pg"] != "pages":
            raise ValueError(f"mislabeled PG control for {model} B{batch}")
        for row in pair:
            binary = Path(row["binary"])
            if sha(binary) != row["sha256"] or not row["timed_tokens_identical"]:
                raise ValueError(f"changed binary or timed tokens for {model} B{batch}")
            source = work / "endpoint_e2e" / f"{model}_B{batch}" / row["label"]
            data = json.loads((source / "measurements.json").read_text())
            if data["measurement_policy"] != json.loads(policy.read_text()):
                raise ValueError(f"measurement policy changed for {model} B{batch}")
            if len([r for r in data["runs"] if r["N"] == 1024 and not r["warmup"]]) != 3:
                raise ValueError(f"missing three timed 1024-token rounds for {model} B{batch}")
            if len([r for r in data["runs"] if r["N"] == 1 and not r["warmup"]]) != 3:
                raise ValueError(f"missing three timed TTFT rounds for {model} B{batch}")
            guard = [json.loads(line) for line in (source / "guard.jsonl").read_text().splitlines()]
            if not guard or any("label" not in item for item in guard):
                raise ValueError(f"missing round-level guard observations for {model} B{batch}")
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    raw.append((path, f"{model}_B{batch}/{row['label']}/{path.relative_to(source)}"))
        summary.append(dict(model=model, batch=batch,
                            off_e2e_s=by_label["off"]["e2e_seconds"],
                            pages_e2e_s=by_label["pages"]["e2e_seconds"],
                            pages_over_off=by_label["pages"]["e2e_seconds"] /
                                by_label["off"]["e2e_seconds"],
                            off_sha256=by_label["off"]["sha256"],
                            pages_sha256=by_label["pages"]["sha256"]))
    with (out / "summary.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    with tarfile.open(out / "raw.tar.xz", "w:xz") as archive:
        for path, name in raw:
            info = archive.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    (out / "README.md").write_text(
        "# Fixed-geometry PG-1 endpoint controls\n\n"
        "Each model uses B=1 and B=16, as requested. Within each cell, the\n"
        "PG-off and optimized 16 KiB PG-1 arms share GEMM geometry, grid,\n"
        "residency, κ, attention coordinates and sync settings. The two arms\n"
        "were timed in one session under the same predeclared contamination\n"
        "policy. Three full-request timings and three TTFT timings per arm\n"
        "follow one warmup each. The raw archive retains per-round guard,\n"
        "clock, token and step-time records. These are fixed-geometry controls,\n"
        "not final solver-selected EV-2 results.\n\n"
        f"Raw archive SHA256: `{sha(out / 'raw.tar.xz')}`.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
