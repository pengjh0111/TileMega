#!/usr/bin/env python3
"""Freeze current-source PG-1 endpoint diagnostics with their raw provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
CELLS = (("llama", 1), ("llama", 16), ("qwen3", 1), ("qwen3", 16))
SUMMARY = (
    "model", "batch", "steps", "measured_chain_past", "measured_chain_links",
    "measured_chain_span_ns", "measured_chain_floor_ns",
    "measured_chain_over_floor", "measured_chain_residual_bubble_ns_per_link",
    "page_full_and_dependency_wait_mean_cta_ns_per_step",
    "page_full_and_dependency_wait_cta_ns", "launch_gap_mean_ns",
    "launch_gap_p50_ns",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("/root/r11_work/page_vector_once"))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/experiments/SERVING_R11/page_vector_diagnostics")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--protocol", type=Path, action="append")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    work, out = args.work, args.out
    cases = json.loads((args.cases or work / "all_cases.json").read_text())
    if len(cases) != len(CELLS):
        raise ValueError("the four endpoint case table is incomplete")
    protocol_paths = args.protocol or [work / name / "summary.json"
                                       for name in ("fresh50", "endpoint_fresh50")]
    protocol = [json.loads(path.read_text()) for path in protocol_paths]
    if not all(item.get("complete") and item.get("failed") == 0 for item in protocol):
        raise ValueError("the current-source page protocol did not pass")
    if sum(item["passed"] for item in protocol) != 200:
        raise ValueError("expected four cells with 50 fresh processes each")
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol_summary.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = []
    provenance = []
    raw = []
    results = args.results or work / "diagnostic_results"
    for index, (model, batch) in enumerate(CELLS):
        name = f"{model}_B{batch}"
        case = cases[index]
        if case["batch"] != batch or model not in case["model"]:
            raise ValueError(f"case order changed at {name}")
        source = results / name
        report = source / "report"
        summary = json.loads((report / "page_chain_summary.json").read_text())
        if summary["model"] != model or summary["batch"] != batch or summary["steps"] != 1023:
            raise ValueError(f"incomplete diagnostic summary for {name}")
        if summary["measured_chain_past"] != 575 or summary["measured_chain_links"] <= 0:
            raise ValueError(f"missing realized dependency chain for {name}")
        step_rows = list(csv.DictReader((report / "page_chain.tsv").open(), delimiter="\t"))
        if len(step_rows) != 1023 or [int(row["step"]) for row in step_rows] != list(range(1023)):
            raise ValueError(f"missing exact per-step diagnostic at {name}")
        destination = out / name
        destination.mkdir(exist_ok=True)
        for filename in ("page_chain.tsv", "page_chain_summary.json"):
            shutil.copy2(report / filename, destination / filename)
        summaries.append({field: summary[field] for field in SUMMARY})
        diagnostic_binary = work / "diagnostics" / name / "trace.so"
        provenance.append(dict(cell=name, page_binary_sha256=sha(Path(case["decode"])),
                               trace_binary_sha256=sha(diagnostic_binary),
                               floor_source=json.loads((source / "floor_source.json").read_text())))
        for path in sorted(source.rglob("*")):
            if path.is_file():
                raw.append((path, f"{name}/{path.relative_to(source)}"))
    with (out / "summary.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY, delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    archive = out / "raw.tar.xz"
    with tarfile.open(archive, "w:xz") as target:
        for path, name in raw:
            info = target.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            with path.open("rb") as stream:
                target.addfile(info, stream)
    (out / "README.md").write_text(
        "# Current-source PG-1 diagnostics at batch endpoints\n\n"
        "The four cells use B=1/16 as requested. `summary.tsv` gives the realized\n"
        "chain at past=575 beside the exact CG-derived DRAM floor, the residual\n"
        "bubble per chain link, per-CTA time with a full page ring while waiting\n"
        "for dependencies, and adjacent decode-launch gaps. Each cell's\n"
        "`page_chain.tsv` retains all 1023 exact per-step floors and measurements.\n"
        "The chain length is observed at past=575 and reused as the denominator\n"
        "for other steps; it is not remeasured at every past value. Page stall\n"
        "times are CTA-local and cannot be summed into request wall time.\n"
        "Trace instrumentation perturbs timing; non-instrumented performance\n"
        "is measured separately with the same geometry. `raw.tar.xz` preserves\n"
        "the raw page/chain traces, floor evaluation, commands, and guard logs.\n"
        f"Raw archive SHA256: `{sha(archive)}`.\n"
    )
    print(json.dumps(dict(cells=len(summaries), archive=str(archive),
                          sha256=sha(archive)), indent=2))


if __name__ == "__main__":
    main()
