#!/usr/bin/env python3
"""Trace the selected Llama B=1/16 paged decode plans after EV-2 completes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent / "ev2" / "selected_diagnostics"
PYTHON = Path(os.environ.get("TILEMEGA_PYTHON", "/root/venvs/tilemega-torch213-cu126/bin/python"))
TILEMEGA = Path(os.environ.get("TILEMEGA_BIN", "/root/r11_work/build/tools/tilemega"))


def run(command: list[str], log: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {shlex.join(command)}; see {log}")


def build_trace(source: Path, output: Path) -> None:
    command = shlex.split((source.parent / "plan.so.build_command.txt").read_text())
    source_args = [i for i, value in enumerate(command) if value.endswith(".cu") and
                   "candidate.so.cu" in value]
    if len(source_args) != 1 or command.count("-o") != 1:
        raise ValueError("selected plan has no unique candidate compile command")
    command[source_args[0]] = str(source)
    command[command.index("-o") + 1] = str(output)
    command[1:1] = ["-DTILEMEGA_PAGE_TRACE=1", "-DTILEMEGA_TRACE_V2=1"]
    (output.parent / "build_command.json").write_text(json.dumps(command, indent=2) + "\n")
    run(command, output.parent / "build.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "runs/llama-r11-acceptance")
    args = parser.parse_args()
    plans = json.loads((args.run / "plans.json").read_text())
    target = Path(json.loads((args.run / "doctor.json").read_text())["target"])
    for batch in (1, 16):
        folder = EVIDENCE / f"llama_B{batch}"
        folder.mkdir(parents=True, exist_ok=True)
        prefill = Path(plans[str(batch)]["prefill"])
        decode = Path(plans[str(batch)]["decode"])
        mode = json.loads((decode.parent / "plan.so.plan.json").read_text())["mode"]
        source = decode.parent / "plan.so.cu"
        trace_so = folder / "trace.so"
        if not trace_so.exists():
            build_trace(source, trace_so)
        floor_steps = folder / "floor_steps.tsv"
        if not floor_steps.exists():
            run([str(TILEMEGA), "inspect", "request-floor",
                 str(decode.parent / "selected.mlir"),
                 str(target),
                 str(batch), "64", "1086", str(folder / "floor.json"), str(floor_steps)],
                folder / "floor.log")
        common = ["--model", "/root/models/llama3_2_1b", "--prefill-so", str(prefill),
                  "--decode-so", str(trace_so), "--batch", str(batch)]
        # The R11 paged L1 kernel is stage-major and does not write per-slot
        # Trace V2 rows. Its dump contains zero stamps, so the older L2 chain
        # analyzer cannot reconstruct a realized chain from it. Keep the
        # selected-plan page/launch evidence and leave chain fields null.
        if mode == "L2" and not (folder / "chain/analysis.tsv").exists():
            run([str(PYTHON), str(ROOT / "docs/experiments/SERVING_R11/trace_selected_mode.py"),
                 *common, "--past", "575", "--launches", "32", "--mode", mode,
                 "--out", str(folder / "trace_v2")],
                folder / "trace.log")
            run([sys.executable, str(ROOT / "docs/experiments/TRACE_V2/analyze.py"),
                 str(folder / "trace_v2"), "--source", str(source), "--window", "1",
                 "--out", str(folder / "chain")], folder / "analyze.log")
        if not (folder / "request/page_trace.tsv").exists():
            run([str(PYTHON), str(ROOT / "docs/experiments/SERVING_R11/measure_page_chain.py"),
                 *common, "--prompt-ids",
                 str(ROOT / "docs/experiments/SERVING_R10/prompts/llama_ids.json"),
                 "--steps", "1024", "--mode", mode, "--out", str(folder / "request")],
                folder / "request.log")
        if mode == "L1":
            (folder / "chain_unavailable.txt").write_text(
                "The selected paged L1 kernel does not stamp per-slot Trace V2 rows; "
                "a realized chain cannot be inferred from its zero-valued slot dump.\n")
        summarize = [sys.executable, str(ROOT / "docs/experiments/SERVING_R11/analyze_page_chain.py"),
                     "--page-trace", str(folder / "request/page_trace.tsv"),
                     "--model", "llama", "--batch", str(batch), "--floor-steps", str(floor_steps),
                     "--out", str(folder / "report")]
        if mode == "L2":
            summarize.extend(["--chain-analysis", str(folder / "chain/analysis.tsv")])
        run(summarize, folder / "summarize.log")
        if mode == "L1":
            report = folder / "report/page_chain_summary.json"
            data = json.loads(report.read_text())
            data['measurement_limit'] = (
                'Paged L1 records page stalls and kernel stamps, but does not '
                'stamp per-task dependency waits or per-slot chains; its zero '
                'dependency-wait counter is not a measured absence of waiting.')
            for key in ('page_full_and_dependency_wait_cta_ns',
                        'page_full_and_dependency_wait_mean_cta_ns_per_step',
                        'dependency_wait_mean_cta_ns_per_step'):
                data[key] = None
            report.write_text(json.dumps(data, indent=2) + "\n")
        print(batch, (folder / "report/page_chain_summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
