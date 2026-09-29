#!/usr/bin/env python3
"""Start a clean R12b validation queue once, after a fixed delay and queue 4.

This file is outside the compiler source fingerprint.  It does not change the
running queue or rebuild a valid plan; queue.py owns retries and GPU guards.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "retest_3h"
PY = "/root/venvs/tilemega-torch213-cu126/bin/python"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def valid_plans(path):
    if not path.exists():
        return False
    try:
        plans = json.loads(path.read_text())
        return all(
            Path(plans[str(batch)][phase]).is_file()
            and Path(plans[str(batch)][phase] + ".plan.json").is_file()
            and (Path(plans[str(batch)][phase]).parent / "floor.json").is_file()
            for batch in (1, 16) for phase in ("prefill", "decode")
        )
    except (KeyError, TypeError, ValueError):
        return False


def prepare(model):
    original = ROOT / "runs" / f"r12b-{model}"
    source = original / "plans.json"
    if not valid_plans(source):
        print(f"{model}: plans incomplete; resume cached build", flush=True)
        code = subprocess.run(
            [PY, "-m", "tilemega", "build", "--config",
             f"configs/e2e/{model}_r12b.json", "--run-dir", str(original)],
            cwd=ROOT).returncode
        if code:
            return code
    if not valid_plans(source):
        print(f"{model}: build returned without all plan artifacts", file=sys.stderr)
        return 1
    target = ROOT / "runs" / f"r12b-retest-{model}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / "plans.json")
    print(f"{model}: reuse four compiled .so files; plans copied to {target}", flush=True)
    return 0


def await_existing_queue(pid):
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists() or b"SERVING_R12B/queue.py" not in cmdline.read_bytes():
        return
    try:
        handle = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        poller = select.poll()
        poller.register(handle, select.POLLIN)
        print(f"waiting for existing R12b queue pid={pid} to exit", flush=True)
        poller.poll()  # One blocking kernel wait, no polling loop.
    finally:
        os.close(handle)


def make_queue():
    original = json.loads((HERE / "queue.json").read_text())
    by_name = {step["name"]: step for step in original}
    steps = []
    for model, name in (("llama", "R0"), ("qwen3", "R1")):
        steps.append(dict(name=name, command=[PY, __file__, "prepare", model],
                          timeout=28800, after=[], after_any=[]))
    for name in ("Q7a", "Q7b", "Q8a", "Q8b", "Q9", "Q10", "Q12", "Q11"):
        step = json.loads(json.dumps(by_name[name]))
        command = step["command"]
        command = [part.replace("runs/r12b-llama", "runs/r12b-retest-llama")
                       .replace("runs/r12b-qwen3", "runs/r12b-retest-qwen3")
                       .replace("SERVING_R12B/protocol_cases.json",
                                "SERVING_R12B/retest_3h/protocol_cases.json")
                       .replace("SERVING_R12B/protocol --processes",
                                "SERVING_R12B/retest_3h/protocol --processes")
                       .replace("SERVING_R12B/fidelity.json",
                                "SERVING_R12B/retest_3h/fidelity.json")
                       .replace("SERVING_R12B/arch_audit",
                                "SERVING_R12B/retest_3h/arch_audit")
                       .replace("SERVING_R12B/sass_audit.json",
                                "SERVING_R12B/retest_3h/sass_audit.json")
                   for part in command]
        if name == "Q10":
            command = [PY, __file__, "s3"]
        step["command"] = command
        step["after"] = ["R0"] if name in ("Q7a", "Q9", "Q10") else (
            ["R1"] if name == "Q7b" else (
                ["R0", "R1"] if name == "Q12" else step["after"]))
        steps.append(step)
    write_json(OUT / "queue.json", steps)


def retest_s3():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r12b_experiments", HERE / "experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.s3()
    except subprocess.CalledProcessError as error:
        return error.returncode
    shutil.copy2(HERE / "s3.json", OUT / "s3.json")
    return 0


def launch(delay, existing_pid):
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    write_json(OUT / "timer.json", dict(created_utc=now.isoformat(),
        not_before_utc=(now + dt.timedelta(seconds=delay)).isoformat(),
        existing_queue_pid=existing_pid, status="waiting"))
    print(f"timer armed; not before {(now + dt.timedelta(seconds=delay)).isoformat()}", flush=True)
    time.sleep(delay)  # Exactly one timer sleep.
    await_existing_queue(existing_pid)
    make_queue()
    write_json(OUT / "timer.json", dict(created_utc=now.isoformat(),
        not_before_utc=(now + dt.timedelta(seconds=delay)).isoformat(),
        existing_queue_pid=existing_pid, started_utc=stamp(), status="running"))
    with (OUT / "queue.log").open("a") as log:
        result = subprocess.run([PY, str(HERE / "queue.py"), "--queue",
                                 str(OUT / "queue.json"), "--out", str(OUT)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    progress = OUT / "progress.tsv"
    rows = [line.split("\t", 2) for line in progress.read_text().splitlines()[1:]] if progress.exists() else []
    all_done = len(rows) == len(json.loads((OUT / "queue.json").read_text())) and all(
        len(row) > 1 and row[1] == "done" for row in rows)
    state = "done" if result.returncode == 0 and all_done else "failed"
    write_json(OUT / "timer.json", dict(created_utc=now.isoformat(),
        not_before_utc=(now + dt.timedelta(seconds=delay)).isoformat(),
        existing_queue_pid=existing_pid, started_utc="see queue.log",
        finished_utc=stamp(), status=state, queue_exit_code=result.returncode))
    return 0 if state == "done" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    timer = sub.add_parser("launch")
    timer.add_argument("--delay-s", type=int, default=10800)
    timer.add_argument("--existing-pid", type=int, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("model", choices=("llama", "qwen3"))
    sub.add_parser("s3")
    args = parser.parse_args()
    if args.action == "launch":
        raise SystemExit(launch(args.delay_s, args.existing_pid))
    if args.action == "prepare":
        raise SystemExit(prepare(args.model))
    raise SystemExit(retest_s3())
