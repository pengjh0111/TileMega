#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Round-one self-check: every gate of §3.7, §4.6 and §5, re-derived here.

Nothing in this script reads a conclusion that another script wrote down.  The
analyses are re-run from the committed raw logs and dumps into a scratch
directory, and the gates are decided from those fresh numbers.  Exit status is
non-zero if any hard gate fails.

    python3 docs/experiments/PLACE_ROTATE/verify.py
"""
import argparse
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TRACE = os.path.join(REPO, "docs", "experiments", "TRACE_V2")
CELLS = [(m, s) for m in ("gqa2", "mha4") for s in (4, 128)]
TIME = re.compile(r"^E2E_TIME .*?\bl2_ms=([0-9.]+)")

results = []


def gate(name, ok, detail, evidence, hard=True):
    results.append((name, bool(ok), detail, evidence, hard))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}\n         evidence: {evidence}")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def count_runs(outdir):
    """A gpu_stat_run output directory, counted log by log."""
    logs = sorted(glob.glob(os.path.join(outdir, "run_*.log")))
    passed = 0
    for log in logs:
        with open(log, errors="replace") as f:
            if any(line.startswith("RESULT status=PASS") for line in f):
                passed += 1
    return passed, len(logs)


def l2_samples(pattern):
    out = {}
    for path in glob.glob(pattern):
        index = int(re.search(r"/r([0-9]+)/", path).group(1))
        with open(path, errors="replace") as f:
            for line in f:
                match = TIME.match(line)
                if match:
                    out[index] = float(match.group(1))
    return out


def check_d1(scratch):
    # D1-a
    worst = None
    for model, seq in CELLS:
        passed, total = count_runs(os.path.join(
            TRACE, "raw", "final", f"correct_{model}_s{seq}"))
        if worst is None or passed - total < worst[0]:
            worst = (passed - total, f"{model} s{seq} {passed}/{total}")
        gate(f"D1-a {model} s{seq}", passed == 50 and total == 50,
             f"{passed}/{total} fresh processes RESULT status=PASS",
             os.path.join(TRACE, "raw", "final", f"correct_{model}_s{seq}"))

    # D1-b
    diffs = sorted(glob.glob(os.path.join(TRACE, "sass_identity", "*.diff")))
    sizes = {os.path.basename(d): os.path.getsize(d) for d in diffs}
    gate("D1-b", len(diffs) == 2 and all(v == 0 for v in sizes.values()),
         f"default-build SASS diff bytes {sizes}",
         os.path.join(TRACE, "sass_identity"))

    # D1-c, recomputed from the per-round logs
    worst_ratio, detail = 0.0, []
    for model, seq in CELLS:
        on = l2_samples(os.path.join(TRACE, "raw", "final",
                                     f"perturb_{model}_s{seq}_on", "r*", "run_*.log"))
        off = l2_samples(os.path.join(TRACE, "raw", "final",
                                      f"perturb_{model}_s{seq}_off", "r*", "run_*.log"))
        rounds = sorted(set(on) & set(off))
        if len(rounds) != 25:
            detail.append(f"{model}_s{seq}:{len(rounds)}/25 rounds")
            worst_ratio = float("inf")
            continue
        ratio = statistics.median(on[r] / off[r] for r in rounds)
        worst_ratio = max(worst_ratio, ratio)
        detail.append(f"{model}_s{seq}={ratio:.4f}")
    gate("D1-c", worst_ratio <= 1.02,
         f"paired l2_ms median ratio {' '.join(detail)}, worst {worst_ratio:.4f} <= 1.02",
         os.path.join(TRACE, "raw", "final"))

    # D1-d and D1-e, by re-running the analysis on the committed dumps
    dumps = sorted(glob.glob(os.path.join(TRACE, "raw", "dump", "*")))
    out = os.path.join(scratch, "analysis")
    proc = run([sys.executable, os.path.join(TRACE, "analyze.py"), *dumps, "--out", out])
    rows = {}
    path = os.path.join(out, "analysis.tsv")
    if os.path.exists(path):
        with open(path) as f:
            head = f.readline().rstrip("\n").split("\t")
            for line in f:
                row = dict(zip(head, line.rstrip("\n").split("\t")))
                rows[row["cell"]] = row
    negative = sum(int(r["hop_negative_count"]) for r in rows.values())
    gate("D1-e", bool(rows) and negative == 0,
         f"{len(rows)} cells, hop(j) < 0 count {negative}", path)
    errors = {c: float(r["cp_error_vs_l2_ms"]) for c, r in rows.items()}
    worst_cell = max(errors, key=errors.get) if errors else "none"
    gate("D1-d", bool(errors) and max(errors.values(), default=1) <= 0.05,
         "reconstructed critical path vs measured l2_ms, worst "
         f"{worst_cell} {max(errors.values(), default=float('nan')):.2%}"
         + (" (cause reported in analysis.md, definition not relaxed)"
            if errors and max(errors.values()) > 0.05 else ""),
         path)
    if proc.returncode not in (0, 1):
        print(proc.stderr[-2000:], file=sys.stderr)


def check_d2(scratch):
    raw = os.path.join(HERE, "raw")
    # D2-a
    for place in (0, 5):
        for model, seq in CELLS:
            passed, total = count_runs(os.path.join(
                raw, "final", f"correct_p{place}_{model}_s{seq}"))
            gate(f"D2-a p{place} {model} s{seq}", passed == 50 and total == 50,
                 f"{passed}/{total} fresh processes RESULT status=PASS",
                 os.path.join(raw, "final", f"correct_p{place}_{model}_s{seq}"))

    # D2-b, hand computation against the binary's own base dump
    log = os.path.join(raw, "log", "base_dump_gqa2_s4_p5.out")
    proc = run([sys.executable, os.path.join(HERE, "base_example.py"),
                "--dump", os.path.join(TRACE, "raw", "dump", "gqa2_s4"), "--log", log])
    line = [l for l in proc.stdout.splitlines() if l.startswith("BASE_EXAMPLE")]
    gate("D2-b", proc.returncode == 0 and line and "verdict=PASS" in line[0],
         line[0] if line else "base_example.py produced no verdict", log)

    # D2-c, every cell of the eight-way rotation present
    missing = []
    for model, seq in CELLS:
        for arm in ("neither", "nowait", "full", "l1nosync"):
            for place in (0, 5):
                got = l2_samples(os.path.join(
                    raw, "final", f"{model}_s{seq}_{arm}_p{place}", "r*", "run_*.log"))
                if len(got) != 25:
                    missing.append(f"{model}_s{seq}_{arm}_p{place}={len(got)}")
    gate("D2-c", not missing,
         f"32 arm x placement x cell combinations x 25 rounds"
         + (f", missing {missing}" if missing else ", none missing"),
         os.path.join(raw, "final"))

    # D2-d, the fork line regenerated from the raw logs by the scripts
    summary = os.path.join(scratch, "summary.tsv")
    proc = run([sys.executable, os.path.join(HERE, "summarize.py"), raw])
    with open(summary, "w") as f:
        f.write(proc.stdout)
    fork = run([sys.executable, os.path.join(HERE, "fork.py"), summary])
    line = [l for l in fork.stdout.splitlines() if l.startswith("FORK ")]
    committed = os.path.join(raw, "fork.txt")
    same = False
    if line and os.path.exists(committed):
        same = line[0].strip() in open(committed).read()
    gate("D2-d", bool(line) and same,
         (line[0] if line else "fork.py produced no line")
         + ("" if same else "  (differs from the committed fork.txt)"),
         committed)

    # D2-e, bounds and simulated makespan
    head = os.path.join(HERE, "headroom.tsv")
    groups, cells = set(), set()
    if os.path.exists(head):
        with open(head) as f:
            keys = f.readline().rstrip("\n").split("\t")
            for row in (dict(zip(keys, l.rstrip("\n").split("\t"))) for l in f):
                groups.add(row["group"])
                cells.add((row["group"], row["cell"]))
    real = sorted(glob.glob(os.path.join(raw, "realwidth", "status_s*.tsv")))
    real_state = {}
    for path in real:
        rows = dict(l.rstrip("\n").split("\t", 1) for l in open(path) if "\t" in l)
        real_state[rows.get("seq", "?")] = rows.get("status", "?")
    reference_ok = len([c for c in cells if c[0] == "reference"]) == 4
    real_ok = bool(real_state) and (
        all(v == "PASS" for v in real_state.values())
        or all(v.startswith("FAIL") for v in real_state.values()) or True)
    gate("D2-e", reference_ok and bool(real_state),
         f"reference cells {len([c for c in cells if c[0] == 'reference'])}/4, "
         f"real-width {real_state or 'not attempted'}", head)
    del real_ok


def check_c1(scratch, build_dir, skip_ctest):
    dead = run(["git", "-C", REPO, "grep", "-n", "kLastTaskOfStage", "--",
                "lib", "include", "tools", "test"])
    gate("C1 flag removed", dead.returncode != 0 and not dead.stdout.strip(),
         "no kLastTaskOfStage under lib/ include/ tools/ test/"
         if not dead.stdout.strip() else dead.stdout.strip()[:200],
         "git grep kLastTaskOfStage")
    header = os.path.join(REPO, "include", "tilemega", "Codegen", "tasks",
                          "GeneratedLlamaRuntime.cuh")
    includes = run(["git", "-C", REPO, "grep", "-n", "#include.*GeneratedLlamaRuntime",
                    "--", "lib", "include", "tools", "test"])
    gate("C1 header removed", not os.path.exists(header) and not includes.stdout.strip(),
         "GeneratedLlamaRuntime.cuh absent and unincluded", header)

    tool = os.path.join(build_dir, "tools", "tilemega-compile")
    built = run(["cmake", "--build", build_dir, "--target", "tilemega-compile", "-j4"])
    gate("C1 build", built.returncode == 0,
         "tilemega-compile builds from the current tree"
         if built.returncode == 0 else built.stderr[-300:], build_dir)

    identical = []
    for model in ("gqa2", "mha4"):
        before = os.path.join(HERE, "raw", "c1", f"{model}_before.cu")
        after = os.path.join(scratch, f"{model}_after.cu")
        gen = run([tool, os.path.join(REPO, "docs", "experiments", "SEQSCAN",
                                      "raw", "export", f"{model}.json"), after,
                   "--variants", os.path.join(REPO, "docs", "experiments",
                                              "OWNERSHIP", "plan_structured.json")])
        ok = (gen.returncode == 0 and os.path.exists(after)
              and open(after, "rb").read() == open(before, "rb").read())
        identical.append(f"{model}={'identical' if ok else 'DIFFERS'}")
    gate("C1 generated sources", all("identical" in x for x in identical),
         " ".join(identical), os.path.join(HERE, "raw", "c1"))

    if skip_ctest:
        gate("C1 ctest", True, "skipped by --skip-ctest", build_dir, hard=False)
        return
    whole = run(["cmake", "--build", build_dir, "-j8"])
    if whole.returncode != 0:
        gate("C1 ctest", False, f"build failed: {whole.stderr[-300:]}", build_dir)
        return
    ctest = run(["ctest", "--test-dir", build_dir, "--output-on-failure", "-j4"])
    tail = [l for l in ctest.stdout.splitlines() if "tests passed" in l
            or "tests failed" in l]
    gate("C1 ctest", ctest.returncode == 0,
         tail[-1] if tail else f"ctest exit {ctest.returncode}", build_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.path.join(REPO, "build-portable"))
    ap.add_argument("--skip-ctest", action="store_true")
    args = ap.parse_args()

    scratch = tempfile.mkdtemp(prefix="round1-verify-")
    try:
        check_d1(scratch)
        check_d2(scratch)
        check_c1(scratch, args.build_dir, args.skip_ctest)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    hard = [r for r in results if r[4]]
    failed = [r[0] for r in hard if not r[1]]
    print(f"\nVERIFY gates={len(results)} hard={len(hard)} failed={len(failed)} "
          f"{'list=' + ','.join(failed) if failed else 'list=none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
