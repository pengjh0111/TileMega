#!/usr/bin/env python3
# usage: check_lag_indices.py plan.so.top1.cu
import re, sys
src = open(sys.argv[1]).read()
def block(name):
    m = re.search(r"constexpr \w+ " + re.escape(name) + r"\[\] = \{(.*?)\n\};", src, re.S)
    return m.group(1) if m else sys.exit(f"table {name} not found")
spec = [(k, int(g)) for k, g in re.findall(r"\{TaskKind::(\w+),\s*(\d+)u", block("kStages"))]
gemms = [int(s) for s in re.findall(r"\{\s*\d+u,\s*(\d+)u,\s*\d+u,\s*\d+u,\s*\d+u,\s*\d+u\}", block("kRuntimeGemms0"))]
lags = re.findall(r"\{(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*LagDependency::Kind::(\w+)\}", block("kLagDependencies"))
runtime, entry = [], []
for kind, g in spec:
    entry.append(len(runtime)); runtime.append(kind)
    if kind == "kGemm" and g < len(gemms) and gemms[g] > 1: runtime.append("kGemmCombine")
print(f"spec {len(spec)} stages, runtime {len(runtime)} stages")
for l2, l1, c, kind in lags:
    l2, c = int(l2), int(c)
    bad = entry[c] != c or entry[l2] != l2
    print(f"{kind:13s} consumer {c}: runtime {runtime[c]} | producer_l2 {l2}: runtime {runtime[l2]}"
          f" | correct {entry[c]}/{entry[l2]}" + ("  <-- reads another stage" if bad else ""))
