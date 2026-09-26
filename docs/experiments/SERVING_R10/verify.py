#!/usr/bin/env python3
"""Recheck the R10 structural contract against source and raw evidence."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/experiments/SERVING_R10"
BASE = "f6c270ce0"


def source(path: str) -> str:
    return (ROOT / path).read_text()


def hit(path: str, expression: str) -> str | None:
    for number, line in enumerate(source(path).splitlines(), 1):
        if re.search(expression, line):
            return f"{path}:{number}: {line.strip()}"
    return None


def show(number: int, conditions: list[tuple[bool, str]]) -> bool:
    passed = all(ok for ok, _ in conditions)
    print(f"K-{number} {'PASS' if passed else 'FAIL'}")
    for ok, detail in conditions:
        print(f"  {'PASS' if ok else 'FAIL'} {detail}")
    return passed


def required(path: str, expression: str) -> tuple[bool, str]:
    found = hit(path, expression)
    return bool(found), found or f"{path}: missing /{expression}/"


def forbidden(path: str, expression: str) -> tuple[bool, str]:
    found = hit(path, expression)
    return found is None, found or f"{path}: no /{expression}/"


def main() -> int:
    outcomes: list[bool] = []
    gemm = "include/tilemega/Backend/ServingGemm.h"
    legal = "include/tilemega/Solver/BackendCostQuery.h"
    attention = "include/tilemega/Backend/ServingAttentionMma.h"
    body = "include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h"
    runtime = "include/tilemega/Codegen/tasks/ServingRuntime.cuh"
    engine = "python/tilemega/serving/engine.py"
    search = "lib/Solver/SkeletonSearch.cpp"
    flow = "lib/Solver/FlowPreparation.cpp"

    outcomes.append(show(1, [required(gemm, r"SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>"),
        required(gemm, r"Swizzle<3,\s*3,\s*3>"),
        required(gemm, r"SM75_U32x4_LDSM_N"),
        forbidden(gemm, r"SM80_CP_ASYNC_CACHEALWAYS<std::uint32_t>|DefaultCopy")]))
    outcomes.append(show(2, [required(legal, r"ServingBF16ShapeLegal"),
        required(legal, r"m\s*%\s*16"), forbidden(legal, r"ServingBF16ShapeLegal[^\n]*m\s*%\s*32"),
        required(gemm, r"ServingBF16SmemBytes"),
        required(gemm, r"static_assert\(sizeof\(typename Mainloop::SharedStorage\)")]))

    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "--", "include", "lib", "tools", "python"],
        cwd=ROOT, text=True).splitlines()
    arch_violations, literal_violations = [], []
    for path in changed:
        if not (ROOT / path).is_file():
            continue
        for number, line in enumerate(source(path).splitlines(), 1):
            where = f"{path}:{number}: {line.strip()}"
            if path != "include/tilemega/Target/ArchDispatch.h" and re.search(
                    r"__CUDA_ARCH__\s*(?:==|!=|<=|>=|<|>)", line):
                arch_violations.append(where)
            if ("Serving" in path or "serving" in path or "SkeletonSearch" in path) and re.search(
                    r"\b(?:101376|102400)\b|\b(?:99|100)\s*\*\s*1024\b|\b(?:num_sms|grid)\s*=\s*128\b", line):
                literal_violations.append(where)
    outcomes.append(show(3, [(not arch_violations,
        arch_violations[0] if arch_violations else f"git diff {BASE}: {len(changed)} source files, no arch comparison outside ArchDispatch"),
        (not literal_violations, literal_violations[0] if literal_violations else "no serving SM/smem literal from the forbidden list")]))

    serving_tasks = list((ROOT / "include/tilemega/Codegen/tasks").glob("*Serving*TaskBody.h"))
    serving_tasks += [ROOT / body, ROOT / "include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h"]
    bad_math = []
    for file in serving_tasks:
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if re.search(r"\b(?:sinf|cosf|__sinf|__cosf|sincos|double)\b", line):
                bad_math.append(f"{file.relative_to(ROOT)}:{number}: {line.strip()}")
    outcomes.append(show(4, [required(attention, r"SM80_16x8x16_F32BF16BF16F32_TN"),
        required(attention, r"SM75_U32x4_LDSM_N"),required(attention, r"SM75_U16x8_LDSM_T"),
        required(body, r"cp\.async|CpAsync|cp_async"),
        required("include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h", r"exp2"),
        (not bad_math,bad_math[0] if bad_math else "serving TaskBody headers: no trig or double")]))

    launch = source(runtime).split('extern "C" int tm_plan_launch', 1)
    launch_body = launch[1].split('extern "C"', 1)[0] if len(launch) > 1 else ""
    decode = source(engine).split("for step in range(count - 1):", 1)
    decode_body = decode[1].split("final.record", 1)[0] if len(decode) > 1 else ""
    outcomes.append(show(5, [forbidden(runtime, r"ResetBuffersOnly"),
        forbidden(engine, r"ResetBuffersOnly"),
        required(runtime, r"next_iteration\[2\]"),
        required(runtime, r"next_iteration\[mode_index\]"),
        forbidden(runtime, r"selected_mode"),
        required("python/tilemega/serving/plan.py", r"self\.iteration = \{1: 0, 2: 0\}"),
        (bool(launch_body) and not re.search(r"cudaMemcpy|cudaDeviceSynchronize|cudaStreamSynchronize", launch_body),
         f"{runtime}: tm_plan_launch has no copy/sync" if launch_body else "tm_plan_launch missing"),
        (bool(decode_body) and not re.search(r"synchronize|\.item\(|\.cpu\(", decode_body),
         f"{engine}: decode loop has no sync/host read" if decode_body else "decode loop missing")]))

    outcomes.append(show(6, [required("include/tilemega/Codegen/tasks/ModelRuntime.h", r"int batch = 1"),
        required("include/tilemega/Codegen/tasks/ModelRuntime.h", r"per_batch"),
        required("lib/Frontend/Frontend.cpp", r'"batch", builder\.getStringAttr'),
        required("python/tilemega/serving/export.py", r"dtype=torch\.int32"),
        required("lib/Frontend/ModelPlan.cpp", r"id_bits = TokenIdBits\(ids_node\)"),
        forbidden("lib/Frontend/Frontend.cpp", r"batch\s*==\s*(?:1|16)")]))

    dumps = list(EVIDENCE.glob("**/*.mlir"))
    affine = [p for p in dumps if "kFusedAttention" in p.read_text(errors="replace") or
              "attention_kv_block" in p.read_text(errors="replace")]
    outcomes.append(show(7, [required("lib/Frontend/ServingSemanticLifting.cpp", r"const auto Ec = C\(stage\.attention_kv_block\)"),
        (bool(affine), f"{affine[0].relative_to(ROOT)}: serving CG dump present" if affine
         else "SERVING_R10: no raw serving CG dump for affine relation check")]))

    pruning = "include/tilemega/Solver/ServingPruning.h"
    outcomes.append(show(8, [*[required(pruning, rf"PruneServingR{i}") for i in range(1,4)],
        required(pruning, r"MakeServingSearchOrderR4"),
        required("include/tilemega/Solver/OperatorClasses.h", r"PruneServingR1\("),
        required("include/tilemega/Solver/OperatorClasses.h", r"PruneServingR2\("),
        required(search, r"MakeServingSearchOrderR4\("),
        required(search, r"PruneServingAttentionSmemR1\("),
        required(search, r"ATTENTION_COORDINATE"),
        required(search, r"incremental_prepare"),
        required(search, r"EvaluateFlow\(low\.flow->flow\)"),
        required(search, r"4\*point\.candidate\.score")]))

    outcomes.append(show(9, [required(flow, r"RuntimeReleaseEndpoint\("),
        required(flow, r"RuntimeDependencyBounds\("),
        required(flow, r"std::max\(last,bounds\.last\(\)\)"),
        required("include/tilemega/Codegen/tasks/ServingRuntime.cuh", r"RuntimeDependencyBounds\(")]))
    outcomes.append(show(10, [required("include/tilemega/Target/TargetSpec.h", r"inflight_curve_bytes"),
        required("include/tilemega/Target/TargetSpec.h", r"cta_stream_curve_bytes"),
        required("lib/Solver/StageFlowModel.cpp", r"InflightDramServer"),
        required("lib/Solver/FluidExecutionSimulator.cpp", r"InflightDramServer")]))
    outcomes.append(show(11, [required(search, r"0\.98\*ea\.evaluation\.makespan_ns")]))

    audits = list(EVIDENCE.glob("plans/**/*fp64*.json"))
    audit_counts = []
    for path in audits:
        data = json.loads(path.read_text())
        audit_counts.append((path, data.get("fp64_total")))
    outcomes.append(show(12, [
        forbidden("docs/experiments/SERVING_R10/seed_split_price/command.txt", r"MIDPOINT_REFINE=1"),
        (len(audit_counts) >= 20 and all(n == 0 for _, n in audit_counts),
         f"{len(audit_counts)}/20 solved-plan SASS audits, counts={audit_counts[:2]}")]))
    outcomes.append(show(13, [required("include/tilemega/Codegen/tasks/ModelRuntime.h", r"eft_past_lo"),
        required("include/tilemega/Codegen/tasks/ModelRuntime.h", r"eft_past_hi"),
        required(runtime, r"RuntimeDependencyBounds\(")]))

    vllm = "python/tilemega/serving/vllm_baseline.py"
    outcomes.append(show(14, [required(vllm, r"temperature=0\.0"),
        required(vllm, r"ignore_eos=True"),required(vllm, r"detokenize=False"),
        required(vllm, r"TokensPrompt"),forbidden(vllm, r"enforce_eager=True"),
        required(vllm, r"prompt_ids")]))
    weights = "python/tilemega/serving/weights.py"
    outcomes.append(show(15, [required("lib/Frontend/ModelPlan.cpp", r"qkv_group_interleave"),
        required("lib/Frontend/ModelPlan.cpp", r"gate_up_interleave"),
        required(weights, r"qkv_group_interleave"),
        required(weights, r"gate_up_interleave"),
        forbidden(weights, r"if\s+.*(?:llama|qwen)")]))
    token_log = EVIDENCE / "token_sets.log"
    outcomes.append(show(16, [required("include/tilemega/Codegen/tasks/ServingEmbeddingTaskBody.h", r"past"),
        required("include/tilemega/Codegen/tasks/ServingArgmaxReduceTaskBody.h", r"output_position"),
        required("include/tilemega/Codegen/tasks/ModelHarness.cuh", r"dims\.past \+ dims\.seq|dims\.past \+ p\.dims\.seq"),
        (token_log.is_file() and "TOKENS_DISJOINT" in token_log.read_text(),
         f"{token_log.relative_to(ROOT)}: " +
         (token_log.read_text().strip() if token_log.is_file() else "missing raw log"))]))

    passed = sum(outcomes)
    print(f"G-1 {'PASS' if passed == 16 else 'FAIL'}: {passed}/16 structural checks")
    return 0 if passed == 16 else 1


if __name__ == "__main__":
    raise SystemExit(main())
