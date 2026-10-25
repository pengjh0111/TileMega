# MX-5 C550 pipeline and specification audit

Verified by source inspection, 2026-10-25, branch `tilemega-metax`, HEAD
`37ac67c148eb90d6a5e9a9c879abfb36945bef8e`; existing dirty headers/results preserved.
Root subsequently committed the engine repair as `120abd680`; updates through
`ba573643a` include protocol-script fixes, the evaluated-grid invariant, five manual development
checks and the CPU v2 solver.
Authority: `/root/Prompt/TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
Read AGENTS.md/CLAUDE.md and the complete prompt. This audit is CPU source
inspection; guarded component measurements are documented in
`MX5_integration_c550.md`. The earlier pipeline audit is preserved in git history.
Measurements below are stated by existing reports, not reverified by this audit.

## Actual pipeline and implementation status

| Path | Verified current behavior and exact source |
|---|---|
| Export | `scripts/mx_serving_cell.sh:25` exports decode/prefill, makes bridge JSON and registers MoE export pytree at line36. Checkpoint packing happens when serving loads plans. |
| Compile route | `tools/commands/compile.cpp:571` routes decoder/decode JSON to `mx_decode::TryCompile`; target table path is resolved at line581. |
| Default dense | `tools/commands/mx_decode.cpp:96` selects v2 only when pin contains `mx_decode_v2`; otherwise BuildModelPlan/ImportPlan/AnalyzeMxDecode still feed MX-4. `DecodeProgram.cuh:190` still executes Embed; `DecodeRuntime.cuh:52` counts 5L+2 stages. |
| Explicit v2 | `tools/commands/mx_decode_v2.cpp:71` accepts only `manual` or `resource_probe` purpose; line80 rejects interpreter flags; line89 rejects DM plans. Manifest at line255 records solver_selected=false/default_enabled=false. This is a diagnostic integration path, not a final solver selection. |
| Static emitter | `lib/Codegen/MxDecodeProgramV2.cpp:198` constructs StageProgram; lines219–235 emit QKV(kStore), attention, O, gate_up, down/DownLast and head. `StageProgram.cpp:69` emits type aliases, line83 the unrolled-disabled layer loop and line87 compile-time body calls. No runtime stage descriptor was introduced. |
| Embedding/edges | `DecodeProgramV2.cuh:76` selects embedding A for layer0; line83 selects its O residual; emitter line230 sends DownLast to head and line235 sends head to layer0 QKV weights. Embed stage is absent from v2. |
| V2 kernel/runtime | `DecodeProgramV2.cuh:165` receives Binding and launch controls by value; line181 synchronizes before entrance prefetch, line209 uses stage-local 64-byte counters, line215 broadcasts next-step tokens after head. `DecodeRuntimeV2.cuh:45` counts 5L+1 stages; line123 reports L1 only, line241 requires one resident CTA, line251 validates all pasts, line288 provides launch_steps. |
| Synchronization | `StageProgram.cuh:28` implements P-B grid release/add, polling, acquire/broadcast and LDS generation release; watchdog code is conditional at line34. `LastArriverV2.cuh:15` implements CTA publish/ticket/last-only acquire/reduction/reset. These are current model clients, not only unit/bench code. |
| V2 prefetch | `MxDecodeProgramV2.cpp:83` derives P_w from measured r_next when prefetch=-1; line107 bounds explicit prefetch against provided r_next. Wait-limit validation begins line118. Manual pins can omit r_next; this is not qualified production selection. |
| V2 resources | `mx_decode_v2.cpp:166` parses actual resource report and rejects every violation; allow_resources cannot bypass it. `python/tilemega/mx5_resources.py:117` enforces MTreg227/255, frame0 and static-smem0. Per-body fields remain pending by default at line130. mx5_attribution.py now proves one old-v7 actual-program partition via executable equality, with residuals; each other plan still requires its own proof. |
| Prefill fragment opt-in | `compile.cpp:1851` consumes pin marker mx_kv_fragments_v2 only for dense prefill; line1854 requires seq64; line2040 records kv_storage=block_fragments. `PrefillKvFragmentsV2.h:14` fixes Tokens64 and line21 traps nonzero past/misaligned capacity/block extent. Lines45/52 publish complete K/V fragments using the old prefill shared arrays. |
| Weights | `MxDecodeProgramV2.cpp:149` wraps GEMM recipes in tile_pages(xcore_mma,TN,32) and names them by TN. `serving/weights.py:109` rejects conflicting same-name recipes; line162 performs physical page packing, line234 loads/deduplicates checkpoint-derived weights by source hashes and complete recipe. Different phase layouts can coexist under distinct names. |
| Request storage | `serving/state.py:66` allocates one shared KV backing store; line75 binds each layer, line78 validates dtype/elements/contiguity. Equal byte counts do not identify the physical KV order. |
| Engine | `serving/engine.py:40` checks sequence/capacity, line51 checks kv_layout; new line55 checks kv_storage before binding. Line73 onward loads weights/state, line88 onward sets decode pasts, line128 onward performs one prefill then separate/device-loop decode. `execution.py:45` already resolves mx_static to L1. |
| Measurement/check | `scripts/mx_check_modes.sh:10` runs C-2 then teacher-forcing HF C-1; the measure path uses existing ServingEngine. All five512 manual development controls pass512-step C-1/C-2; solver-selected TPOT acceptance remains open. The checker disables optional step events because v2 step_ns is diagnostic-only. |
| Bench/calibration | GEMM/head/attention benches share v2 bodies. `experiments/mx5/calibration.py:335` handles prescribed resource-reduction selection; `stage_table.py:271` aggregates three rounds, requires matching source/binary/resource identity and P0 reference, and labels output measured_rows_only/full_table=false at line304. |
| Production solver/table | `configs/targets/xcore1000.json:2360` still selects xcore1000_mx_decode.json. `MxDecodeSolver.cpp:130` consumes the old schema; line229 chooses two candidates by changing only the smallest-gap class. `mx_decode.cpp:283` times those two. MxDecodeSolverV2.cpp implements exact k-best4/evidence checks. Explicit --mx-decode-v2-solve plus mx5_select.py orchestration is now wired and CPU-tested (`c532da9b5`); automatic per-plan reductions/D7 fallback, real top4 E2E and final promotion remain pending. |
| Cell scripts | `mx_serving_cell.sh:65` still excludes experts from static decode and expects an mx4_pin. Protocol fix `ba573643a` makes candidate timers900s, prefill watchdog `${WATCHDOG:-0}`, check prefer prefill_w/decode_w, timed decode disable step events and expert jobs reserve40960MiB. Explicit WATCHDOG=1 prefill solves write prefill_w (`23a53044b`). Final script top4 routing/MoE prompt construction remain pending. |
| MoE | `mx_decode.cpp:104` still returns -1 for DM and `compile.cpp:826` sets moe_batch only on the legacy path. Lines838–850 build/specialize DM plans. No AnalyzeMxMoe/MxGroupedGemm/native M1–M7, --mx-prompt route or native prompt plan exists. |
| DNN | `compile.cpp:738` builds DnnModelPlan then continues toward legacy DM lowering, with no mx_dnn native branch. `dnn/cli.py:58` defaults forward_executor=L2, line229 uses build-dm; no time command/native bodies/dw_rsc/c550 config suite exists. `dnn/check.py:30` already provides reusable NativeForward ABI/identity/external-binding validation. |

## Differences, defects and evidence boundaries

1. **Found and repaired physical-layout safety gap.** Prompt §6.7 assumes the
   existing engine's kv_layout check is sufficient; DT-KV block fragments make
   two plans both say vt while addressing incompatible bytes. Artifact identity
   checks each plan individually (`build/feature_contract.py:61`, `identity.py:101`)
   and cannot enforce compatibility between plans. The original engine/state
   accepted this pair. With root's explicit follow-up authorization, added a
   four-line kv_storage equality guard at `serving/engine.py:55`, defaulting old
   manifests to linear. No executor/default/layout changed. Focused CPU tests
   mock CUDA/PlanLibrary and verify rejection before execution/binding and
   acceptance of matching old/fragment metadata. Combined with feature-contract
   and unified-identity suites: **27 passed**, GPU visibility disabled.
2. **Default source still has MX-4 architecture.** This is visible incompletion,
   not evidence that v2 secretly falls back when rejected. Explicit v2 resource
   failures throw; the default must not be represented as final MX-5 acceptance.
   Old dense decode remains static (not an interpreter); MoE/DNN legacy paths
   remain callable, conflicting with final §0.2/§6.1/§8.6 compile-time rejection.
3. **StageProgram+dense integration now exists.** Historical
   `docs/results/MX5_pipeline_c550.md` predates these files and its “not started”
   statements are stale. `MX5_integration_c550.md` is the relevant checkpoint.
4. **512-only success is not a DT-W override.** Source supports manual512/576,
   but qualifying a real512 plan still requires prescribed576 prefetch reductions,
   D7 diagnosis and manifest/report evidence. No automated plan-level reduction
   and justified576→512 selection workflow is present in TryCompileV2.
5. **Schedule overlap is recorded.** Integration checkpoint states B/C overlap
   to expose whole-program resource defects. This departs from strict §9 order
   but preserves open gates; no reason found to waive B, model checks or timing.
6. **Pending resource attribution is explicit.** Total executable bytes and
   kernel instructions are collected; per-body code sizes/instructions and I1
   program-order differences are not complete in model manifests (§4.12).
7. **Component/campaign counts are not end-to-end results.** The integration
   checkpoint states G-A passed, G-B/G-PROGRAM open, prior4545-key campaign
   coverage, later invariant repairs and new resource regressions. This audit
   reverified neither timing nor numerical counts; current-source campaigns
   must replace stale-source qualification without rewriting historical rows.

## Critical next gates

1. Finish root's current invariant/resource regression: all legal v2 bodies
   frame0, correct arithmetic, fresh five-process synchronization evidence for
   changed paths, steady-ring MIR without4160/generic/private/call defects.
2. Rebuild/requery P0 R_next and derived-prefetch calibration for the **same
   final source**, then three-process timings/full domain. Do not promote
   partial tables, prune failures, infer R_next or replace576 measurements
   with512. Diagnose whole-program576 using D7 and the prescribed reductions.
3. Build five real dense development cells with compatible fragment-prefill
   watchdog0 timing and watchdog1 correctness twins; require C-1/C-2 and
   uninstrumented end-to-end comparisons/ledger before default selection.
4. Wire the CPU-tested v2 solver/top4 into compilation and two fresh processes
   per candidate, selection threshold and pred/meas attribution; update table,
   pins and script's prefill_w preference/900s timing guards.
5. Complete native MoE decode/prompt/routing/packing and40GiB guards; then all
   25 dense+5 MoE checks/timings and G-LLM. Complete native DNN and25 synthetic
   checks/timings afterward; only DT-F may be cut for budget.

Later measured evidence (verified) is recorded in MX5_integration_c550.md:
the old-grid Qwen B1 manual control regresses19.83% against MX-4. Its attention
skip contribution23.39us versus22.27us bench triggers D7; the merger's serial
per-partial Exp2 differs from the prompt's parallel design. A copied-header
MIR investigation is underway. This is not a solver-selected final cell or a
basis for accepting the performance predictions. The engine storage assumption
is the concrete cross-phase incompatibility already corrected here.
