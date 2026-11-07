# SM90X1 execution protocol

[verified] Branch `tilemega`, worktree `/root/TileMega-x1`, baseline `3ff07bc2e`.
The original `tilemega` ref was fast-forwarded from the R4 closeout to the existing
R5 final commit before adding tooling. No experiment edit touches kernel bodies,
cost models, target constants, search rules or `SearchSm90Dense`.

[verified] `configs/serving/sm90x1/cells.json` freezes all 14 required cells,
reference coordinates, canonical coordinate hashes, accepted TPOT, prefills and
classes. Both MoE references were bound through the R5 final table's e123 arm,
its spec's binary SHA, its build binding's coordinate SHA, and equality with the
binary plan's native coordinates. All prefill hashes were checked. No optional
cell is included in the required cell list.

[verified] Setup uses three fresh P-FINAL processes for each pilot after five
P-CORR processes, with c1=256, c2=true, reference_processes=1. Native
`--hf-positions 256` produces the required 256-token teacher-forced prefix.
Main arms use three processes, or five for Llama 8B, Mistral and Qwen MoE.
Dense units use `trial`; MoE units use `machine`. Every GPU process is launched
by the existing scheduler and physical-card guard. GPU6 remains excluded by
the timing policy. Only GPU0–3 are visible in this container. Guard acceptance
may be provisional because host PIDs are outside the container namespace; this
is retained in raw guard receipts rather than described as verified ownership.

[verified] CLI enumeration can import once for every family:

```sh
runs/sm90x1/compiler/tools/tilemega inspect native-sweep \
  --export runs/sm90x1/cells/qwen3_0_6b-B1/decode.export.json \
  --target configs/targets/sm_90_h800.json --batch 1 \
  --reference runs/sm90x1/cells/qwen3_0_6b-B1/W.json \
  --family all --top 4 --out runs/sm90x1/example
```

[verified] Individual families, `--mode check`, `--mode compose --selection FILE`,
and optional `--mode T` / `--mode N` are supported. Composition searches the
measured per-family rankings in increasing total fallback rank, then family
rank order, and records every selected fallback index. An unavailable control
fails explicitly; the harness preserves the failure. W keeps its exact group
entry; unselected and illegal groups remain in the candidate manifest.

[inferred] For fused-expert cost ties, fused shared bytes then the two TileK
values extend the solver's individual-family ranking. Each pair first takes
its deepest common legal slot count. This extension is a tooling choice.

[verified] Interactive phase execution:

```sh
python3 scripts/sm90x1.py prepare --cell all
python3 scripts/sm90x1.py build
python3 scripts/sm90x1.py corr
python3 scripts/sm90x1.py time
python3 scripts/sm90x1.py trace --cell qwen3_0_6b-B1
python3 scripts/sm90x1.py tables
# Continue only after Phase 0 passes.
python3 scripts/sm90x1.py prepare --phase 1
python3 scripts/sm90x1.py build --phase 1
python3 scripts/sm90x1.py trace --phase 1
python3 scripts/sm90x1.py analyze --phase 1
python3 scripts/sm90x1.py arms --phase 1
python3 scripts/sm90x1.py build --phase 1 --arms-only
python3 scripts/sm90x1.py corr --phase 1
python3 scripts/sm90x1.py time --phase 1
python3 scripts/sm90x1.py tables --phase 1 --cell all
```

[stated] Phase 2 repeats those commands with `--phase 2` after the pilot cost
gate; it selects the remaining 12 cells. Optional controls use `arms --phase 3`,
then build/corr/time/tables after all 14 main cells finish. They are not launched
by the required campaign. `--k 3` requires the recorded failed k=4 projection.

[verified] `scripts/sm90x1_campaign.py` executes the required phases in order,
waits for existing Phase 0 parent processes when passed `--wait-pids`, stops at
the 2% or cost gates, and preserves k=4 pilot directories before a k=3 retry.
Its state and per-command logs live in `runs/sm90x1/`. Core/runtime changes after
a Phase 0 build require rebuilding it. Main builds freeze one measurement HEAD;
keep it until Phase 2 finishes. Compiler executables are copied into immutable
SHA-named directories before builds.

[verified] CPU checks cover dense and fused MoE legality, exact W retention,
per-group GEMM ranking, deepest fused slots, deterministic enumeration, N
legality, latest-arriver own-time attribution, release telescoping, process-range
ties, rank reversals, aggregate ratios, the 82-pair denominator and retention of
all missing cells. The two end-to-end CLI enumerations emitted byte-identical
coordinates for all six pilot families. Main GPU sweep/arm orchestration remains
unvalidated until the Phase 0 gate opens.

[verified] Empty/pending CSV fields are unavailable measurements. They are not
zeros. Diagnostic tau is tau-b with exact ties, pooled by concatenating variant
observations; selections/reversals also apply the conservative noise threshold.
Paper aggregates remain unavailable while any required cell is missing.
