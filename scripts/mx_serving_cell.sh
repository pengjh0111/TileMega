#!/usr/bin/env bash
# TileMega cell pipeline on the C550 (MX-E).
#   tm_cell.sh export  <id> <family> <cfg_dir>
#   tm_cell.sh solve   <id> <batch> <decode|prefill>          (CPU; device query via guard on $QGPU)
#   tm_cell.sh check   <id> <batch> <rand_dir> <prompts> <gpu> (C-2 and C-1)
#   tm_cell.sh measure <id> <batch> <rand_dir> <prompts> <gpu> <round>
# MX5_RESOLVED_V2 explicitly adds guarded v2 checks and top4 timing to decode solve.
# Plans: runs/mx2/tm/<id>/B<b>/<phase>/plan.so. Raw data stays under runs/mx1.
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd $ROOT
# The committed target, not the gitignored copy under runs/: MX-1 and MX-2 both
# solved against runs/mx1/calib/xcore1000_c550.json, so a clone could not
# reproduce a cell (MX3-0 item 1). The two differ only by calibration_sections,
# which is provenance stamps the solver never reads.
TARGET=${TARGET:-configs/targets/xcore1000.json}
# Campaign output root, so MX-2 and MX-3 plans do not collide (MX3-0 item 3).
CELLS=${CELLS:-runs/mx2/tm}
QGPU=${QGPU:-6}
JOBS=${JOBS:-8}
# Solves compile against a frozen worktree (/root/tilemega_solve, its own
# build): an artifact's identity snapshot covers every source file, so header
# edits in the development tree during a 30-minute solve would invalidate it.
SOLVE_BIN=${SOLVE_BIN:-$ROOT/build/tools/tilemega}
cmd=$1; id=$2
needs_mib=20480
query_needs_mib=2048
if [ -f runs/mx1/export/${id}_decode/bridge.json ] && \
   grep -q -m1 expert runs/mx1/export/${id}_decode/bridge.json; then
  needs_mib=40960
  query_needs_mib=40960
fi
case $cmd in
export)
  family=$3; cfg=$4
  for phase in decode prefill; do
    D=runs/mx1/export/${id}_$phase
    [ -f $D/bridge.json ] && continue
    mkdir -p $D
    [ -f $D/exported_program.pt2 ] || env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= PYTHONPATH=python \
      python -m tilemega.serving.export --config $cfg/config.json --model $family --phase $phase \
      --capacity 1088 --prompt-len 64 --out $D > $D/export.log 2>&1
    env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= PYTHONPATH=python \
      python -m tilemega.export_bridge $D/exported_program.pt2 --out $D/bridge.json \
      $( [ $family = qwen3_moe ] && echo --pytree-module tilemega.moe.export ) >> $D/export.log 2>&1
    rm -f $D/exported_program.pt2
  done ;;
solve)
  b=$3; phase=$4
  phase_dir=$phase
  [ "$phase" = prefill ] && [ "${WATCHDOG:-0}" = 1 ] && phase_dir=prefill_w
  C=$CELLS/$id/B$b/$phase_dir; mkdir -p $C
  # Explicit v2 only: audit READY is revalidated by a purpose-limited runner.
  if [ "$phase" = decode ] && [ -n "${MX5_RESOLVED_V2:-}" ]; then
    : "${MX5_DOMAIN_TOOL:?MX5_DOMAIN_TOOL must name the trusted CPU geometry tool}"
    [ -z "${PIN:-}" ] && [ "${SELECTION:-measure}" = measure ] || { echo "resolved v2 requires measured selection without a manual PIN" >&2; exit 2; }
    exec 9>"$C/.lock"
    flock -n 9 || { echo "cell already being solved" >&2; exit 3; }
    cell="$CELLS/$id/B$b"; output="$C/resolved_v2"
    if [ -f "$output/results.json" ] && [ -f "$C/plan.so" ]; then
      PYTHONPATH="$ROOT/python" python -m tilemega.mx5_resolved_runner \
        --verify-results "$output/results.json" --protocol "$MX5_RESOLVED_V2" \
        --domain-tool "$MX5_DOMAIN_TOOL" --cell-dir "$cell"
      exit $?
    fi
    [ ! -e "$C/plan.so" ] && [ ! -L "$C/plan.so" ] && [ ! -e "$cell/decode_w/plan.so" ] && [ ! -L "$cell/decode_w/plan.so" ] || {
      echo "resolved v2 refuses to replace an existing cell artifact; use a fresh CELLS campaign" >&2; exit 2; }
    prompts=${PROMPTS:-$ROOT/configs/prompts/$(case $id in qwen*) echo qwen;; *) echo llama3;; esac)_isl64.json}
    PYTHONPATH="$ROOT/python" python -m tilemega.mx5_resolved_runner \
      --protocol "$MX5_RESOLVED_V2" --domain-tool "$MX5_DOMAIN_TOOL" --root "$ROOT" \
      --model "${MODEL:-/root/models/${id}_rand}" --batch "$b" --gpu "${MGPU:-7}" \
      --prefill-so "$cell/prefill/plan.so" --prefill-watchdog-so "$cell/prefill_w/plan.so" \
      --prompt-ids "$prompts" --policy "$ROOT/configs/bench/measurement_policy_c550.json" \
      --needs-mib "$needs_mib" --out "$output" --cell-dir "$cell" \
      > "$C/compile.log" 2>&1 || { echo "rc=$?" >> "$C/compile.log"; exit 1; }
    echo rc=0 >> "$C/compile.log"
    exit 0
  fi
  [ -f $C/plan.so ] && grep -q "^rc=0" $C/compile.log 2>/dev/null && exit 0
  # One compile per cell. Every artifact in $C has a fixed name, so two
  # concurrent solves of the same cell interleave their plan.so, identity and
  # search output -- including the rc=0 line that makes the cell look done.
  # Another session sharing this checkout is enough to cause that, so the lock
  # is held for the whole solve and a second attempt leaves the cell alone.
  exec 9>$C/.lock
  flock -n 9 || { echo "cell already being solved by pid $(cat $C/.lock.pid 2>/dev/null)" >&2; exit 3; }
  echo $$ > $C/.lock.pid
  rm -f $C/plan.so
  # MX-4: a dense decoder's decode plan is the static stage program
  # (tools/commands/mx_decode.cpp, MX4 prompt §3.9/§4.4); MoE keeps the
  # interpreted executor below. Its solver's two plans are timed end to end
  # with the cell's prefill plan (solve prefill first), 2 fresh processes
  # each, alternating, on MGPU. The timed plan has no watchdog; decode_w is
  # the same pin with the watchdog, for check (C-1/C-2).
  if [ $phase = decode ] && ! grep -q -m1 expert runs/mx1/export/${id}_decode/bridge.json; then
    P=$ROOT/$CELLS/$id/B$b/prefill/plan.so
    [ -f $P ] || { echo "MX-4 decode is measured with $P; solve prefill first" >&2; exit 1; }
    prompts=${PROMPTS:-$ROOT/configs/prompts/$(case $id in qwen*) echo qwen;; *) echo llama3;; esac)_isl64.json}
    selection=${SELECTION:-measure}
    measure=()
    [ $selection = measure ] && measure=(--measure-cmd "env NEEDS_MIB=$needs_mib $ROOT/scripts/mx_gpu_job.sh ${MGPU:-7} 900 $C/measure_guard env -u ALL_PROXY -u all_proxy python -m tilemega.serving.measure --model /root/models/${id}_rand --prefill-so $P --prompt-ids $prompts --batch $b --mode L1 --decode-loop auto --max-new-tokens 1024 --warmup 1 --repeats 2 --step-events 0 --policy $ROOT/configs/bench/measurement_policy_c550.json")
    timeout ${CELL_TIMEOUT:-18000} nice $SOLVE_BIN compile runs/mx1/export/${id}_decode/bridge.json $C/plan.so \
      --serving decode --emit serving --batch $b --prompt-len 64 --runtime-target $TARGET \
      --selection $selection "${measure[@]}" --mx-tmin ${MX_TMIN:-128} --watchdog 0 ${PIN:+--evaluate-configs $PIN} \
      > $C/compile.log 2>&1 || { echo rc=$? >> $C/compile.log; exit 1; }
    W=$CELLS/$id/B$b/decode_w; mkdir -p $W
    $SOLVE_BIN compile runs/mx1/export/${id}_decode/bridge.json $W/plan.so --serving decode --emit serving \
      --batch $b --prompt-len 64 --runtime-target $TARGET --evaluate-configs $C/plan.so.mx4_pin.json --watchdog 1 \
      > $W/compile.log 2>&1 || { echo rc=$? >> $C/compile.log; exit 1; }
    echo rc=0 >> $C/compile.log
    exit 0
  fi
  trap 'rm -f $C/plan.so.m*.tsv $C/plan.so.verified*.tsv $C/plan.so.attn_*.m*.tsv $C/plan.so.attn_*.verified*.tsv' EXIT INT TERM
  # Row-major weights: decode with M<=4 dispatches to ServingGemvTaskBody,
  # whose xcore1000 specialisation rejects the tiled layout outright
  # (static_assert(!Tiled)), and the MX-C/MX-T cells that validated this port
  # all ran TILEMEGA_NONPAGED_TILED=0. A cell's two plans also share one
  # process, so both phases must agree on the layout.
  # Last-arriver handoff needs TILEMEGA_SERVING_SEQ==1 on a non-DM build, so a
  # prefill plan carrying one is refused by tm_plan_create at load time
  # (ServingRuntime.cuh). It is a decode-only option.
  # Last-arriver off for both phases, which is what the MX-C/MX-T cells that
  # validated this port ran (TILEMEGA_NONPAGED_LA=0). On prefill a plan
  # carrying it is refused at load (SERVING_SEQ!=1, ServingRuntime.cuh), and
  # on decode the candidate it selects is not buildable: qwen3_0_6b B1/B2
  # decode logged NONPAGED_LAST_ARRIVER selected=112 and then died in cucc on
  # static_assert(ServingBF16ShapeLegal) for ServingGemmConfig<Xcore1000,
  # 16,32,16,3> -- tile_k 16 with 3 stages, where the serving GEMM body allows
  # k in {64,128}. PruneServingR1 checks that predicate, so the last-arriver
  # path reaches codegen without it.
  # MX3-4 item 3: LA=1 re-tests the last-arriver handoff now that codegen
  # emits the selected tile_k and stages (fixed in MX-1, never re-measured).
  la=${LA:-0}
  # Fragment-packed ("tiled") weights measure faster than row-major on this
  # device, so they are the default; LAYOUT=row overrides for an A/B.
  layout=${LAYOUT:-tiled}
  # The solver picks a GEMV variant for decode at M<=4 and prices it as the
  # better choice, but measured it is not: forcing MMA on every GEMM class
  # took qwen3_0_6b B1 from 8.026 to 5.823 ms/token and B2 from 8.281 to
  # 5.991 (-27%). GEMV also cannot take fragment-packed weights
  # (ServingGemvTaskBody asserts !Tiled), so selecting it costs the tiled
  # layout as well. Keep it out of the domain until the cost model prices it
  # correctly; GEMM_IMPL=gemv restores the old behaviour for an A/B.
  # Decode attention: `auto` searches every body the target can hold (simt,
  # and mma16 -- PrefillAttentionMma at Tokens=1 -- where its shared memory
  # fits) and lets the selection choose per cell (MX3-3, G-ATTN); ATTN=simt
  # or ATTN=mma16 fixes one for an A/B.
  extra=""; [ $phase = decode ] && extra="--attention-impl ${ATTN:-auto}"
  if [ "${GEMM_IMPL:-mma16}" = mma16 ]; then
    for class in qkv o gate-up down lm-head; do
      extra="$extra --gemm-impl-$class mma16"
    done
  fi
  # MX3-5 / MX4 prompt §5.7: decode (MoE) and prefill are chosen by
  # measurement -- at most 4 candidates, one pilot round, the top 2 timed
  # twice -- with the L1 executor, on the measurement card MGPU, through the
  # guard. SELECTION=predicted restores the MX-2 behaviour.
  selection=${SELECTION:-measure}
  measure=()
  if [ $selection = measure ]; then
    model=/root/models/${id}_rand
    measure=(--measure-top 4 --measure-finalists 2 --measure-rounds 2
             --measure-cmd "env NEEDS_MIB=$needs_mib $ROOT/scripts/mx_gpu_job.sh ${MGPU:-7} 900 $C/measure_guard env -u ALL_PROXY -u all_proxy python -m tilemega.serving.measure_candidate --model $model")
  fi
  env CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= \
    TILEMEGA_DEVICE_RUN_PREFIX="env NEEDS_MIB=$query_needs_mib $ROOT/scripts/mx_gpu_job.sh $QGPU 300 $C/devq" \
    timeout ${CELL_TIMEOUT:-18000} nice $SOLVE_BIN compile runs/mx1/export/${id}_$phase/bridge.json $C/plan.so \
      --serving $phase --emit serving --batch $b --prompt-len 64 --pg off $extra \
      --solve $TARGET --runtime-target $TARGET --selection $selection "${measure[@]}" --watchdog ${WATCHDOG:-0} \
      --nonpaged-weight-layout $layout --nonpaged-la $la --search-jobs $JOBS \
      --search-passes 2 --search-budget-ms ${SEARCH_BUDGET_MS:-1200000} \
      ${PIN:+--evaluate-configs $PIN} \
      > $C/compile.log 2>&1 && echo rc=0 >> $C/compile.log || {
        rc=$?
        # The rejection surfaces while probing resources for a candidate that
        # selected a GEMV variant, so it lands in a per-candidate build log
        # under plan.so.resources, not in ptxas.log.
        if [ $layout = tiled ] && grep -rqs "tiled GEMV weights are not supported" \
             $C/plan.so.resources $C/plan.so.ptxas.log 2>/dev/null; then
          echo "layout=tiled rejected by a selected GEMV variant; retrying row" >> $C/compile.log
          layout=row
          env CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= \
            TILEMEGA_DEVICE_RUN_PREFIX="env NEEDS_MIB=$query_needs_mib $ROOT/scripts/mx_gpu_job.sh $QGPU 300 $C/devq" \
            timeout ${CELL_TIMEOUT:-18000} nice $SOLVE_BIN compile \
              runs/mx1/export/${id}_$phase/bridge.json $C/plan.so \
              --serving $phase --emit serving --batch $b --prompt-len 64 --pg off $extra \
              --solve $TARGET --runtime-target $TARGET --selection predicted --watchdog ${WATCHDOG:-0} \
              --nonpaged-weight-layout row --nonpaged-la $la --search-jobs $JOBS \
              --search-passes 2 --search-budget-ms ${SEARCH_BUDGET_MS:-1200000} \
              >> $C/compile.log 2>&1 && echo rc=0 >> $C/compile.log || {
                echo rc=$? >> $C/compile.log; exit 1; }
        else
          echo rc=$rc >> $C/compile.log; exit 1
        fi
      }
  echo "layout=$layout" >> $C/compile.log
  # Search artifacts are large and not needed after selection. The trap runs
  # on failure and on kill too: one 48-cell sweep of leftovers filled the disk
  # (357 GB). dump_janitor.sh also prunes them while a solve is running.
  # A second attention body searches under plan.so.attn_<body> (MX3-3).
  rm -f $C/plan.so.m*.tsv $C/plan.so.verified*.tsv $C/plan.so.attn_*.m*.tsv $C/plan.so.attn_*.verified*.tsv ;;
check)
  b=$3; rand=$4; prompts=$5; gpu=$6
  C=$CELLS/$id/B$b
  # Correctness prefers the separately built watchdog twins.
  P=$C/prefill; [ -f $C/prefill_w/plan.so ] && P=$C/prefill_w
  D=$C/decode; [ -f $C/decode_w/plan.so ] && D=$C/decode_w
  NEEDS_MIB=$needs_mib $ROOT/scripts/mx_gpu_job.sh $gpu 1700 $C/check env -u ALL_PROXY -u all_proxy \
    bash $ROOT/scripts/mx_check_modes.sh $rand $P/plan.so $D/plan.so $b 512 $C $prompts ;;
measure)
  b=$3; rand=$4; prompts=$5; gpu=$6; round=$7
  C=$CELLS/$id/B$b; O=$C/measure/round$round
  [ -f $O/.done ] && exit 0
  mkdir -p $O
  # MX4 prompt §9: one fresh process per round, warm-up 1 + timed 2; a timed
  # job is at most 15 minutes.
  NEEDS_MIB=$needs_mib $ROOT/scripts/mx_gpu_job.sh $gpu 900 $O/guard env -u ALL_PROXY -u all_proxy \
    python -m tilemega.serving.measure --model $rand --prefill-so $C/prefill/plan.so --decode-so $C/decode/plan.so \
    --prompt-ids $prompts --batch $b --out $O --mode L1 --decode-loop auto \
    --max-new-tokens 1024 --warmup 1 --repeats 2 --step-events 0 \
    --policy configs/bench/measurement_policy_c550.json > $O/run.log 2>&1 && touch $O/.done ;;
esac
