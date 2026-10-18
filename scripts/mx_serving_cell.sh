#!/usr/bin/env bash
# TileMega cell pipeline on the C550 (MX-E).
#   tm_cell.sh export  <id> <family> <cfg_dir>
#   tm_cell.sh solve   <id> <batch> <decode|prefill>          (CPU; device query via guard on $QGPU)
#   tm_cell.sh check   <id> <batch> <rand_dir> <prompts> <gpu> (C-2 and C-1)
#   tm_cell.sh measure <id> <batch> <rand_dir> <prompts> <gpu> <round>
# Plans: runs/mx2/tm/<id>/B<b>/<phase>/plan.so. Raw data stays under runs/mx1.
set -eu
ROOT=/root/tilemega; cd $ROOT
TARGET=runs/mx1/calib/xcore1000_c550.json
QGPU=${QGPU:-6}
JOBS=${JOBS:-8}
# Solves compile against a frozen worktree (/root/tilemega_solve, its own
# build): an artifact's identity snapshot covers every source file, so header
# edits in the development tree during a 30-minute solve would invalidate it.
SOLVE_BIN=${SOLVE_BIN:-/root/tilemega/build/tools/tilemega}
cmd=$1; id=$2
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
  C=runs/mx2/tm/$id/B$b/$phase; mkdir -p $C
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
  trap 'rm -f $C/plan.so.m*.tsv $C/plan.so.verified*.tsv' EXIT INT TERM
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
  la=0
  # Fragment-packed ("tiled") weights measure faster than row-major on this
  # device, so they are the default; LAYOUT=row overrides for an A/B.
  layout=${LAYOUT:-tiled}
  extra=""; [ $phase = decode ] && extra="--attention-impl simt"
  env CUDA_VISIBLE_DEVICES= MACA_VISIBLE_DEVICES= \
    TILEMEGA_DEVICE_RUN_PREFIX="env NEEDS_MIB=2048 /root/deps/mxjob.sh $QGPU 300 $C/devq" \
    timeout ${CELL_TIMEOUT:-18000} nice $SOLVE_BIN compile runs/mx1/export/${id}_$phase/bridge.json $C/plan.so \
      --serving $phase --emit serving --batch $b --prompt-len 64 --pg off $extra \
      --solve $TARGET --runtime-target $TARGET --selection predicted --watchdog 1 \
      --nonpaged-weight-layout $layout --nonpaged-la $la --search-jobs $JOBS \
      --search-passes 2 --search-budget-ms ${SEARCH_BUDGET_MS:-1200000} \
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
            TILEMEGA_DEVICE_RUN_PREFIX="env NEEDS_MIB=2048 /root/deps/mxjob.sh $QGPU 300 $C/devq" \
            timeout ${CELL_TIMEOUT:-18000} nice $SOLVE_BIN compile \
              runs/mx1/export/${id}_$phase/bridge.json $C/plan.so \
              --serving $phase --emit serving --batch $b --prompt-len 64 --pg off $extra \
              --solve $TARGET --runtime-target $TARGET --selection predicted --watchdog 1 \
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
  rm -f $C/plan.so.m*.tsv $C/plan.so.verified*.tsv ;;
check)
  b=$3; rand=$4; prompts=$5; gpu=$6
  C=runs/mx2/tm/$id/B$b
  NEEDS_MIB=20480 /root/deps/mxjob.sh $gpu 1700 $C/check env -u ALL_PROXY -u all_proxy \
    bash /root/deps/c1c2.sh $rand $C/prefill/plan.so $C/decode/plan.so $b 512 $C $prompts ;;
measure)
  b=$3; rand=$4; prompts=$5; gpu=$6; round=$7
  C=runs/mx2/tm/$id/B$b; O=$C/measure/round$round
  [ -f $O/.done ] && exit 0
  mkdir -p $O
  NEEDS_MIB=20480 /root/deps/mxjob.sh $gpu 1700 $O/guard env -u ALL_PROXY -u all_proxy \
    python -m tilemega.serving.measure --model $rand --prefill-so $C/prefill/plan.so --decode-so $C/decode/plan.so \
    --prompt-ids $prompts --batch $b --out $O --mode L1 --decode-loop auto \
    --max-new-tokens 1024 --warmup 1 --repeats 3 \
    --policy configs/bench/measurement_policy_c550.json > $O/run.log 2>&1 && touch $O/.done ;;
esac
