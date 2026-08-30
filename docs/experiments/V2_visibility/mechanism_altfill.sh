#!/usr/bin/env bash
# Paired experiment isolating the mechanism: stale shared-memory warp partials.
#
# Same kernel (v2_f_hostdata: spin -> load a HOST-written chunk -> cross-warp
# reduce), same grid, same rep count. The only difference is whether the value
# being reduced changes between reps.
#
#   C) constant fill  -> the correct partial is the same every rep, so a warp
#                        slot left over from the previous launch happens to hold
#                        the right number. Only rep 0 fails.
#   B) alternating    -> the correct partial differs per rep, so a stale slot is
#                        now visible as a wrong answer. EVERY rep fails, and the
#                        wrong values are sums of four terms drawn from
#                        {0, 256*v_current, 256*v_previous}.
set -euo pipefail
cd "$(dirname "$0")"
GRID=${GRID:-340}
REPS=${REPS:-8}
echo "=== C) constant fill (control) ==="
for i in 1 2 3; do
  V2_HOSTFILL_CHUNK=128 V2_REPS=$REPS V2_VERBOSE=1 \
    ./v2_host V2_f/v2_f_hostdata.cubin v2_f_hostdata $GRID scalar 1024 1 2>&1 | grep -E "rep " || echo "  (all reps clean)"
  echo "  --"
done
echo "=== B) alternating fill 1.0 / 3.0 ==="
for i in 1 2 3; do
  V2_HOSTFILL_CHUNK=128 V2_ALT_FILL=1 V2_ALT_HI=3.0 V2_REPS=$REPS V2_VERBOSE=1 \
    ./v2_host V2_f/v2_f_hostdata.cubin v2_f_hostdata $GRID scalar 1024 1 2>&1 | grep -E "rep " || echo "  (all reps clean)"
  echo "  --"
done
echo "=== D) v1_min with the expectation pinned (producer writes a constant, so"
echo "       the correct answer does NOT scale with the host fill) ==="
for i in 1 2 3; do
  V2_ALT_NOSCALE=1 V2_ALT_FILL=1 V2_ALT_HI=3.0 V2_REPS=$REPS V2_VERBOSE=1 \
    ./v2_host sass_audit/v1_min_133.cubin v1_min $GRID scalar 1024 1 2>&1 | grep -E "rep " || echo "  (all reps clean)"
  echo "  --"
done
