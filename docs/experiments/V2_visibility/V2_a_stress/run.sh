#!/bin/bash
# V2-a': is V1-a's clean sheet mechanism-safe, or just timing-lucky?
#   plain    : 500 fresh processes (V1 only ever ran 50)
#   hostdelay: host sleeps before launch -> producer/consumer skew changes
#   stream   : non-blocking stream instead of the null stream
#   nosync   : flag pre-set, nobody ever spins (upper bound on "no spin" safety)
set -u
cd "$(dirname "$0")/.."
N=${N:-500}
cell() { local lbl=$1; shift; local fail=0
  for i in $(seq 1 $N); do
    o=$(env "$@" V2_REPS=1 timeout 25 ./v2_host V2_a_stress/v1_a.cubin v1_a 340 blockN 2.0 1024 2>&1)
    grep -q "all slots verified" <<<"$o" || fail=$((fail+1))
  done
  printf "  %-24s %s/%s\n" "$lbl" "$fail" "$N"; }
echo "V2-a': V1-a, grid=340"
cell "plain"              V2_X=1
cell "host delay 5000us"  V2_HOST_DELAY_US=5000
cell "host delay 50us"    V2_HOST_DELAY_US=50
cell "non-blocking stream" V2_USE_STREAM=1
cell "no-spin (flag preset)" V2_NO_SYNC=1
