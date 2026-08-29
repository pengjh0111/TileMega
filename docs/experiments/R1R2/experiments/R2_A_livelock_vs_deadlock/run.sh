#!/bin/bash
# R2-A: livelock vs deadlock single-shot determination.
# Reuses E4's spin_wait_tokenchain.cubin + host_test_biggrid_tokenchain (grid=170 hangs).
set -x
cd "$(dirname "$0")/../E4_spin_wait/variants"

# 1. Launch the hang in the background (do NOT kill/timeout it).
rm -f /tmp/r2a_hang.log
./host_test_biggrid_tokenchain 170 > /tmp/r2a_hang.log 2>&1 &
echo $! > /tmp/r2a_hang.pid
sleep 8   # let it settle into the stuck state

# 2. SM utilization while hung
nvidia-smi dmon -s u -c 15

# 3. Power / clock while hung
nvidia-smi -q -d POWER,CLOCK

# 4. cuda-gdb attach (requires ptrace_scope=0)
sudo sysctl -w kernel.yama.ptrace_scope=0
PID=$(cat /tmp/r2a_hang.pid)
cuda-gdb -p $PID -batch -ex "info cuda threads"
sleep 15
cuda-gdb -p $PID -batch -ex "info cuda threads"
cuda-gdb -p $PID -batch -ex "cuda kernel 0 block 14,0,0 thread 0,0,0" -ex "x/16i \$pc-0x40"

# 5. clean up the hang
kill -9 $PID

# 6. compute-sanitizer (fresh launches, may or may not reproduce the hang)
timeout 90  compute-sanitizer --tool synccheck  ./host_test_biggrid_tokenchain 170
timeout 120 compute-sanitizer --tool racecheck   ./host_test_biggrid_tokenchain 170
