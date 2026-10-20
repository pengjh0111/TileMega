# AS-1 native Ascend 950PR probes

All commands run from `/home/d50064638/junhuip/TileMega`. Source the private
environment explicitly. Builds use one CPU compilation at a time. Run each
device command individually with the telemetry wrapper; it creates a new output
directory and never requires the device to be idle.

```bash
source /home/d50064638/junhuip/tilemega-as1/scripts/as1_env.sh
```

## ACL mapping and resource inventory

Verified with CANN 9.1.0, driver 25.7.0, `Ascend950PR_9579`, physical devices 0
and 1 in separate processes. Each sees only logical device 0. No kernel is
launched by this inventory. The exact successful compiler argv and failed
attempts are retained in `tilemega-as1/build/acl_inventory/command*.json`.

```bash
c++ -std=c++17 -O0 tools/probes/ascend950/acl_inventory.cpp \
  -I/usr/local/Ascend/cann/include \
  -I/usr/local/Ascend/cann/x86_64-linux/pkg_inc \
  -I/usr/local/Ascend/cann/x86_64-linux/pkg_inc/runtime \
  -L/usr/local/Ascend/cann/lib64 -Wl,-rpath,/usr/local/Ascend/cann/lib64 \
  -lascendcl -lruntime /usr/local/Ascend/cann/x86_64-linux/lib64/libtiling_api.a \
  -lplatform -ldl -lc_sec -pthread -lunified_dlog \
  -o /home/d50064638/junhuip/tilemega-as1/build/acl_inventory/acl_inventory
```

```bash
python3 python/tilemega/bench/as1_run.py --physical-device 0 \
  --out "$REPO/runs/as1/jobs/acl_inventory_repro_0" --timeout-s 30 -- \
  "$AS1_WORK/build/acl_inventory/acl_inventory"
```

Use a fresh output directory for each invocation. For physical device 1 change
both the selection and output name; never start both processes concurrently.

## Existing torch_npu environment

The private `envs/as1-npu` venv exposes the preinstalled `/data/miniconda3/envs/dev`
packages read-only. Verified: torch 2.7.1+cpu and torch_npu
2.7.1.post7.dev20260707 execute a 128-element BF16 addition with zero error.
No pip installation or model download is required.

```bash
python3 python/tilemega/bench/as1_run.py --physical-device 0 \
  --out "$REPO/runs/as1/jobs/torch_npu_repro" --timeout-s 60 -- \
  "$AS1_WORK/envs/as1-npu/bin/python" \
  "$REPO/tools/probes/ascend950/torch_npu_probe.py"
```

## Wrapper and cleanup safety checks

Run these individually. They launch no device kernel. Cleanup fixtures remain
under `tilemega-as1/tmp` as evidence; only the deliberately removable fixture
is unlinked. The timeout check starts and tracks its own sentinel process.

```bash
python3 tools/probes/ascend950/check_cleanup.py
python3 tools/probes/ascend950/check_runner.py
```

These probes do not establish GEMM, MIX, persistent scheduling, cross-core
visibility, or TileMega end-to-end correctness. Unmeasured capabilities remain
disabled.
