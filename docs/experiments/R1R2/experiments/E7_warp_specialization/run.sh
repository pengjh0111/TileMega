#!/bin/bash
# E7: warp-specialization controllability investigation.
# Reproduces the exact commands/outputs cited in result.md.
set -x
cd "$(dirname "$0")"

# 1. Current (actually-used-throughout) cuda-tile toolchain rejects num_worker_warps_per_cta.
/data/cuda-tile/build/bin/cuda-tile-opt entry_num_worker_warps.mlir
echo "exit=$?"

# 2. Check whether the newer cuda-tile (fetched as a tensor-ir dependency) built a standalone
#    cuda-tile-opt binary that could test the field's acceptance there.
find /data/tensor-ir/build/_deps/tensor_ir_cuda_tile-build -iname "cuda-tile-opt*"
ls -la /data/tensor-ir/build/_deps/tensor_ir_cuda_tile-build/bin/ 2>/dev/null

# 3. Check whether tensor_ir-opt (which links the newer cuda-tile as a library) registers the
#    cuda_tile dialect and could be used as an alternative test route.
/data/tensor-ir/build/bin/tensor_ir-opt --help | head -5
