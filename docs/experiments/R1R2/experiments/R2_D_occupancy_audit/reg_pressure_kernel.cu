// Control group for R2-D: a trivial nvcc-compiled CUDA kernel with a register count
// deliberately forced close to the tileiras spin_wait_tokenchain kernel's REG=228,
// via -maxrregcount=228. Does nothing meaningful; only exists so we can query
// cuOccupancyMaxActiveBlocksPerMultiprocessor on an ordinary (non-tileiras) cubin
// and confirm the driver's occupancy math itself is correct when given the ACTUAL
// launch blockDim (as opposed to R2-D's finding that the tileiras kernel's occupancy
// queries in E4/E5 used blockSize=1, which does not match the kernel's actual
// required 256 threads/block baked in via EIATTR_REQNTID).
#include <cstdio>

extern "C" __global__ void reg_pressure_kernel(float* out, float seed) {
    float r[64];
    r[0] = seed + threadIdx.x;
#pragma unroll
    for (int i = 1; i < 64; i++) {
        r[i] = r[i-1] * 1.0000001f + 0.0000001f * i;
    }
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; i++) acc += r[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
