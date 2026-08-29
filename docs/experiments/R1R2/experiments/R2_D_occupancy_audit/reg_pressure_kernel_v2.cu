// Control group v2 for R2-D: force higher register pressure (~228) via nvcc.
// Uses 220 independent local scalars is impractical to write by hand, so instead
// use a moderately large array with a non-trivial (non-simplifiable) dependency
// pattern, compiled at -O0 with a HIGH -maxrregcount cap (so ptxas doesn't spill)
// to get register count in the same ballpark as tileiras's REG=228.
#include <cstdio>

extern "C" __global__ void reg_pressure_kernel(float* out, float seed) {
    float r[220];
    r[0] = seed + threadIdx.x;
    r[1] = seed * 2.0f + threadIdx.x;
#pragma unroll
    for (int i = 2; i < 220; i++) {
        // non-linear-recurrence-friendly pattern: depends on two distinct prior
        // elements with different strides, harder for the compiler to collapse
        // into a short dependency chain / constant-fold away.
        r[i] = r[i-1] * 1.0000001f + r[i-2] * 0.0000003f + 0.0000001f * i;
    }
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 220; i++) acc += r[i] * (float)(i % 7 + 1);
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
