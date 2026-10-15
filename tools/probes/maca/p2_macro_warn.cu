// P-2: report architecture macros per compilation pass via #warning.
#if defined(__MACA_ARCH__)
#define STR2(x) #x
#define STR(x) STR2(x)
#pragma message("device pass: __MACA_ARCH__=" STR(__MACA_ARCH__))
#else
#pragma message("host pass")
#endif
#if defined(__CUDA_ARCH__)
#pragma message("__CUDA_ARCH__ defined")
#endif
#if defined(__CUDACC__)
#pragma message("__CUDACC__ defined")
#endif
#if defined(__NVCC__)
#pragma message("__NVCC__ defined")
#endif
#if defined(__MACACC__)
#pragma message("__MACACC__ defined")
#endif
__global__ void k(int* o) { constexpr int w = warpSize; o[0] = w; }
