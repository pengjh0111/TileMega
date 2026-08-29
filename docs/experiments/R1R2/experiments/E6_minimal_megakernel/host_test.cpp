// E6: correctness + overlap-evidence check for the minimal megakernel.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { CUresult _e = (x); if (_e != CUDA_SUCCESS) { \
  const char *s=nullptr; cuGetErrorString(_e,&s); \
  fprintf(stderr, "%s:%d CUDA error %d: %s\n", __FILE__, __LINE__, _e, s?s:"?"); \
  exit(1);} } while(0)

static const int N = 262144;
static const int NX = 1024;

int main(int argc, char** argv) {
  int trials = argc > 1 ? atoi(argv[1]) : 1;

  CUDA_CHECK(cuInit(0));
  CUdevice dev; CUDA_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CUDA_CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
  CUmodule mod; CUDA_CHECK(cuModuleLoad(&mod, "minimal_megakernel.cubin"));
  CUfunction fn; CUDA_CHECK(cuModuleGetFunction(&fn, mod, "minimal_megakernel"));

  std::vector<float> hA(N), hB(N), hX(NX);
  for (int i = 0; i < N; i++) { hA[i] = (float)(i % 97); hB[i] = (float)((i*3) % 89); }
  for (int i = 0; i < NX; i++) hX[i] = (float)(i % 17);

  CUdeviceptr dA,dB,dC,dD,dX,dE,dFlag,dProgress,dSample;
  CUDA_CHECK(cuMemAlloc(&dA, N*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dB, N*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dC, N*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dD, N*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dX, NX*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dE, NX*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dFlag, sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&dProgress, sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&dSample, sizeof(int)));
  CUDA_CHECK(cuMemcpyHtoD(dA, hA.data(), N*sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(dB, hB.data(), N*sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(dX, hX.data(), NX*sizeof(float)));

  int pass = 0, fail = 0;
  int minSample = 1<<30, maxSample = -1;
  for (int t = 0; t < trials; t++) {
    // poison C/D/E and zero flag/progress/sample before every trial
    CUDA_CHECK(cuMemsetD32(dC, 0xdeadbeef, N));
    CUDA_CHECK(cuMemsetD32(dD, 0xdeadbeef, N));
    CUDA_CHECK(cuMemsetD32(dE, 0xdeadbeef, NX));
    CUDA_CHECK(cuMemsetD32(dFlag, 0, 1));
    CUDA_CHECK(cuMemsetD32(dProgress, 0, 1));
    CUDA_CHECK(cuMemsetD32(dSample, 0xdeadbeef, 1));

    void* args[] = { &dA,&dB,&dC,&dD,&dX,&dE,&dFlag,&dProgress,&dSample };
    CUDA_CHECK(cuLaunchKernel(fn, 3,1,1, 1,1,1, 0,0,args,NULL));
    CUresult sync = cuCtxSynchronize();
    if (sync != CUDA_SUCCESS) { printf("[trial %d] cuCtxSynchronize FAILED\n", t); fail++; continue; }

    std::vector<float> hC(N), hD(N), hE(NX);
    int sample=0, progress=0;
    CUDA_CHECK(cuMemcpyDtoH(hC.data(), dC, N*sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(hD.data(), dD, N*sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(hE.data(), dE, NX*sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(&sample, dSample, sizeof(int)));
    CUDA_CHECK(cuMemcpyDtoH(&progress, dProgress, sizeof(int)));

    bool ok = true;
    for (int i = 0; i < N && ok; i++) {
      float expC = hA[i] + hB[i];
      float expD = expC * 2.0f;
      if (fabsf(hC[i]-expC) > 1e-4f || fabsf(hD[i]-expD) > 1e-4f) {
        if (fail < 3) printf("[trial %d] MISMATCH at i=%d: C=%f exp=%f D=%f exp=%f\n", t,i,hC[i],expC,hD[i],expD);
        ok = false;
      }
    }
    for (int i = 0; i < NX && ok; i++) {
      float expE = hX[i]*3.0f;
      if (fabsf(hE[i]-expE) > 1e-4f) {
        if (fail < 3) printf("[trial %d] MISMATCH E at i=%d: E=%f exp=%f\n", t,i,hE[i],expE);
        ok = false;
      }
    }
    if (ok) pass++; else fail++;
    if (sample < minSample) minSample = sample;
    if (sample > maxSample) maxSample = sample;
    if (t < 5 || !ok) printf("[trial %d] ok=%d block2_sample(progress counter at time block2 started)=%d final_progress=%d\n", t, ok, sample, progress);
  }
  printf("=== SUMMARY over %d trials ===\nPASS=%d FAIL=%d block2_sample range=[%d,%d] (producer's final per-chunk counter = 256; a sample < 256 means block 2 observed the producer mid-flight, i.e. concurrent execution within the single launch)\n",
         trials, pass, fail, minSample, maxSample);
  return fail == 0 ? 0 : 1;
}
