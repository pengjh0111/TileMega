#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              errStr);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

float data[] = {0,   5,   10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,
                65,  70,  75,  80,  85,  90,  95,  100, 105, 110, 115, 120, 125,
                130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190,
                195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255,
                260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320,
                325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385,
                390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450,
                455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515,
                520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
                585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635};

int main() {
  CUdevice cuDevice;
  CUcontext cuContext;
  CUmodule cuModule;
  CUfunction example_kernel;
  CUstream stream;

  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CUDA_CHECK(cuModuleLoad(&cuModule, "example.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&example_kernel, cuModule, "example_kernel"));

  CUdeviceptr data_ptr;
  CUDA_CHECK(cuMemAlloc(&data_ptr, sizeof(data)));
  CUDA_CHECK(cuMemcpyHtoD(data_ptr, data, sizeof(data)));

  void *kernel_args[] = {&data_ptr};
  CUDA_CHECK(cuLaunchKernel(example_kernel,
                            1, 1, 1,
                            1, 1, 1,
                            0,
                            stream,
                            kernel_args,
                            NULL
                            ));
  CUDA_CHECK(cuCtxSynchronize());

  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));

  return 0;
}
