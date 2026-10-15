// P-2: which architecture macros each compiler pass defines.
#include <cstdio>
__global__ void k(int* o) { o[0] = warpSize; }
int main() { return 0; }
