#include<math.h>

__global__ void log1D(float *a1, float *output, float *N)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N[0])
    output[idx] = log(a1[idx]);
}