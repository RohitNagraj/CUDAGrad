__global__ void mul1D(float *a1, float *a2, float *output, float *N)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N[0])
    output[idx] = a1[idx] * a2[idx];
}