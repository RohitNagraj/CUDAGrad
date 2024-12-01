__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  for (int j=0; j<1000000; j++)
      dest[i] = a[i] * b[i];
}