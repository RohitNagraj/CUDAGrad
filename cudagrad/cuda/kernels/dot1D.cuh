__global__ void dot1D(float *a, float *b, float *output, float *N)
{
    extern __shared__ float sharedMemory[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes its partial dot product
    float partialSum = 0.0;
    if (idx < N[0])
    {
        partialSum = a[idx] * b[idx];
    }

    sharedMemory[tid] = partialSum;
    __syncthreads();

    // Store partial results in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sharedMemory[tid] += sharedMemory[tid + stride];
        }
        __syncthreads();
    }

    if (tid==0)
    {
        atomicAdd(&output[0], sharedMemory[0]);
    }
}