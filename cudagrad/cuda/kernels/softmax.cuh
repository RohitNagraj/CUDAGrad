// extern "C++"
// template <typename T, int NUM>
// __inline__ __global__ T warpReduceMax(T* val, int thread_group_width = 32) {
// #pragma unroll
//   for (int i = 0; i < NUM; i++) {
// #pragma unroll
//     for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
//       val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
//     }
//   }
//   return (T)(0.0f);
// }

extern "C"
__device__ void warpReduceMaxOptimization(float *data, float *result){
    // Warp reduce max for 32 threads
    // Calculate a mask for all the active threads in the warp
    unsigned mask = (1 << blockDim.x) - 1;

    // Store thread assigned value in a register
    float maxVal = data[blockDim.x * blockIdx.x + threadIdx.x];

    // Recursively reduce the max value
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        
        maxVal = max(maxVal, __shfl_xor_sync(mask, maxVal, offset));
        __syncwarp(mask=mask);

    }

    // Store the max value in the result array
    result[blockIdx.x] = maxVal;
}
