import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cudagrad.cuda._kernel_utils import cuda_kernel, BLOCK_SIZE, KERNEL_DIR

SIZE_OF_FLOAT = 4


@cuda_kernel
def dot1D(a1: np.array, a2: np.array) -> np.array:
    assert (a1.ndim == 1 and a2.ndim == 1), "Inputs must be 1D numpy arrays"
    assert (a1.size == a2.size), "Input size mismatch"

    a1 = a1.astype(np.float32)
    a2 = a2.astype(np.float32)
    input_size = np.array([a1.size]).astype(np.float32)

    output = np.array([0, ]).astype(np.float32)
    with open(f"{KERNEL_DIR}/dot1D.cuh", "r") as f:
        kernel_code = f.read()
    mod = SourceModule(kernel_code)
    cuda.start_profiler()
    kernel = mod.get_function("dot1D")

    kernel(cuda.In(a1), cuda.In(a2), cuda.Out(output), cuda.In(input_size), block=(BLOCK_SIZE, 1, 1),
           grid=(int(np.ceil(a1.size / BLOCK_SIZE)), 1, 1), shared=BLOCK_SIZE * SIZE_OF_FLOAT)
    cuda.stop_profiler()
    return output


if __name__ == "__main__":
    # Testing
    size = 100_000_000
    a = np.random.rand(size)
    b = np.random.rand(size)

    c = dot1D(a, b)
    print(c)
