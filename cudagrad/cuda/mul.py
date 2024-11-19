import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cudagrad.cuda._kernel_utils import cuda_kernel, BLOCK_SIZE, KERNEL_DIR


@cuda_kernel
def mul1D(a1: np.array, a2: np.array) -> np.array:
    assert (a1.ndim == 1 and a2.ndim == 1), "Inputs must be 1D numpy arrays"
    assert (a1.size == a2.size), "Input size mismatch"

    a1 = a1.astype(np.float32)
    a2 = a2.astype(np.float32)
    input_size = np.array([a1.size]).astype(np.float32)

    output = np.zeros_like(a1)
    with open(f"{KERNEL_DIR}/mul1D.cuh", "r") as f:
        kernel_code = f.read()
    mod = SourceModule(kernel_code)
    cuda.start_profiler()
    kernel = mod.get_function("mul1D")

    kernel(cuda.In(a1), cuda.In(a2), cuda.Out(output), cuda.In(input_size), block=(BLOCK_SIZE, 1, 1),
           grid=(int(np.ceil(a1.size / BLOCK_SIZE)), 1, 1))
    cuda.stop_profiler()
    return output


if __name__ == "__main__":
    # Testing
    a = np.arange(6)
    b = np.arange(6)

    c = mul1D(a, b)
    print(c)
