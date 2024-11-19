import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from cudagrad.cuda._kernel_utils import cuda_kernel, BLOCK_SIZE, KERNEL_DIR


@cuda_kernel
def log1D(a1: np.array) -> np.array:
    """
    Element wise log base e.
    :param a1:
    :param a2:
    :return:
    """
    assert (a1.ndim == 1 ), "First param must be 1D numpy array"

    a1 = a1.astype(np.float32)
    input_size = np.array([a1.size]).astype(np.float32)

    output = np.zeros_like(a1)
    with open(f"{KERNEL_DIR}/log1D.cuh", "r") as f:
        kernel_code = f.read()
    mod = SourceModule(kernel_code)
    cuda.start_profiler()
    kernel = mod.get_function("log1D")

    kernel(cuda.In(a1), cuda.Out(output), cuda.In(input_size), block=(BLOCK_SIZE, 1, 1),
           grid=(int(np.ceil(a1.size / BLOCK_SIZE)), 1, 1))
    cuda.stop_profiler()
    return output


if __name__ == "__main__":
    # Testing
    a = np.array([np.exp(x) for x in range(1, 2)])

    c = log1D(a)
    print(c)
