import pycuda.driver as cuda

KERNEL_DIR = "cudagrad/cuda/kernels"
BLOCK_SIZE = 1024
CUDA_DEVICE = 0


def cuda_kernel(func):
    def wrapper(*args, **kwargs):
        cuda.init()
        device = cuda.Device(CUDA_DEVICE)
        ctx = device.make_context(CUDA_DEVICE)
        result = func(*args, **kwargs)
        # ctx.synchronize()
        ctx.pop()
        return result

    return wrapper
