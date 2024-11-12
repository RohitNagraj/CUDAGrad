import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
import time

SIZE = 2 ** 20
BLOCK_SIZE = 1024

start = time.time()

with open("hadamard.cuh", "r") as f:
    kernel_code = f.read()

# Compile the kernel
mod = SourceModule(kernel_code)

multiply_them = mod.get_function("multiply_them")

a = numpy.random.rand(SIZE).astype(numpy.float32)
b = numpy.random.rand(SIZE).astype(numpy.float32)

dest = numpy.zeros_like(a)

multiply_them(drv.Out(dest), drv.In(a), drv.In(b),block=(BLOCK_SIZE, 1, 1), grid=(int(numpy.ceil(SIZE / BLOCK_SIZE)), 1))


res = dest[:5] - (a[:5] * b[:5])
print(res)
stop = time.time() - start
print(f"Time is {stop * 1e3} ms")


