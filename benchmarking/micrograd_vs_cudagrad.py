import time
import sys
import numpy as np
from micrograd.engine import Value
from cudagrad.tensor import Tensor1D

sys.setrecursionlimit(400000)

if __name__ == "__main__":
    size = 200_000

    start = time.time()

    a_values = [Value(np.random.randn(1)) for i in range(size)]
    b_values = [Value(np.random.randn(1)) for i in range(size)]


    def dot(a_arr, b_arr):
        sum = 0.0
        for i in range(len(a_arr)):
            sum += a_arr[i] * b_arr[i]
        return sum


    dot_product = dot(a_values, b_values)
    dot_product.backward()

    print(f"Micrograd Time: {time.time() - start} seconds")

    start = time.time()

    a_values = Tensor1D(np.random.randn(size))
    b_values = Tensor1D(np.random.randn(size))

    dot_product = a_values.dot(b_values)
    dot_product.backward()
    print(f"Cuda Time: {time.time() - start} seconds")
