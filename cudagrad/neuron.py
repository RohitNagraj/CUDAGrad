import time

import numpy as np

from cudagrad.tensor import Tensor1D


class Neuron:
    def __init__(self, n_input):
        self.backend = 'cuda'
        self.n_input = n_input
        self.w = Tensor1D(np.random.randn(n_input), backend=self.backend)
        self.b = Tensor1D(np.random.randn(1), backend=self.backend)

    def __repr__(self):
        return f"Neuron(n_input={self.n_input})"

    def __call__(self, x: Tensor1D):
        start = time.time()
        activation = self.w.dot(x) + self.b
        output = self._tanh(activation)
        print(f"Time taken per neuron: {time.time() - start}")
        return output

    def _tanh(self, x: Tensor1D):
        two_x = x * 2
        exp_two_x = two_x.exp()
        numerator = exp_two_x - 1
        denominator = exp_two_x + 1
        result = numerator / denominator
        return result

    def parameters(self):
        return [self.w, self.b]


if __name__ == '__main__':
    size = 5
    n = Neuron(size)
    x = Tensor1D(np.random.randn(size))
    y = n(x)
    y.backward()
    print(y)
