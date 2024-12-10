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
        activation = self.w.dot(x) + self.b
        output = self._tanh(activation)
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
    x = Tensor1D([10, 20, 30, 40])
    w = Tensor1D([1, 2, 3, 4])
    b = Tensor1D([1, ])
    activation = w.dot(x) + b
    output = activation.relu()

    output.backward()

    from cudagrad.tensor import Tensor2D

x = Tensor1D([10, 20, 30, 40]).reshape(-1, 1)
w = Tensor2D([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
b = Tensor2D([1, 2, 3, 4]).reshape(-1, 1)

activation = w.dot(x) + b
output = activation.relu()

output.backward()
