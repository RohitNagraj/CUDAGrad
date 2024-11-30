import numpy as np

from cudagrad.neuron import Neuron
from cudagrad.tensor import Tensor1D


class Layer:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __repr__(self):
        return f"Layer(n_input={self.n_input}, n_output={self.n_output})"

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        outputs = Tensor1D.concat(outputs)
        return outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


if __name__ == '__main__':
    n_input = 3
    n_output = 3
    l = Layer(n_input, n_output)
    x = Tensor1D(np.random.rand(n_input))
    y = l(x)
    y.backward()
    print(y)
