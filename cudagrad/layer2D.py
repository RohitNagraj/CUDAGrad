import numpy as np

from cudagrad.tensor import Tensor2D


# TODO: There is some bug with the Tensor2D class. The grad of self.b are suspiciously whole numbers.


class Layer:
    def __init__(self, n_input: int, n_output: int):
        self.n_input = n_input
        self.n_output = n_output

        self.w = Tensor2D(np.random.randn(n_input, n_output))
        self.b = Tensor2D(np.random.randn(n_output))

    def __repr__(self):
        return f"Layer(n_input={self.n_input}, n_output={self.n_output})"

    def __call__(self, x: Tensor2D):
        activation = (x @ self.w) + self.b
        output = activation.relu()
        return output

    def parameters(self):
        return [self.w, self.b]


if __name__ == '__main__':
    batch_size = 16
    n_input = 10
    n_output = 20

    layer = Layer(n_input, n_output)

    x = Tensor2D(np.random.randn(batch_size, n_input))
    y = layer(x)
    y.backward()
    print(y)
