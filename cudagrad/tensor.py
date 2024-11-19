import numpy as np
from triton.tools.build_extern import build

from cudagrad import cuda


class Tensor1D:
    def __init__(self, data: list | np.ndarray, _children=(), _op="", label="", backend='cuda') -> None:
        self.data = np.array(data).astype(np.float32)
        self.shape = self.data.shape
        self.backend = backend
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor1D(data={str(self.data.tolist())})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor1D) else Tensor1D(other * np.ones_like(self.data))

        assert len(self.data) == len(other.data), "Length of the two tensors must match."

        if self.backend == 'cuda':
            out = Tensor1D(cuda.add.add1D(self.data, other.data), (self, other), "+")
        else:
            out = Tensor1D(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO: Make it run on CUDA.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # Torch does not allow vector outputs to provide gradients implicitly. Our behaviour is different
        other = other if isinstance(other, Tensor1D) else Tensor1D(other * np.ones_like(self.data))
        if self.backend == 'cuda':
            out = Tensor1D(cuda.mul.mul1D(self.data, other.data), (self, other), "*")
        else:
            out = Tensor1D(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: Make it run on cuda
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other, modulo=None):
        assert isinstance(other, (int, float)), "Only int and float powers are supported for now."
        if self.backend == 'cuda':
            out = Tensor1D(cuda.pow.pow1Dconst(self.data, other), (self,), "**")
        else:
            out = Tensor1D(self.data ** other, (self,), "**")

        def _backward():
            # TODO: Make it run on cuda
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def dot(self, other):
        # The behaviour of dot(Tensor1D, scalar) is different from numpy. Here we convert it to vector and process.
        other = other if isinstance(other, Tensor1D) else Tensor1D(other * np.ones_like(self.data))

        if self.backend == 'cuda':
            out = Tensor1D(cuda.dot.dot1D(self.data, other.data), (self, other), "dot")
        else:
            out = Tensor1D(np.dot(self.data, other.data), (self, other), "dot")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        if self.backend == 'cuda':
            out = Tensor1D(cuda.log.log1D(self.data), (self,), "log")
        else:
            out = Tensor1D(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        TODO: The current logic runs every child sequentially, even if they are at the same level in the graph,
        parallelize it for all childen in one layer at once.
        """
        topological_order = []
        visited = set()

        def build_topological_order(v):
            if v not in visited:
                for child in v._prev:
                    build_topological_order(child)
                topological_order.append(v)
                visited.add(v)

        build_topological_order(self)
        self.grad = np.ones_like(self.data)

        for v in reversed(topological_order):
            v._backward()

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor1D) else Tensor1D(other * np.ones_like(self.data))
        return self * (other ** -1)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return -1 * self.data

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self / other

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other


if __name__ == '__main__':
    # Testing the tensor class here, enforcing output is similar to torch.
    import time

    backend = 'numpy'
    # backend = 'cuda'
    size = 4

    start = time.time()

    # a = Tensor1D(np.random.rand(size), backend=backend, label='a')
    # b = Tensor1D(np.random.rand(size), backend=backend, label='b')
    a = Tensor1D([2, 3])
    b = Tensor1D([3, 4])
    c = a * b
    d = 2 * c

    d.backward()
    print(f"Backend: {backend}, Time Taken: {time.time() - start} seconds")
