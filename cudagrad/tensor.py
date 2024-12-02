import numpy as np
import cupy as cp
import cupyx.profiler

from cudagrad import cuda
from cuda._kernel_utils import KERNEL_DIR


class Tensor1D:
    """
    All the backward functions are running on CPU. Need to parallelize them.
    """

    def __init__(self, data: list | np.ndarray, _children=(), _op="", label="", backend='cuda') -> None:
        self.data = np.array(data).astype(np.float32)
        if self.data.ndim == 0:
            self.data = np.array([data, ]).astype(np.float32)
        self.shape = self.data.shape
        self.backend = backend
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        if self.label != "":
            return f"Tensor1D(data={str(self.data.tolist())}, label={self.label})"
        else:
            return f"Tensor1D(data={str(self.data.tolist())})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor1D) else Tensor1D(other * np.ones_like(self.data))

        assert self.data.size == other.data.size, "Length of the two tensors must match."

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

    def sum(self):
        other = Tensor1D(np.ones_like(self.data))
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

    def exp(self):
        if self.backend == 'cuda':
            out = Tensor1D(cuda.exp.exp1D(self.data), (self,), "exp")
        else:
            out = Tensor1D(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        TODO: The current logic runs every child sequentially, even if they are at the same level in the graph,
        parallelize it for all children in one layer at once.
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
        other = other if isinstance(other, Tensor1D) else Tensor1D(other * np.ones_like(self.data))

        assert self.data.size == other.data.size, "Length of the two tensors must match."

        if self.backend == 'cuda':
            out = Tensor1D(cuda.add.add1D(self.data, -other.data), (self, other), "-")
        else:
            out = Tensor1D(self.data - other.data, (self, other), "-")

        def _backward():
            # TODO: Make it run on CUDA.
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    @staticmethod
    def concat(tensor_list):
        data = np.concat([x.data for x in tensor_list])
        grads = np.concat([x.grad for x in tensor_list])

        out = Tensor1D(data, _children=tensor_list, _op="concat")
        out.grad = grads

        def _backward():
            for (tensor, grad) in zip(tensor_list, out.grad):
                tensor.grad += grad

        out._backward = _backward
        return out

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


class Tensor2D:

    def __init__(self, data: list | np.ndarray | cp.ndarray, _children=(), _op="", label="",
                 trackGradient=True) -> None:
        self.data = cp.array(data, dtype=cp.float32)
        self.grad = cp.zeros_like(self.data, dtype=cp.float32)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.trackGradient = trackGradient
        self._backward = lambda: None
        self._initialize_cuda_kernel()

    def CEIL_DIV(self, M, N):
        return (((M) + (N) - 1) // (N))

    def __repr__(self):
        if self.label != "":
            return f"Tensor2D(shape={self.data.shape}, label={self.label})"
        return f"Tensor2D(shape={self.data.shape})"

    def __add__(self, other):
        '''
        Add two tensors elementwise
        '''
        # Broadcast the other tensor to the shape of self tensor

        if not isinstance(other, Tensor2D):
            raise ValueError("Addition is only supported between two tensors")

        c = Tensor2D(cp.zeros_like(self.data, dtype=cp.float32), (self, other), "+")

        blocks, threads = self._calculate_elementwise_grid_and_block()
        with cupyx.profiler.profile():
            if self.data.shape != other.data.shape:
                self.addBroadcastedTensor(blocks, threads, (self.data, other.data, c.data, self.data.size))
            else:
                self.addTensor(blocks, threads, (self.data, other.data, c.data, self.data.size))


        def _backward():
            self.grad += 1.0 * c.grad
            # If shapes are not same, then we need to sum the gradients
            if self.data.shape != other.data.shape:
                other.grad += 1.0 * c.grad.sum(axis=0)
            else:
                other.grad += 1.0 * c.grad

        if self.trackGradient:
            c._backward = _backward

        return c

    def __sub__(self, other):
        '''
        Inverson of Add
        '''
        if self.data.shape != other.data.shape:
            raise ValueError(f"Shape of Tensors are not same. {self.data.shape} != {other.data.shape}")
        other = other if isinstance(other, Tensor2D) else Tensor2D(other)
        c = Tensor2D(np.zeros_like(self.data), (self, other), "-")

        with cupyx.profiler.profile():
            c.data = self.data - other.data

        def _backward():
            self.grad += 1.0 * c.grad
            other.grad -= 1.0 * c.grad

        if self.trackGradient:
            c._backward = _backward

    def __mul__(self, other):
        '''
        Multiply two tensors elementwise
        '''
        if self.data.shape != other.data.shape:
            raise ValueError(f"Shape of Tensors are not same. {self.data.shape} != {other.data.shape}")
        other = other if isinstance(other, Tensor2D) else Tensor2D(other)
        c = Tensor2D(np.zeros_like(self.data), (self, other), "*")

        with cupyx.profiler.profile():
            c.data = self.data * other.data

        def _backward():
            self.grad += 1.0 * c.grad * other.data
            other.grad += 1.0 * c.grad * self.data

        if self.trackGradient:
            c._backward = _backward

        return c

    def __matmul__(self, other):
        '''
        Matrix Multiply two tensors
        '''
        # If Shape is not same, then raise an error
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(f"Shape of Tensors are not same. {self.data.shape} != {other.data.shape}")
        other = other if isinstance(other, Tensor2D) else Tensor2D(other)
        c = Tensor2D(np.zeros((self.data.shape[0], other.data.shape[1])), (self, other), "@")
        with cupyx.profiler.profile():
            c.data = self.data @ other.data

        def _backward():
            self.grad += 1.0 * c.grad @ other.data.T
            other.grad += self.data.T @ c.grad

        if self.trackGradient:
            c._backward = _backward

        return c

    def __truediv__(self, other):
        '''
        Divide two tensors elementwise
        '''
        if self.data.shape != other.data.shape:
            raise ValueError(f"Shape of Tensors are not same. {self.data.shape} != {other.data.shape}")
        other = other if isinstance(other, Tensor2D) else Tensor2D(other)
        c = Tensor2D(np.zeros_like(self.data), (self, other), "/")

        with cupyx.profiler.profile():
            c.data = self.data / other.data

        def _backward():
            self.grad += 1.0 * c.grad / other.data
            other.grad -= 1.0 * c.grad * self.data / (other.data ** 2)

        if self.trackGradient:
            c._backward = _backward

        return c

    def T(self):
        return Tensor2D(self.data.T, label=self.label + "_T", trackGradient=self.trackGradient)

    def log(self):
        c = Tensor2D(cp.log(self.data), (self,), "log(" + self.label + ")")

        def _backward():
            self.grad += 1.0 * c.grad / self.data

        if self.trackGradient:
            c._backward = _backward
        return c

    def exp(self):
        c = Tensor2D(cp.exp(self.data), (self,), "exp(" + self.label + ")")

        def _backward():
            self.grad += 1.0 * c.grad * c.data

        if self.trackGradient:
            c._backward = _backward
        return c

    def sum(self, axis=None, keepdims=False):
        c = Tensor2D(cp.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum(" + self.label + ")")

        def _backward():
            self.grad += 1.0 * c.grad

        if self.trackGradient:
            c._backward = _backward
        return c

    def softmax(self, data):
        with cupyx.profiler.profile():
            data = self.data - cp.max(self.data, axis=1, keepdims=True)
            data = cp.exp(data)
            data = data / cp.sum(data, axis=1, keepdims=True)

        return data

    def cross_entropy_loss(self, y):
        '''self.data is the logits and y is a vetor of correct labels'''
        c = Tensor2D(cp.zeros_like(self.data), (self, self), "crossEntropyLoss(" + self.label + ")")

        with cupyx.profiler.profile():
            c.data = self.softmax(self.data)
            c.data = cp.log(c.data)
            c.data = -c.data[:, y].mean()

        def _backward():
            self.grad = self.softmax(self.data)
            self.grad[cp.arange(self.data.shape[0]), y] -= 1.0
            self.grad /= self.data.shape[0]

        if self.trackGradient:
            c._backward = _backward

        return c

    def relu(self):
        c = Tensor2D(cp.maximum(0, self.data), (self,), "relu(" + self.label + ")")

        def _backward():
            self.grad += (c.data > 0) * c.grad

        if self.trackGradient:
            c._backward = _backward
        return c

    def _initialize_cuda_kernel(self):
        with open(f"{KERNEL_DIR}/add2D.cuh", "r") as f:
            addTensorCode = f.read()
        self.addTensor = cp.RawKernel(addTensorCode, "addTensor")
        self.addTensor.compile()

        self.addBroadcastedTensor = cp.RawKernel(addTensorCode, "addBroadcastedTensor")
        self.addBroadcastedTensor.compile()

    def _calculate_grid_and_block(self):
        blocks = (self.CEIL_DIV(self.data.shape[0], 32), self.CEIL_DIV(self.data.shape[1], 32))
        threads = (32 * 32,)
        return (blocks, threads)

    def _calculate_elementwise_grid_and_block(self):
        numThreads = min(self.data.shape[1], 1024)
        numBlocks = self.CEIL_DIV(self.data.size, numThreads)
        return (numBlocks,), (numThreads,)

    def backward(self):
        #  [[Tensor, tensor], [Tensor, tensor, Tensor]] where 0th tensor can be computed first
        print("Running Backward")
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = cp.ones_like(self.data)

        for v in reversed(topo):
            v._backward()


if __name__ == '__main__':
    import time

    backend = 'cuda'
    size = 4

    start = time.time()

    # a = Tensor1D(np.random.rand(size), backend=backend, label='a')
    # b = Tensor1D(np.random.rand(size), backend=backend, label='b')
    a = Tensor1D([1], label='a')
    b = Tensor1D([2], label='b')
    c = Tensor1D.concat([a, b])
    c.label = 'c'
    d = Tensor1D([3], label='d')
    e = Tensor1D([3], label='e')
    f = Tensor1D.concat([d, e])
    g = c.dot(f)
    g.label = 'g'
    # e = a.exp()

    g.backward()
    print(f"Backend: {backend}, Time Taken: {time.time() - start} seconds")
