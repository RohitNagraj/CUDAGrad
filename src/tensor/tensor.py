import cupy as cp
import cupyx.profiler
import numpy as np
import cupyx
import queue

KERNEL_PATH = "tensor/cuda/"
# src/tensor/cuda/addTensor.cuh
class Tensor2D:
    def initilizeCudaKernel(self):
        with open(f"{KERNEL_PATH}/addTensor.cuh", "r") as f:
            addTensorCode = f.read()
        self.addTensor = cp.RawKernel(addTensorCode, "addTensor")
        self.addTensor.compile()

        self.addBroadcastedTensor = cp.RawKernel(addTensorCode, "addBroadcastedTensor")
        self.addBroadcastedTensor.compile()

    #     with open(f"{KERNEL_PATH}matmulTensor.cuh", "r") as f:
    #         matmulCode = f.read()
        
    #     self.matmulTensor = cp.RawKernel(matmulCode, "multiplyTensor")
    #     self.matmulTensor.compile()
    
    def CEIL_DIV(self, M, N):
        return (((M) + (N)-1) // (N))
    
    def calculateGridAndBlock(self):
        blocks = (self.CEIL_DIV(self.data.shape[0], 32), self.CEIL_DIV(self.data.shape[1], 32))
        threads = (32 * 32,)
        return (blocks, threads)

    def __init__(self, data: list | np.ndarray | cp.ndarray, _children=(), _op="", label="", trackGradient=True) -> None:
        self.data = cp.array(data, dtype=cp.float32)
        self.grad = cp.zeros_like(self.data, dtype=cp.float32)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.trackGradient = trackGradient
        self._backward = lambda: None
        self.initilizeCudaKernel()
    
    def __repr__(self):
        return f"Tensor(label={self.label}, trackGradient={self.trackGradient})"
    
    # Define a .T function to get the Transpose of the Tensor
    def T(self):
        return Tensor2D(self.data.T, label=self.label+"_T", trackGradient=self.trackGradient)
    

    
    def calculateElementwiseGridAndBlock(self):
        numThreads = min(self.data.shape[1], 1024)
        numBlocks = self.CEIL_DIV(self.data.size, numThreads)
        return (numBlocks,), (numThreads,)
    
    def __add__(self, other):
        '''
        Add two tensors elementwise
        '''
        # Broadcast the other tensor to the shape of self tensor


        if not isinstance(other, Tensor2D):
            raise ValueError("Addition is only supported between two tensors")
        
        c = Tensor2D(cp.zeros_like(self.data, dtype=cp.float32), (self, other), "+")

        blocks, threads = self.calculateElementwiseGridAndBlock()
        print(blocks, threads)
        print(type(self.data), type(other.data), type(c.data))
        with cupyx.profiler.profile():
            if self.data.shape != other.data.shape:
                self.addBroadcastedTensor(blocks, threads, (self.data,  other.data, c.data, self.data.size))
            else:
                self.addTensor(blocks, threads, (self.data, other.data, c.data, self.data.size))

            # c.data = self.data + other.data
        
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
        # blocks, threads = self.calculateGridAndBlock()
        # print(blocks, threads)
        with cupyx.profiler.profile():
            # self.matmulTensor(blocks, threads, (self.data, other.data, c.data, self.data.shape[0], self.data.shape[1], other.data.shape[1]))
            c.data = self.data @ other.data
        
        def _backward():
            self.grad += 1.0 * c.grad @ other.data.T
            other.grad += self.data.T @ c.grad
        
        if self.trackGradient:
            c._backward = _backward

        return c
    
    def log(self):
        c = Tensor2D(cp.log(self.data), (self,), "log("+self.label+")")
        def _backward():
            self.grad += 1.0 * c.grad / self.data
        if self.trackGradient:
            c._backward = _backward
        return c
    
    def exp(self):
        c = Tensor2D(cp.exp(self.data), (self,), "exp("+self.label+")")
        def _backward():
            self.grad += 1.0 * c.grad * c.data
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


    def sum(self, axis=None, keepdims=False):
        c = Tensor2D(cp.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum("+self.label+")")
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
    
    def crossEntropyLoss(self, y):
        '''self.data is the logits and y is a vetor of correct labels'''
        c = Tensor2D(cp.zeros_like(self.data), (self, self), "crossEntropyLoss("+self.label+")")

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
        c = Tensor2D(cp.maximum(0, self.data), (self,), "relu("+self.label+")")
        def _backward():
            self.grad += (c.data > 0) * c.grad
        if self.trackGradient:
            c._backward = _backward
        return c

    
    def backward(self):
        # Run the Backward Function of all the children
        # First construct the topological order of the graph with structure
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
        print("Order of Computation: ", topo[::-1])

        for v in reversed(topo):
            v._backward()

    
