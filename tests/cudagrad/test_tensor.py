import torch
import pytest

from cudagrad.tensor import Tensor1D


class TestTensor1D:
    def setup_method(self):
        """Setup the tensors for each test."""
        self.data1 = [1, 2, 3, 4, 5, 6]
        self.data2 = [12, 14, 16, 18, 20, 22]
        self.ones = [1, 1, 1, 1, 1, 1]
        self.backend = 'numpy'
        self.backend = 'cuda'
        self.cudagrad_tensor1 = Tensor1D(self.data1, backend=self.backend)
        self.cudagrad_tensor2 = Tensor1D(self.data2, backend=self.backend)
        self.torch_tensor1 = torch.tensor(self.data1, dtype=torch.float32, requires_grad=True)
        self.torch_tensor2 = torch.tensor(self.data2, dtype=torch.float32, requires_grad=True)

    def test_add(self):
        cudagrad_result = (self.cudagrad_tensor1 + self.cudagrad_tensor2).dot(self.ones)
        torch_result = (self.torch_tensor1 + self.torch_tensor2).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert bool(cudagrad_result.data == torch_result.data)
        assert all(self.torch_tensor1.grad == self.cudagrad_tensor1.grad)
        assert all(self.torch_tensor2.grad == self.cudagrad_tensor2.grad)

    def test_sub(self):
        cudagrad_result = (self.cudagrad_tensor1 - self.cudagrad_tensor2).dot(self.ones)
        torch_result = (self.torch_tensor1 - self.torch_tensor2).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert bool(cudagrad_result.data == torch_result.data)
        assert all(self.torch_tensor1.grad == self.cudagrad_tensor1.grad)
        assert all(self.torch_tensor2.grad == self.cudagrad_tensor2.grad)

    def test_mul(self):
        cudagrad_result = (self.cudagrad_tensor1 * self.cudagrad_tensor2).dot(self.ones)
        torch_result = (self.torch_tensor1 * self.torch_tensor2).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert bool(cudagrad_result.data == torch_result.data)
        assert all(self.torch_tensor1.grad == self.cudagrad_tensor1.grad)
        assert all(self.torch_tensor2.grad == self.cudagrad_tensor2.grad)

    def test_pow(self):
        cudagrad_result = (self.cudagrad_tensor1 ** 3).dot(self.ones)
        torch_result = (self.torch_tensor1 ** 3).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert bool(cudagrad_result.data == torch_result.data)
        assert all(self.torch_tensor1.grad == self.cudagrad_tensor1.grad)

    def test_dot(self):
        cudagrad_result = self.cudagrad_tensor1.dot(self.cudagrad_tensor2)
        torch_result = self.torch_tensor1.dot(self.torch_tensor2)

        cudagrad_result.backward()
        torch_result.backward()

        assert bool(cudagrad_result.data == torch_result.data)
        assert all(self.torch_tensor1.grad == self.cudagrad_tensor1.grad)
        assert all(self.torch_tensor2.grad == self.cudagrad_tensor2.grad)

    def test_log(self):
        cudagrad_result = (self.cudagrad_tensor1.log()).dot(self.ones)
        torch_result = (self.torch_tensor1.log()).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert bool(cudagrad_result.data == torch_result.data)
        assert all(self.torch_tensor1.grad == self.cudagrad_tensor1.grad)

    def test_exp(self):
        cudagrad_result = (self.cudagrad_tensor1.exp()).dot(self.ones)
        torch_result = (self.torch_tensor1.exp()).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert torch.allclose(torch.tensor(cudagrad_result.data), torch_result.data)
        assert torch.allclose(torch.tensor(self.cudagrad_tensor1.grad), self.torch_tensor1.grad)

    def test_truediv(self):
        cudagrad_result = (self.cudagrad_tensor2 / self.cudagrad_tensor1).dot(self.ones)
        torch_result = (self.torch_tensor2 / self.torch_tensor1).dot(torch.tensor(self.ones, dtype=torch.float32))

        cudagrad_result.backward()
        torch_result.backward()

        assert torch.allclose(torch.tensor(cudagrad_result.data), torch_result.data)
        assert torch.allclose(torch.tensor(self.cudagrad_tensor1.grad), self.torch_tensor1.grad)
        assert torch.allclose(torch.tensor(self.cudagrad_tensor2.grad), self.torch_tensor2.grad)
