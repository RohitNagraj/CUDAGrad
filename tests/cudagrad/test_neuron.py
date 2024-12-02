import numpy as np
import torch
import pytest

from cudagrad.neuron1D import Neuron
from cudagrad.tensor import Tensor1D


class TestNeuron:
    def setup_method(self):
        self.size = 5
        self.w = np.array([0.37669507, -1.16691716, 1.39108197, -0.81720608, 1.12384145])
        self.x = np.array([0.12746163, -1.65847459, -0.60745209, 1.32508392, 0.55519409])
        self.b = np.array([-0.07618158])

        self.cudagrad_neuron = Neuron(5)
        self.cudagrad_neuron.w = Tensor1D(self.w)
        self.cudagrad_neuron.b = Tensor1D(self.b)
        self.cudagrad_x = Tensor1D(self.x)

        # Forward pass with torch
        self.torch_w = torch.tensor(self.w, requires_grad=True, dtype=torch.float32)
        self.torch_b = torch.tensor(self.b, requires_grad=True, dtype=torch.float32)
        self.torch_x = torch.tensor(self.x, requires_grad=True, dtype=torch.float32)

    def test_forward(self):
        cudagrad_y = self.cudagrad_neuron(Tensor1D(self.x))

        torch_activation = self.torch_w.dot(self.torch_x) + self.torch_b
        torch_output = torch.tanh(torch_activation)

        assert torch.allclose(torch.tensor(cudagrad_y.data), torch_output.data)

    def test_backward(self):
        cudagrad_y = self.cudagrad_neuron(self.cudagrad_x)
        cudagrad_y.backward()

        torch_activation = self.torch_w.dot(self.torch_x) + self.torch_b
        torch_output = torch.tanh(torch_activation)
        torch_output.backward()

        assert torch.allclose(torch.tensor(self.cudagrad_neuron.w.grad), self.torch_w.grad)
        assert torch.allclose(torch.tensor(self.cudagrad_neuron.b.grad), self.torch_b.grad)
        assert torch.allclose(torch.tensor(self.cudagrad_x.grad), self.torch_x.grad)

