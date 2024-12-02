from cudagrad.tensor import Tensor2D
import numpy as np
import torch
import cupy as cp


def validateSoftmax():
    print("Validating Softmax")
    np.random.seed(0)
    a_base = np.random.rand(100, 100)
    # y is the one hot encoded vector with only only one element set to 1 in each row
    y_base = np.zeros_like(a_base, dtype=np.int64)
    y_base[np.arange(100), np.random.randint(0, 100, 100)] = 1

    a = Tensor2D(a_base, label="a")
    b = a.crossEntropyLoss(cp.array(y_base.argmax(axis=1)))
    b.label = "b"
    b.backward()
    # print("B ", b.data)

    a_t = torch.tensor(a_base, requires_grad=True, dtype=torch.float32)
    b_t = torch.nn.functional.cross_entropy(a_t, torch.tensor(y_base.argmax(axis=1), dtype=torch.long))

    # print("B Torch", b_t)
    b_t.backward()

    # print("A Grad", a.grad[:-10])
    # print("A Torch Grad", a_t.grad[:-10])

    # print("Check A: ", np.array_equal(cp.asnumpy(a.grad), a_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(a.grad), a_t.grad.cpu().numpy()))


def validateNN():
    """Create a simple neural network with two layers of Weights and bias, relu activation and cross entropy loss with a toy dataset and compare all the gradients with PyTorch"""
    np.random.seed(0)
    a_base = cp.array(np.random.rand(100, 100) - 0.5)
    # y is the one hot encoded vector with only only one element set to 1 in each row
    y_base = np.zeros_like(a_base, dtype=np.int64)
    y_base[np.arange(100), np.random.randint(0, 100, 100)] = 1

    # Create a Tensor
    a = Tensor2D(a_base, label="a")
    W1 = Tensor2D(np.random.rand(100, 100) - 0.5, label="W1")
    b1 = Tensor2D(np.random.rand(100) - 0.5, label="b1")
    W2 = Tensor2D(np.random.rand(100, 100) - 0.5, label="W2")
    b2 = Tensor2D(np.random.rand(100) - 0.5, label="b2")

    print("-------------------")
    # Write Equivalent code in PyTorch
    a_t = torch.tensor(a_base, requires_grad=True, dtype=torch.float32)
    W1_t = torch.tensor(W1.data, requires_grad=True, dtype=torch.float32)
    b1_t = torch.tensor(b1.data, requires_grad=True, dtype=torch.float32)
    W2_t = torch.tensor(W2.data, requires_grad=True, dtype=torch.float32)
    b2_t = torch.tensor(b2.data, requires_grad=True, dtype=torch.float32)

    c = a @ W1 + b1
    print("C ", c.data)
    c.label = "c"
    c = c.relu()
    print("C Relu", c.data)
    c.label = "c_relu"
    d = c @ W2 + b2
    print("D ", d.data)
    d.label = "d"
    e = d.cross_entropy_loss(cp.array(y_base.argmax(axis=1)))
    print("E ", e.data)
    e.label = "e"
    e.backward()

    print("E ", e.data)

    c_t = a_t @ W1_t + b1_t
    print("C Torch", c_t)
    c_t = torch.nn.functional.relu(c_t)
    print("C Torch Relu", c_t)
    d_t = c_t @ W2_t + b2_t
    print("D Torch", d_t)
    e_t = torch.nn.functional.cross_entropy(d_t, torch.tensor(y_base.argmax(axis=1), dtype=torch.long))
    print("E Torch", e_t)
    # Compare the outputs of c and c_t

    c_t.retain_grad()
    d_t.retain_grad()
    e_t.backward()

    # Check all the gradients
    print("Check C: ", np.array_equal(cp.asnumpy(c.grad), c_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(c.grad), c_t.grad.cpu().numpy()))
    print("Check W1: ", np.array_equal(cp.asnumpy(W1.grad), W1_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(W1.grad), W1_t.grad.cpu().numpy()))
    print("Check b1: ", np.array_equal(cp.asnumpy(b1.grad), b1_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(b1.grad), b1_t.grad.cpu().numpy()))
    print("Check W2: ", np.array_equal(cp.asnumpy(W2.grad), W2_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(W2.grad), W2_t.grad.cpu().numpy()))
    print("Check b2: ", np.array_equal(cp.asnumpy(b2.grad), b2_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(b2.grad), b2_t.grad.cpu().numpy()))
    print("Check A: ", np.array_equal(cp.asnumpy(a.grad), a_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(a.grad), a_t.grad.cpu().numpy()))


def validate():
    validateNN()


if __name__ == '__main__':
    validate()
