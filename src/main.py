from tensor.tensor import Tensor
import numpy as np
import torch
import cupy as cp

# Runner File to test the Tensor Class

if __name__ == '__main__':
    np.random.seed(0)
    a_base = np.random.rand(100, 100)
    b_base = np.random.rand(100, 100)
    d_base = np.random.rand(100, 100)
    grad_base = np.random.rand(100, 100)
    

    # Create a Tensor
    a = Tensor(a_base, label="a")
    b = Tensor(b_base, label="b")
    d = Tensor(d_base, label="d")
    c = a + b
    c.label = "c"
    e = c @ d
    e.label = "e"
    f = e.sum()
    f.label = "f"
    f.grad = 0.2321
    f.backward()

    # Write Equivalent code in PyTorch
    a_t = torch.tensor(a_base, requires_grad=True, dtype=torch.float32)
    b_t = torch.tensor(b_base, requires_grad=True, dtype=torch.float32)
    c_t = a_t + b_t
    d_t = torch.tensor(d_base, requires_grad=True, dtype=torch.float32)
    e_t = c_t @ d_t

    c_t.retain_grad()
    e_t.retain_grad()
    f_t = e_t.sum()
    f_t.retain_grad()
    f_t.backward(torch.tensor(0.2321))


    # Check all the gradients
    print("Check C: ", np.array_equal(cp.asnumpy(c.grad), c_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(c.grad), c_t.grad.cpu().numpy()))
    print("Check B: ", np.array_equal(cp.asnumpy(b.grad), b_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(b.grad), b_t.grad.cpu().numpy()))
    print("Check A: ", np.array_equal(cp.asnumpy(a.grad), a_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(a.grad), a_t.grad.cpu().numpy()))
    print("Check D: ", np.array_equal(cp.asnumpy(d.grad), d_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(d.grad), d_t.grad.cpu().numpy()))
    print("Check E: ", np.array_equal(cp.asnumpy(e.grad), e_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(e.grad), e_t.grad.cpu().numpy()))
    print("Check F: ", np.array_equal(cp.asnumpy(f.grad), f_t.grad.cpu().numpy()))
    print("Aproximate Check: ", np.allclose(cp.asnumpy(f.grad), f_t.grad.cpu().numpy()))
