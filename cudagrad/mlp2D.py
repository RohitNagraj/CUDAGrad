import time

from cudagrad.layer2D import Layer
from cudagrad.tensor import Tensor2D


class MLP:
    def __init__(self, n_input, n_output):
        size = [n_input] + n_output
        # Understand the below line
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_output))]

    def __call__(self, x: Tensor2D) -> Tensor2D:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


if __name__ == '__main__':
    learning_rate = 0.1
    n_iters = 5
    # Sample dataset
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = Tensor2D([1.0, -1.0, -1.0, 1.0])
    ys.reshape_inplace(-1, 1)


    # MSE Loss
    def calculate_loss(y_true, y_pred):
        return ((y_true - y_pred) ** 2).sum()


    mlp = MLP(3, [4, 4, 1])

    # Training loop
    for iter in range(n_iters):
        start = time.time()
        # Forward pass
        y_pred = mlp(Tensor2D(xs))
        loss = calculate_loss(ys, y_pred)

        # Backward pass
        mlp.zero_grad()
        loss.backward()

        # Parameter update
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad

        # if iter % 10 == 0:
        print(f"Iter: {iter}, Loss: {loss.data}, Time: {time.time() - start}")
