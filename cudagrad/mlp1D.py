import time

from cudagrad.layer1D import Layer
from cudagrad.tensor import Tensor1D


class MLP:
    def __init__(self, n_input, n_output):
        size = [n_input] + n_output
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_output))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def parameters_count(self):
        return sum([p.data.size for layer in self.layers for p in layer.parameters()])

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


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
    ys = Tensor1D([1.0, -1.0, -1.0, 1.0])


    # MSE Loss
    def calculate_loss(y_true, y_pred):
        return ((y_true - y_pred) ** 2).sum()


    n = MLP(3, [4, 4, 1])

    # Training loop
    for iter in range(n_iters):
        start = time.time()
        # Forward pass
        y_pred = [n(x) for x in xs]
        y_pred = Tensor1D.concat(y_pred)
        loss = calculate_loss(ys, y_pred)

        # Backward pass
        n.zero_grad()
        loss.backward()

        for p in n.parameters():
            p.data -= learning_rate * p.grad

        print(f"Iter: {iter}, Loss: {loss}, Time: {time.time() - start}")
    final_preds = Tensor1D.concat([n(x) for x in xs])
    print(f"Final Preds: ", final_preds)
