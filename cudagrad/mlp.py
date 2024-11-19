from cudagrad.layer import Layer


class MLP:
    def __init__(self, n_input, n_output):
        size = [n_input] + n_output
        # Understand the below line
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_output))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    learning_rate = 0.01
    n_iters = 200
    # Sample dataset
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]


    # MSE Loss
    def calculate_loss(y_true, y_pred):
        return sum((y - yp) ** 2 for y, yp in zip(y_true, y_pred))


    n = MLP(3, [4, 4, 1])

    # Training loop
    for iter in range(n_iters):
        # Forward pass
        y_pred = [n(x) for x in xs]
        loss = calculate_loss(ys, y_pred)

        # Backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # Parameter update
        for p in n.parameters():
            p.data -= learning_rate * p.grad

        if iter % 50 == 0:
            print(f"Iter: {iter}, Loss: {loss}")
    final_preds = [n(x) for x in xs]
    print(f"Final Preds: ", final_preds)
