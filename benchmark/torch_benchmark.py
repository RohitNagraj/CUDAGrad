import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.dataset import Dataset
from benchmark.parameters import Parameters


class _SimpleNN(nn.Module):
    def __init__(self, n_input, n_output):
        super(_SimpleNN, self).__init__()
        size = [n_input] + n_output
        self.layers = nn.ModuleList([nn.Linear(size[i], size[i + 1]) for i in range(len(n_output))])

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx != len(self.layers) - 1:
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        return x


class TorchBenchmark:
    def __init__(self, n_inputs, n_outputs):
        self.model = _SimpleNN(n_inputs, n_outputs)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _accuracy(self, y_true, y_pred):
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y_true, y_pred)]
        return sum(accuracy) / len(accuracy)

    def _train(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Training loop
        epochs = 100  # Number of training epochs
        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass: compute predicted y by passing X to the model
            y_pred = self.model(X_tensor)

            # Compute the loss
            loss = self.criterion(y_pred, y_tensor.reshape(-1, 1))
            accuracy = self._accuracy(y_tensor, y_pred)

            # Backward pass: compute gradients
            loss.backward()

            # Update the weights
            self.optimizer.step()

            print(f"step {epoch + 1},  loss {loss}, "
                  f"accuracy {accuracy * 100}%")

    def run(self, dataset_size):
        dataset = Dataset()
        X, y = dataset.data(size=dataset_size)
        self._train(X, y)


if __name__ == "__main__":
    benchmark = TorchBenchmark(Parameters.N_INPUTS, Parameters.N_OUTPUTS)
    benchmark.run(Parameters.DATASET_SIZE)
