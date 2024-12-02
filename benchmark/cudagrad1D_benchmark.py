import numpy as np

from benchmark.parameters import Parameters
from cudagrad.mlp1D import MLP
from cudagrad.tensor import Tensor1D
from data.dataset import Dataset


class CUDAGrad1DBenchmark:
    def __init__(self, n_input: int, n_outputs: list):
        self.model = MLP(n_input, n_outputs)
        print(self.model)
        print("No. of Tensors in CUDAGrad", len(self.model.parameters()))

    def _loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).sum()

    def _accuracy(self, y_true, y_pred):
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y_true, y_pred)]
        return sum(accuracy) / len(accuracy)

    def _train(self, X: np.array, y: np.array, batch_size=256, n_iter=100, lr=1e-1):

        for iter in range(n_iter):
            batch_loss = []
            batch_accuracy = []

            for i in range(0, y.shape[0], batch_size):
                start_idx = i
                end_idx = min(i + batch_size, y.shape[0])

                Xb, yb = X[start_idx:end_idx], y[start_idx:end_idx]

                # Forward pass
                y_pred = list(map(self.model, Xb))
                y_pred = Tensor1D.concat(y_pred)

                accuracy = self._accuracy(yb, y_pred)
                loss = self._loss(yb, y_pred)

                # Backward pass
                self.model.zero_grad()
                loss.backward()

                batch_loss.append(loss)
                batch_accuracy.append(accuracy)

                # SGD
                for p in self.model.parameters():
                    p.data -= lr * p.grad

            print(f"step {iter + 1}, loss {sum(batch_loss) / len(batch_loss)}, "
                  f"accuracy {sum(batch_accuracy) / len(batch_accuracy) * 100}%")

    def run(self, dataset_size):
        dataset = Dataset()
        X, y = dataset.data(size=dataset_size)
        self._train(X, y)


if __name__ == "__main__":
    benchmark = CUDAGrad1DBenchmark(Parameters.N_INPUTS, Parameters.N_OUTPUTS)
    benchmark.run(Parameters.DATASET_SIZE)
