import time

import numpy as np
import cupy as cp

from benchmark.parameters import Parameters
from cudagrad.mlp2D import MLP
from cudagrad.tensor import Tensor2D
from data.dataset import Dataset


class CUDAGrad2DBenchmark:
    def __init__(self, n_input: int, n_outputs: list):
        self.model = MLP(n_input, n_outputs)
        print(self.model)
        print("No. of 2D Tensors in CUDAGrad", len(self.model.parameters()))

    def _loss(self, y_true, y_pred):
        z = -y_true * y_pred
        z = z + Tensor2D(cp.ones_like(z.data))
        z = z.relu()
        z = z.sum() / Tensor2D(cp.array(len(z.data)))
        return z

    def _accuracy(self, y_true, y_pred):
        accuracy = [(yi > 0) == (scorei > 0) for yi, scorei in zip(y_true.data, y_pred.data)]
        return (sum(accuracy) / len(accuracy))[0]

    def _train(self, X: np.array, y: np.array, batch_size=256, n_iter=1, lr=1e-1):

        for iter in range(n_iter):
            start = time.time()
            batch_loss = []
            batch_accuracy = []

            for i in range(0, y.shape[0], batch_size):
                start_idx = i
                end_idx = min(i + batch_size, y.shape[0])

                Xb, yb = Tensor2D(X[start_idx:end_idx]), Tensor2D(y[start_idx:end_idx])
                yb.reshape_inplace(-1, 1)

                # Forward pass
                y_pred = self.model(Xb)

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

            print(f"step {iter + 1}, loss {sum([x.data for x in batch_loss]) / len(batch_loss)}, "
                  f"accuracy {sum(batch_accuracy) / len(batch_accuracy) * 100}%")
            print(f"Time taken: {time.time() - start}")

    def run(self, dataset_size):
        dataset = Dataset()
        X, y = dataset.data(size=dataset_size)
        self._train(X, y)


if __name__ == "__main__":
    benchmark = CUDAGrad2DBenchmark(Parameters.N_INPUTS, Parameters.N_OUTPUTS)
    benchmark.run(Parameters.DATASET_SIZE)
