import time

from micrograd.nn import MLP as MicrogradMLP
from micrograd.engine import Value
import numpy as np

from data.dataset import Dataset
from benchmark.parameters import Parameters

import sys
sys.setrecursionlimit(1024 * 16)


class MicrogradBenchmark:
    def __init__(self, n_input: int, n_outputs: list):
        self.model = MicrogradMLP(n_input, n_outputs)
        print("No. of parameters in Micrograd: ", len(self.model.parameters()))

    def _loss(self, y_true, y_pred):
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(y_true, y_pred)]
        data_loss = sum(losses) * (1.0 / len(losses))
        return data_loss

    def _accuracy(self, y_true, y_pred):
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y_true, y_pred)]
        return sum(accuracy) / len(accuracy)

    def _train(self, X: np.array, y: np.array, batch_size=256, n_iter=20):
        print("Starting micrograd training. This will take some time...")
        for iter in range(n_iter):
            start = time.time()

            batch_loss = []
            batch_accuracy = []

            for i in range(0, y.shape[0], batch_size):
                start_idx = i
                end_idx = min(i + batch_size, y.shape[0])

                Xb, yb = X[start_idx:end_idx], y[start_idx:end_idx]

                inputs = [list(map(Value, xrow)) for xrow in Xb]

                # forward pass
                scores = list(map(self.model, inputs))

                loss = self._loss(yb, scores)
                accuracy = self._accuracy(yb, scores)

                # backward
                self.model.zero_grad()
                loss.backward()

                batch_loss.append(loss)
                batch_accuracy.append(accuracy)

                # update (sgd)
                learning_rate = 1.0 - 0.9 * iter / 100
                for p in self.model.parameters():
                    p.data -= learning_rate * p.grad

            print(f"step {iter + 1},  loss {(sum(batch_loss) / len(batch_loss)).data}, "
                  f"accuracy {sum(batch_accuracy) / len(batch_accuracy) * 100}%")
            # print(f"Time taken {(time.time() - start)} seconds")

    def run(self, dataset_size):
        dataset = Dataset()
        X, y = dataset.data(size=dataset_size)
        self._train(X, y)


if __name__ == '__main__':
    benchmark = MicrogradBenchmark(Parameters.N_INPUTS, Parameters.N_OUTPUTS)
    benchmark.run(Parameters.DATASET_SIZE)
