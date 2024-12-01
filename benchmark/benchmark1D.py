from micrograd.nn import MLP as MicrogradMLP
from micrograd.engine import Value
import numpy as np
import pandas as pd

from cudagrad.mlp import MLP as CUDAGradMLP
from cudagrad.tensor import Tensor1D


class Dataset:
    def __init__(self, path: str):
        """
        Dataset Description:
        The first column is the class label (1 for signal, 0 for background), followed by the 28 features (21 low-level
         features then 7 high-level features): lepton  pT, lepton  eta, lepton  phi, missing energy magnitude, missing
         energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt,
          jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv,
          m_bb, m_wbb, m_wwbb. For more detailed information about each feature see the original paper.
        :param path:
        """
        self.path = path

    def data(self, size: int = 1024):
        df = pd.read_csv(self.path, nrows=size, header=None)
        X = df.drop(columns=[0]).to_numpy()
        y = df[0].to_numpy()
        return X, y


class MicrogradNN:
    def __init__(self, n_input: int, n_outputs: list):
        self.model = MicrogradMLP(n_input, n_outputs)
        print(self.model)
        print("number of parameters", len(self.model.parameters()))

    def _loss(self, y_true, y_pred):
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(y_true, y_pred)]
        data_loss = sum(losses) * (1.0 / len(losses))
        return data_loss

    def _accuracy(self, y_true, y_pred):
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y_true, y_pred)]
        return sum(accuracy) / len(accuracy)

    def train(self, X: np.array, y: np.array, batch_size=256, n_iter=100):
        for iter in range(n_iter):

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

            print(f"step {iter} loss {sum(batch_loss) / len(batch_loss)}, "
                  f"accuracy {sum(batch_accuracy) / len(batch_accuracy) * 100}%")


class CUDAGradNN1D:
    def __init__(self, n_input: int, n_outputs: list):
        self.model = CUDAGradMLP(n_input, n_outputs)
        print(self.model)
        print("number of parameters", len(self.model.parameters()))

    def _loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).sum()

    def _accuracy(self, y_true, y_pred):
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y_true, y_pred)]
        return sum(accuracy) / len(accuracy)

    def train(self, X: np.array, y: np.array, batch_size=256, n_iter=100, lr=1e-1):

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

            print(f"step {iter} loss {sum(batch_loss) / len(batch_loss)}, "
                  f"accuracy {sum(batch_accuracy) / len(batch_accuracy) * 100}%")


def run_benchmark():
    N_INPUTS = 28  # No. of features in the dataset
    N_OUTPUTS = [16, 16, 1]

    dataset = Dataset("data/higgs/higgs.csv")
    X, y = dataset.data(size=2048)

    mlp_micrograd = MicrogradNN(n_input=N_INPUTS, n_outputs=N_OUTPUTS)
    mlp_cudagrad = CUDAGradNN1D(n_input=N_INPUTS, n_outputs=N_OUTPUTS)

    # mlp_micrograd.train(X, y)
    mlp_cudagrad.train(X, y)


if __name__ == '__main__':
    run_benchmark()
