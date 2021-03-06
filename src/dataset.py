from imports.ml import *
from imports.general import *
from src.parameters import Parameters


class Dataset(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
        self.generate()

    def generate(self):
        if self.dataset_name == "default":
            c_1 = np.ones((int(self.n_train / 2), self.data_dim)) / np.sqrt(
                self.data_dim
            )
            c_2 = -np.ones((int(self.n_train / 2), self.data_dim)) / np.sqrt(
                self.data_dim
            )
            self.X = np.append(c_1, c_2, axis=0)
            self.y = np.append(
                np.zeros((int(self.n_train / 2),)),
                np.ones((int(self.n_train / 2),)),
                axis=0,
            )
            if self.cluster_std is None:
                self.cluster_std = 1
            self.X = self.X + np.random.normal(
                0, self.cluster_std ** 2, size=self.X.shape
            )
            self.X = self.X.transpose()
        elif self.dataset_name == "make_blobs":
            if self.cluster_std is None:
                self.cluster_std = 1 / (10 * self.data_dim)
            self.X, self.y = make_blobs(
                n_samples=self.n_train,
                n_features=self.data_dim,
                centers=2,
                cluster_std=self.cluster_std,
            )
            self.X = self.X - np.mean(self.X, axis=0)
            self.X = self.X.transpose()
        else:
            # Load the MNIST data set and isolate a subset of it.
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            self.X = x_train[: self.n_train, ...].astype(np.float64) / 256.0
            self.y = y_train[: self.n_train]
            self.X = self.X.reshape(self.n_train, -1).transpose()
