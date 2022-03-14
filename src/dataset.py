from imports.ml import *
from imports.general import *
from parameters import Parameters


class Dataset(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.generate()

    def generate(self):
        if self.dataset_name == "make_blobs":
            self.X, self.y = make_blobs(
                n_samples=self.n_train,
                n_features=self.data_dim,
                centers=2,
                cluster_std=1 / (10 * self.data_dim,),
            )
            self.X = self.X.transpose()
        else:
            # Load the MNIST data set and isolate a subset of it.
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            self.X = x_train[: self.n_train, ...].astype(np.float64) / 256.0
            self.y = y_train[: self.n_train]
            self.X = self.X.reshape(self.n_train, -1).transpose()
