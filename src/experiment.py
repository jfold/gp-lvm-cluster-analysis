from imports.ml import *
from imports.general import *
from src.gp_lvm import GPLVM
from src.dataset import Dataset
from .parameters import Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.dataset = Dataset(parameters)
        self.model = GPLVM(parameters)

    def run(self):
        self.model.fit(self.dataset)
        self.Z = self.model.z_final
        self.k_means = KMeans(n_clusters=2, n_init=20).fit(self.Z)
        self.y_preds = self.k_means.predict(self.Z)
        self.nmi = nmi(self.model.y, self.y_preds)

    def plot_learning_curve(self):
        plt.figure(figsize=(4, 4))
        plt.plot(self.model.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    def plot_latent_space(self, cluster_centroids: np.ndarray = None):
        # Plot the latent locations before and after training
        plt.figure(figsize=(4, 4))
        plt.title("Before training")
        plt.grid(False)
        plt.scatter(
            x=self.model.z_init[:, 0],
            y=self.model.z_init[:, 1],
            c=self.model.y,
            cmap=plt.get_cmap("Paired"),
            s=50,
            alpha=0.4,
        )
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
        plt.show()

        plt.figure(figsize=(4, 4))
        plt.title("After training")
        plt.grid(False)
        plt.scatter(
            x=self.model.z_final[:, 0],
            y=self.model.z_final[:, 1],
            c=self.model.y,
            cmap=plt.get_cmap("Paired"),
            s=50,
            alpha=0.4,
        )
        if cluster_centroids is not None:
            plt.plot(
                cluster_centroids[:, 0], cluster_centroids[:, 1], "x", label="centroid"
            )
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
        plt.legend()
        plt.show()
