from imports.ml import *
from imports.general import *


class Results(object):
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
            plt.legend()
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
        plt.show()
