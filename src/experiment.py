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
        self.plot_learning_curve()

    def plot_learning_curve(self):
        plt.figure(figsize=(4, 4))
        plt.plot(self.model.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    # def plot_gplvm_latents(self):
    #     plt.figure(figsize=(4, 4))
    #     plt.title("Before training")
    #     plt.grid(False)
    #     plt.scatter(
    #         x=init_[:, 0],
    #         y=init_[:, 1],
    #         c=small_y_train,
    #         cmap=plt.get_cmap("Paired"),
    #         s=50,
    #     )
    #     plt.show()

    #     plt.figure(figsize=(4, 4))
    #     plt.title("After training")
    #     plt.grid(False)
    #     plt.scatter(
    #         x=final_lips[-1, :, 0],
    #         y=final_lips[-1, :, 1],
    #         c=small_y_train,
    #         cmap=plt.get_cmap("Paired"),
    #         s=50,
    #     )
    #     plt.show()
