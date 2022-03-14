import json
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict


@dataclass
class Parameters:
    seed: int = 0  # random seed
    data_dim: int = 1  # number of input dimensions
    latent_dim: int = 1  # number of latent space dimensions
    n_train: int = 100  # number of training samples
    n_test: int = 3000  # number of test samples
    n_iterations: int = 100  # number of training iterations
    gplvm_learning_rate: float = 0.001  # hyperparameter learning rate
    plot_it: bool = False  # whether to plot during BO loop
    save_it: bool = True  # whether to save progress
    gp_latent_init_pca: bool = True  # whether to initialize latent space with PCA solution
    savepth: str = os.getcwd() + "/results/"
    experiment: str = ""  # folder name

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
        folder_name = f"({self.d}){self.problem}_{self.surrogate}_{self.acquisition}_seed-{self.seed}"
        setattr(
            self, "experiment", folder_name,
        )
        setattr(self, "savepth", self.savepth + self.experiment + "/")
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
            self.save()

    def update(self, kwargs, save=False) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found")
        if save:
            self.save()

    def save(self) -> None:
        json_dump = json.dumps(asdict(self))
        with open(self.savepth + "parameters.json", "w") as f:
            f.write(json_dump)