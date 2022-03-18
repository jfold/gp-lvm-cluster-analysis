import string
from imports.general import *
from imports.ml import *


@dataclass
class Parameters:
    seed: int = 0  # random seed
    dataset_name: str = "default"  # number of input dimensions
    data_dim: int = 2  # number of input dimensions
    latent_dim: int = 2  # number of latent space dimensions
    n_train: int = 10  # number of training samples
    n_test: int = 100  # number of test samples
    n_iterations: int = 10000  # number of training iterations
    gplvm_learning_rate: float = 0.01  # hyperparameter learning rate
    cluster_std: float = None
    plot_it: bool = False  # whether to plot during BO loop
    save_it: bool = True  # whether to save progress
    test: bool = True  # whether to save progress
    gp_latent_init_pca: bool = (
        False  # whether to initialize latent space with PCA solution
    )
    savepth: str = os.getcwd() + "/results/"
    experiment: str = ""  # folder name

    def __init__(self, kwargs: Dict = {}, mkdir: bool = True) -> None:
        self.update(kwargs)
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
        if self.test:
            folder_name = f"test{self.settings_string(kwargs)}"
        else:
            folder_name = (
                f"experiment{self.experiment}---{datetime.now().strftime('%d%m%y-%H%M%S')}-"
                + "".join(random.choice(string.ascii_lowercase) for x in range(4))
            )
        setattr(self, "savepth", f"{self.savepth}{folder_name}/")
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
            self.save()

    def settings_string(self, kwargs: Dict) -> str:
        output = "---"
        for k, v in kwargs.items():
            output += f"{k}-{v}--"
        output = output.rstrip("-")
        output = "---default" if output == "" else output
        return output

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
