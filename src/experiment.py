from imports.ml import *
from imports.general import *
from src.gp_lvm import GPLVM
from src.dataset import Dataset
from src.parameters import Parameters
from postprocessing.figures import Results


class Experiment(Results):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.dataset = Dataset(parameters)
        self.model = GPLVM(parameters)

    def run(self):
        self.model.fit(self.dataset)
        self.Z = self.model.z_final
        self.k_means = KMeans(n_clusters=2, n_init=20).fit(self.Z)
        self.y_preds = self.k_means.predict(self.Z)
        self.nmi = nmi(self.model.y.astype(int), self.y_preds.astype(int))
        self.summary = {
            "nmi": self.nmi,
            "Z_final": self.Z.tolist(),
            "loss_history": self.model.loss_history.tolist(),
        }
        self.save_summary()

    def save_summary(self) -> None:
        json_dump = json.dumps(self.summary)
        with open(self.savepth + f"results.json", "w") as f:
            f.write(json_dump)
