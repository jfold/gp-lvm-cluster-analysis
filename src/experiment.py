from imports.ml import *
from imports.general import *
from src.gp_lvm import GPLVM
from src.dataset import Dataset
from src.parameters import Parameters
from src.figures import Results


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
        self.nmi = nmi(self.model.y, self.y_preds)
