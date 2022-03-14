import unittest
from main import *


class MainTest(unittest.TestCase):
    def main_test(self):
        kwargs = {
            "savepth": os.getcwd() + "/results/tests/",
            "seed": 0,
            "latent_dim": 2,
        }
        parameters = Parameters(kwargs, mkdir=True)
        experiment = Experiment(parameters)
        experiment.run()
