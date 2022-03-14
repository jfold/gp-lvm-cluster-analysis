import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, kstest, entropy, pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors as KNNsklearn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from scipy.stats.stats import energy_distance
from scipy.spatial.distance import mahalanobis, cdist
import uncertainty_toolbox as uct  # https://github.com/uncertainty-toolbox/uncertainty-toolbox
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tf.enable_v2_behavior()
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["font.size"] = 18
matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["savefig.bbox"] = "tight"
# plot-settings:
ps = {
    "GP": {"c": "red", "m": "x"},
    "RF": {"c": "blue", "m": "4"},
    "BNN": {"c": "orange", "m": "v"},
    "DS": {"c": "black", "m": "*"},
    "DE": {"c": "mediumorchid", "m": "2"},
    "RS": {"c": "palegreen", "m": "h"},
}
