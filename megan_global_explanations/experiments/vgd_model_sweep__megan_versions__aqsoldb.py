"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import tensorflow.keras as ks
from umap import UMAP
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan
from graph_attention_student.models.megan import Megan2
from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.data import mock_importances_from_graphs
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.fidelity import leave_one_out_analysis

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
DATASET_TYPE: int = 'regression'  # classification
NUM_TARGETS: int = 1
NUM_TEST: int = 1000

# == SWEEP PARAMETERS ==
SWEEP_KEYS = ['megan_2', 'megan_1']
REPETITIONS = 1
DEVICE = 'cpu:0'

# == MEGAN MODEL PARAMETERS ==
UNITS = [64, 64, 64]
IMPORTANCE_CHANNELS = 2
IMPORTANCE_FACTOR = 1.0
IMPORTANCE_MULTIPLIER = 0.5
REGRESSION_REFERENCE = -3.0
REGRESSION_WEIGHTS = [1.0, 1.0]
FINAL_UNITS = [64, 32, 16, 1]
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 0.2
FIDELITY_FACTOR = 0.2
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
CONTRASTIVE_SAMPLING_FACTOR = 1.0
CONTRASTIVE_SAMPLING_TAU = 0.9
POSITIVE_SAMPLING_RATE = 5

LOSS_CB = lambda: [ks.losses.MeanSquaredError(), NoLoss(), NoLoss()]
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(
    learning_rate=0.001,
    weight_decay=0.01,
)
EPOCHS = 50
BATCH_SIZE = 8

# == UMAP MAPPER PARAMETERS ==
UMAP_METRIC: str = 'cosine'
UMAP_MIN_DIST: float = 0.0
UMAP_NUM_NEIGHBORS: int = 75
UMAP_REPULSION_STRENGTH: float = 1.0

# == EVALUATION PARAMETERS ==
VALUE_RANGE: tuple = (-12, +4)
NUM_BINS: int = 50

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_model_sweep__megan_versions.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


experiment.run_if_main()