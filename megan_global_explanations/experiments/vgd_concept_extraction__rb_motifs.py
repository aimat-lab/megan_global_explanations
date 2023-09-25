"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.training import NoLoss

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True
__TESTING__ = False

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
SUBSET: t.Optional[int] = None
NUM_TEST: int = 1000
NUM_TARGETS: int = 1
CHANNEL_DETAILS: dict = {
    0: {
        'name': 'negative',
        'color': 'lightskyblue',
    },
    1: {
        'name': 'positive',
        'color': 'lightcoral',
    }
}

# == MODEL PARAMETERS == 
UNITS = [32, 32, 32]
EMBEDDING_UNITS = [64, 32, 16]
IMPORTANCE_CHANNELS = 2
IMPORTANCE_FACTOR = 1.0
IMPORTANCE_MULTIPLIER = 0.5
REGRESSION_REFERENCE = 0.0
REGRESSION_WEIGHTS = [1.0, 1.0]
FINAL_UNITS = [16, 1]
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 1.0
FIDELITY_FACTOR = 0.1
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]

PACKING_EPOCHS_WARMUP: int = 50
PACKING_MIN_SAMPLES: int = 20
PACKING_FACTOR: float = 1.0
PACKING_BATCH_SIZE: int = 6000

# == TRAINING PARAMETERS ==
LOSS_CB = lambda: [ks.losses.MeanSquaredError(), NoLoss(), NoLoss()]
OPTIMIZER_CB = lambda: ks.optimizers.Adam(
    learning_rate=0.001,
)
EPOCHS = 100
BATCH_SIZE = 16

# == CONCEPT EXTRACTION ==
FIDELITY_THRESHOLD: float = 0.5

REMOVE_OUTLIERS: bool = True
OUTLIER_FACTOR: float = 0.01

UMAP_NUM_NEIGHBORS: int = 300
UMAP_MIN_DIST: float = 0.0
UMAP_METRIC: str = 'cosine'
UMAP_REPULSION_STRENGTH: float = 1.0

HDBSCAN_MIN_SAMPLES: int = 50
HDBSCAN_MIN_CLUSTER_SIZE: int = 100
HDBSCAN_MIN_METHOD: str = 'eom'

experiment = Experiment.extend(
    'vgd_concept_extraction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()