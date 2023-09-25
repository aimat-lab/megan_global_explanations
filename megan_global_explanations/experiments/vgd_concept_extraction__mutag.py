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
VISUAL_GRAPH_DATASET_PATH = '/media/ssd/.visual_graph_datasets/datasets/mutag'
DATASET_TYPE: str = 'classification'
SUBSET: t.Optional[int] = None
NUM_TEST: int = 500
NUM_TARGETS: int = 2
CHANNEL_DETAILS: dict = {
    0: {
        'name': 'non-mutagenic',
        'color': 'khaki',
    },
    1: {
        'name': 'mutagenic',
        'color': 'plum',
    }
}

# == MODEL PARAMETERS == 
UNITS = [32, 32, 32]
EMBEDDING_UNITS = [64, 96, 128]
IMPORTANCE_CHANNELS = 2
IMPORTANCE_FACTOR = 1.0
IMPORTANCE_MULTIPLIER = 1.0
REGRESSION_REFERENCE = None
REGRESSION_WEIGHTS = None
FINAL_UNITS = [16, 2]
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 1.0
FIDELITY_FACTOR = 0.2
# FIDELITY_FUNCS = [
#     lambda org, mod: tf.nn.relu(mod - org),
#     lambda org, mod: tf.nn.relu(org - mod),
# ]
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(-(org[:, 1] - mod[:, 1])) + tf.abs(org[:, 0] - mod[:, 0]) + 0.1 * tf.reduce_mean(tf.abs(org - mod), axis=-1),
    lambda org, mod: tf.nn.relu(-(org[:, 0] - mod[:, 0])) + tf.abs(org[:, 1] - mod[:, 1]) + 0.1 * tf.reduce_mean(tf.abs(org - mod), axis=-1),
    # lambda org, mod: tf.nn.relu(-(org[:, 1] - mod[:, 1])) + tf.square(org[:, 0] - mod[:, 0]),
    # lambda org, mod: tf.nn.relu(-(org[:, 0] - mod[:, 0])) + tf.square(org[:, 1] - mod[:, 1]),
]

PACKING_EPOCHS_WARMUP: int = 100
PACKING_MIN_SAMPLES: int = 10
PACKING_FACTOR: float = 1.0
PACKING_BATCH_SIZE: int = 5000

# == TRAINING PARAMETERS ==
LOSS_CB = lambda: [ks.losses.CategoricalCrossentropy(from_logits=True), NoLoss(), NoLoss()]
OPTIMIZER_CB = lambda: ks.optimizers.Adam(learning_rate=0.0001)
EPOCHS = 200
BATCH_SIZE = 32

# == CONCEPT EXTRACTION ==
FIDELITY_THRESHOLD: float = 0.5

REMOVE_OUTLIERS: bool = False
OUTLIER_FACTOR: float = 0.1

UMAP_NUM_NEIGHBORS: int = 300
UMAP_MIN_DIST: float = 0.0
UMAP_METRIC: str = 'cosine'
#UMAP_METRIC: str = 'manhattan'
UMAP_REPULSION_STRENGTH: float = 1.0

HDBSCAN_MIN_SAMPLES: int = 20
HDBSCAN_MIN_CLUSTER_SIZE: int = 20
HDBSCAN_METHOD: str = 'eom'

experiment = Experiment.extend(
    'vgd_concept_extraction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()