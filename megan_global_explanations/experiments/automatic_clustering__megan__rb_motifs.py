"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

import numpy as np

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan
from graph_attention_student.keras import load_model
from graph_attention_student.util import array_normalize

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
DATASET_TYPE: str = 'regression' # 'classification'
NUM_CHANNELS: int = 2

# == MODEL PARAMETERS ==
MODEL_PATH: str = os.path.join(PATH, 'assets', 'predictors', '_rb_dual_motifs')
FIDELITY_THRESHOLD: float = 0.3

# == MAPPER PARAMETERS ==
UMAP_NUM_NEIGHBORS: int = 500
UMAP_MIN_DIST: float = 0.00
UMAP_METRIC: str = 'cosine'
UMAP_REPULSION_STRENGTH: float = 1.0

# == CLUSTERING PARAMETERS ==
USE_LABEL_PROPAGATION: bool = True

HDBSCAN_MIN_CLUSTER_SIZE: int = 100
HDBSCAN_MIN_SAMPLES: int = 100

OPTICS_MIN_SAMPLES: int = 200
OPTICS_MAX_EPS: float = np.inf
OPTICS_METRIC: str = 'cosine'
OPTICS_MIN_CLUSTER_SIZE: int = 50
OPTICS_XI: float = 0.0

KMEANS_NUM_CLUSTERS: int = 6
KMEANS_MAX_ITER: int = 300

experiment = Experiment.extend(
    'automatic_clustering__megan.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()