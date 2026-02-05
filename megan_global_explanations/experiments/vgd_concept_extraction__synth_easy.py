"""
Extends the base experiment "vgd_concept_extraction". This experiment implements the concept extraction
specifically for the synth_easy synthetic dataset.

Parameters are tuned to generate relatively few concept clusters by using:
- Higher MIN_CLUSTER_SIZE (requires larger clusters)
- Higher MIN_SAMPLES (more conservative clustering)

Additionally, this experiment creates a CSV file with SMILES and distances to all cluster centroids
for each sample in the dataset via the post_clustering hook.
"""
import os
import pathlib
import typing as t

import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from megan_global_explanations.utils import EXPERIMENTS_PATH


PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(EXPERIMENTS_PATH, 'assets')

# == DATASET PARAMETERS ==
# The parameters determine the details related to the dataset that should be used as the basis
# of the concept extraction

# :param VISUAL_GRAPH_DATASETS:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'synth_easy'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or
#       "classification"
DATASET_TYPE: str = 'regression'
# :param CHANNEL_INFOS:
#       This dictionary can optionally be given to supply additional information about the individual
#       explanation channels. The key should be the index of the channel and the value should again be
#       a dictionary that contains the information for the corresponding channel.
CHANNEL_INFOS: t.Dict[int, dict] = {
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
# These parameters determine the details related to the model that should be used for the
# concept extraction. For this experiment, the model should already be trained and only
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'synth_easy.ckpt')


# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.
# Parameters are set to generate FEWER clusters.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a
#       fidelity higher than this will be used as possible candidates for the clustering.
#       Higher value = fewer elements eligible for clustering.
FIDELITY_THRESHOLD: float = 0.5
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
#       Higher value = fewer, larger clusters.
MIN_CLUSTER_SIZE: int = 1000
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while
#       lower values tend to result in more clusters.
#       Higher value = more conservative = fewer clusters.
MIN_SAMPLES: int = 100
# :param CLUSTER_SELECTION_METHOD:
#       This string value determines the method that is used to select the clusters from the HDBSCAN
#       algorithm. 'eom' (Excess of Mass) tends to produce fewer, larger clusters compared to 'leaf'.
CLUSTER_SELECTION_METHOD: str = 'eom'
# :param SORT_SIMILARITY:
#       This boolean flag determines whether the clusters should be sorted by their similarity.
SORT_SIMILARITY: bool = True

# == PROTOTYPE OPTIMIZATION PARAMETERS ==
# These parameters configure the process of optimizing the cluster prototype representatation

# :param OPTIMIZE_CLUSTER_PROTOTYPE:
#       This boolean flag determines whether the prototype optimization should be executed at
#       all or not. If this is False, the entire optimization routine will be skipped during the
#       cluster discovery.
OPTIMIZE_CLUSTER_PROTOTYPE: bool = False
# :param DESCRIBE_PROTOTYPE:
#       This boolean flag determines whether the prototype description should be generated.
DESCRIBE_PROTOTYPE: bool = False
# :param HYPOTHESIZE_PROTOTYPE:
#       This boolean flag determines whether the prototype hypothesis should be generated.
HYPOTHESIZE_PROTOTYPE: bool = False

# == VISUALIZATION PARAMETERS ==
# These parameters determine the details of the visualizations that will be created as part of the
# artifacts of this experiment.

# :param PLOT_UMAP:
#       This boolean flag determines whether the UMAP visualization of the graph embeddings should be
#       created or not.
PLOT_UMAP: bool = False


__DEBUG__ = True

experiment = Experiment.extend(
    'vgd_concept_extraction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('post_clustering', default=False, replace=True)
def post_clustering(e: Experiment,
                    cluster_infos: t.List[dict],
                    graphs: t.List[dict],
                    **kwargs) -> None:
    """
    Creates a CSV file containing SMILES strings and Manhattan distances from each sample's
    graph embedding to all cluster centroids. Distances are computed using the embedding
    channel that corresponds to each cluster.
    """
    e.log('post_clustering: creating embeddings distance CSV...')

    # Build rows for CSV
    rows = []
    for graph in graphs:
        row = {
            'smiles': graph.get('graph_repr', ''),
        }

        # Add ground truth label(s)
        if 'graph_labels' in graph:
            labels = graph['graph_labels']
            if isinstance(labels, list) and len(labels) == 1:
                row['label'] = labels[0]
            else:
                row['label'] = labels

        # Add model prediction
        if 'graph_prediction' in graph:
            row['prediction'] = graph['graph_prediction']

        # Add fidelity per channel
        if 'graph_fidelity' in graph:
            for ch_idx, fid in enumerate(graph['graph_fidelity']):
                row[f'fidelity_ch{ch_idx}'] = fid

        # Compute Manhattan distance to each cluster centroid
        # using the embedding channel that belongs to that cluster
        graph_embedding = np.array(graph['graph_embedding'])

        for cluster_info in cluster_infos:
            ch_idx = cluster_info['channel_index']
            cl_idx = cluster_info['index']
            centroid = cluster_info['centroid']

            # Extract embedding for this channel
            channel_embedding = graph_embedding[:, ch_idx]

            # Compute Manhattan distance
            dist = cityblock(channel_embedding, centroid)
            row[f'dist_ch{ch_idx}_cl{cl_idx}'] = dist

        rows.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(e.path, 'embeddings_distances.csv')
    df.to_csv(csv_path, index=False)
    e.log(f'saved embeddings distances CSV with {len(df)} rows and {len(df.columns)} columns to {csv_path}')


experiment.run_if_main()
