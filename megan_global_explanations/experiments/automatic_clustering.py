"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import random
import typing as t
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
# from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from sklearn.semi_supervised import LabelPropagation
from umap import UMAP
from hdbscan import HDBSCAN

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import VisualGraphDatasetReader, NumericJsonEncoder
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from visual_graph_datasets.visualization.importances import plot_node_importances_border

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
DATASET_TYPE: str = 'regression' # 'classification'
NUM_CHANNELS: int = 2
CHANNEL_COLORS = [
    'lightskyblue',
    'lightsalmon',
]

# == MAPPER PARAMETERS ==
UMAP_NUM_NEIGHBORS: int = 100
UMAP_MIN_DIST: float = 0.01
UMAP_METRIC: str = 'cosine'
UMAP_REPULSION_STRENGTH: float = 1.0

# == CLUSTERING PARAMETERS ==
USE_LABEL_PROPAGATION: bool = True

HDBSCAN_MIN_CLUSTER_SIZE: int = 100
HDBSCAN_MIN_SAMPLES: int = 100

OPTICS_MIN_SAMPLES: int = 50
OPTICS_MAX_EPS: float = 0.2
OPTICS_METRIC: str = 'cosine'
OPTICS_MIN_CLUSTER_SIZE: int = 30
OPTICS_XI: float = 0.05

KMEANS_NUM_CLUSTERS: int = 6
KMEANS_MAX_ITER: int = 300

# == VISUALIZATION PARAMETERS ==
NUM_EXAMPLES: int = 10
CHANNEL_COLOR_MAP = defaultdict(lambda: 'lightgray')
CHANNEL_COLOR_MAP.update(dict(enumerate(mcolors.TABLEAU_COLORS.values())))

COLORS = list(mcolors.CSS4_COLORS.values())
random.shuffle(COLORS)
CLUSTER_COLOR_MAP = {
    -1: 'lightgray',
    **dict(enumerate(COLORS))
}

ACTIVE_CHANNEL_COLOR = (0.0, 0.5, 0.0, 0.05)

__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    # Shape definitions:
    # K: number of explanation channels associated with the task.
    # A: total number of elements in the dataset.
    # D: dimensionality of the original high-dimensional latent space.
    e.log('starting experiment for automatic clustering...')

    # ~ loading the dataset
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()

    # ~ original latent space
    # first of all we need to create the original latent sapce. This should be a large numpy array 
    # containing all the original dataset's embeddings in the high-dimensional representation for each 
    # of the explanation channels separately.
    # How those are specifically obtained will have to be implemented by sub experiments.
    @e.hook('get_encoded', default=True)
    def get_encoded(e, index_data_map, num=100):
        encoded = np.concatenate([
            np.random.normal(loc=i, scale=0.2, size=(1, num, 32))
            for i in range(e.NUM_CHANNELS)
        ], axis=0)
        indices = list(range(num * e.NUM_CHANNELS))
        return encoded, indices
    
    # encoded: (K, A, D)
    # indices: (K, A)
    encoded, indices = e.apply_hook('get_encoded', index_data_map=index_data_map)
    encoded_all = np.concatenate(encoded, axis=0)
    index_tuples = np.array([(i, c) for c, inds in enumerate(indices) for i in inds])
    e.log(f'total of {len(encoded_all)} graph embeddings')

    # ~ 2D mapper
    # the second important part is that we need some sort of transformation that maps these high dimensional 
    # latent representations into the 2D space so that they can be properly visualized.
    # The concrete implementation of this will also have to be provided by the sub experiment
    @e.hook('get_mapper', default=True)
    def get_mapper(e, encoded):
        """
        This mock implementation will return a "mapper" instance which will perform the dimensionality 
        reduction simply by selecting the first and the second elements of the high-dimensional vectors.
        """
        e.log('creating UMAP mapper...')
        mapper = UMAP(
            random_state=1,
            n_neighbors=e.UMAP_NUM_NEIGHBORS,
            min_dist=e.UMAP_MIN_DIST,
            metric=e.UMAP_METRIC,
            repulsion_strength=e.UMAP_REPULSION_STRENGTH,
        )
        e.log('fittin UMAP transformation on all the raw embeddings...')
        encoded_all = np.concatenate(encoded, axis=0)
        mapper.fit(encoded_all)
        
        return mapper
        
    mapper = e.apply_hook('get_mapper', encoded=encoded)
    
    e.log(f'mapping into 2D space with {mapper.__class__.__name__} mapper')
    mapped = [mapper.transform(arr) for arr in encoded]
    mapped_all = np.concatenate([arr for arr in mapped], axis=0)
    
    # ~ baseline visualization
    # At first we do a baseline visualization where we simply plot all the embedings in 2D space as 
    # they are.
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
    for channel_index in range(NUM_CHANNELS):
        ax.scatter(
            mapped[channel_index][:, 0], mapped[channel_index][:, 1],
            s=1,
            c=CHANNEL_COLORS[channel_index],
            alpha=0.5,
            label=f'ch. {channel_index}'
        )
        
    fig_path = os.path.join(e.path, 'graph_embeddings.pdf')
    fig.savefig(fig_path)
    plt.close(fig)
    
    # ~ diagnostic visualizations
    # the UMAP python package offers a range of interesting "diagnostic" visualizations which can be used 
    # to show certain properties of the mapped space such as the connectivity structure, the neighborhood 
    # accuracy etc.
    # umap.plot.connectivity(mapper, edge_bundling='hammer')
    # fig_path = os.path.join(e.path, 'graph_embeddings__connectivity.pdf')
    # plt.savefig(fig_path)
    # plt.close('all')
    
    @e.hook('visualize_graphs', default=True)
    def visualize_graphs(e: Experiment, 
                         index_data_map: dict, 
                         indices: t.List[int],
                         active_channel: int,
                         ):
        num_graphs = len(indices)
        num_channels = e.NUM_CHANNELS
        
        fig, rows = plt.subplots(
            ncols=num_graphs, 
            nrows=num_channels,
            figsize=(8 * num_graphs, 8 * num_channels),
            squeeze=False
        )
        
        for c in range(num_channels):
            for i in range(num_graphs):
                ax = rows[c][i]
                
                index = indices[i]
                data = index_data_map[index]
                graph = data['metadata']['graph']
                ni, ei = graph['node_importances'], graph['edge_importances']
                node_positions = graph['node_positions']
                
                draw_image(ax, data['image_path'])
                plot_node_importances_border(ax, graph, node_positions, ni[:, c])
                plot_edge_importances_border(ax, graph, node_positions, ei[:, c])
                
                ax.set_title(f'index: {index} - fidelity: {graph["graph_fidelity"][c]:.1f}')
                
                # If the current channel is the active channel, then we add a background color to the current plot 
                # to visuall indicate that in the resulting PDF.
                if c == active_channel:
                    ax.set_facecolor(e.ACTIVE_CHANNEL_COLOR)
                
        return fig
    
    
    @e.hook('clustering_results', default=True)
    def clustering_results(e: Experiment,
                           clustering_name: str,
                           clustering_labels: t.List[int],
                           index_tuples: t.List[t.Tuple[int, int]],
                           encoded: t.List[list],
                           index_data_map: dict,
                           mapper: UMAP,
                           choose_examples: str = 'centroid'
                           ) -> None:
        e.log(f'processing clustering results "{clustering_name}"...')
        clusters = list(sorted(set(clustering_labels)))
        clustering_labels = np.array(clustering_labels)
        
        # We later want to export all the clustering infroamtion into a single JSON file which can then later be 
        # 
        cluster_map: t.Dict[int, dict] = {}
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        pdf = PdfPages(os.path.join(e.path, f'{clustering_name}__examples.pdf'))
        pdf.__enter__()
        for label in clusters:
            encoded_ = encoded[clustering_labels == label]
            index_tuples_ = index_tuples[clustering_labels == label]
            
            mapped_ = mapper.transform(encoded_)
            ax.scatter(
                mapped_[:, 0], mapped_[:, 1],
                s=1,
                c=e.CLUSTER_COLOR_MAP[label],
                label=f'cluster ({label})'
            )
            
            # There may be some clustering methods (DBSCAN) which do not assign clusters to all elements 
            # and instead some elements might not be clustered at all. This will be represented by the 
            # special -1 cluster label. We do not want to process any statistics or examples for these 
            # non clustered examples because that would be meaningless.
            if label < 0:
                continue
            
            # We determine the centroid of the cluster (simple average of all the cluster embeddings) in 
            # the original embedding space here and can then transform this into the mapped space to 
            # visualize it there.
            centroid = np.mean(encoded_, axis=0, keepdims=True)
            e[f'{clustering_name}/centroid/{label}'] = centroid
            centroid_mapped = mapper.transform(centroid)
            ax.scatter(
                centroid_mapped[0, 0], centroid_mapped[0, 1],
                s=2,
                c='black',
                marker='x',
            )
            ax.annotate(
                xy=centroid_mapped[0],
                text=f'({label})',
            )
            
            # cluster statistics
            channels = [c for (i, c) in index_tuples_]
            active_channel = round(np.mean(channels))
            e[f'{clustering_name}/channels/{label}'] = channels
            contributions = [index_data_map[i]['metadata']['graph']['graph_one_out'][c] for (i, c) in index_tuples_]
            e[f'{clustering_name}/contributions/{label}'] = contributions
            mask_sizes = [sum(index_data_map[i]['metadata']['graph']['node_importances'][:, c]) for (i, c) in index_tuples_]
            e[f'{clustering_name}/mask_sizes/{label}'] = mask_sizes
            graph_sizes = [len(index_data_map[i]['metadata']['graph']['node_indices']) for (i, c) in index_tuples_]
            
            # ~ examples
            indices_examples = []
            if choose_examples == 'centroid':
                distances = cosine_distances(centroid, encoded_)[0]
                nearest_indices = np.argsort(distances)[:e.NUM_EXAMPLES]
                indices_examples = [index_tuples_[i][0] for i in nearest_indices]
            elif choose_examples == 'random':
                indices_examples = random.sample([tup[0] for tup in index_tuples_], k=e.NUM_EXAMPLES)
            else:
                raise ValueError(f'The given option "{choose_examples}" is not a valid option for choosing cluster examples!')
            
            fig_examples = e.apply_hook(
                'visualize_graphs',
                index_data_map=index_data_map,
                indices=indices_examples,
                active_channel=active_channel,
            )
            fig_examples.suptitle(f' cluster ({label})')
            pdf.savefig(fig_examples)
            plt.close(fig_examples)
            
            e.log(f' * cluster ({label:02d})'
                  f' - num_elements: {len(channels):04d}'
                  f' - channel: Ø{np.mean(channels):.1f}(±{np.std(channels):.1f})'
                  f' - contribs: Ø{np.mean(contributions):.1f}(±{np.std(contributions):.1f})'
                  f' - mask size: Ø{np.mean(mask_sizes):.1f}(±{np.std(mask_sizes):.1f})')
            
            # ~ addding information to the cluster map
            cluster_map[int(label)] = {
                'centroid': centroid,
                'description': '',
                'channel': active_channel,
                'size': len(channels),
                'contribution': {
                    'mean': np.mean(contributions),
                    'std': np.std(contributions),
                },
                'mask_size': {
                    'mean': np.mean(mask_sizes),
                    'std': np.std(mask_sizes),
                },
            }
            
        ax.legend()
        fig_path = os.path.join(e.path, f'{clustering_name}__space.pdf')
        fig.savefig(fig_path)
        plt.close(fig)
        pdf.__exit__(None, None, None)
        
        # ~ saving the raw cluster informaion
        # whilte iterating over all the clusters we assemble the cluster dictionary information
        file_path = os.path.join(e.path, f'{clustering_name}__data.json')
        with open(file_path, mode='w') as file:
            content = json.dumps(cluster_map, indent=4, cls=NumericJsonEncoder)
            file.write(content)
        
    # ~ HDBSCAN* Clustering
    hdbscan = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.0,
        metric='precomputed',
    )
    distances = pairwise_distances(encoded_all, metric='cosine', n_jobs=-1)
    hdbscan.fit(distances.astype('float64'))

    e.log(f'hdbscan identified {len(set(hdbscan.labels_))} clusters...')
    
    labels = hdbscan.labels_
    if USE_LABEL_PROPAGATION:
        label_prop = LabelPropagation(
            kernel='knn',
            n_neighbors=25,
        )
        label_prop.fit(encoded_all, labels)
        labels[labels == -1] = label_prop.transduction_[labels == -1]
    
    e.apply_hook(
        'clustering_results',
        clustering_name='hdbscan',
        clustering_labels=labels,
        index_tuples=index_tuples,
        index_data_map=index_data_map,
        encoded=encoded_all,
        mapper=mapper,
        choose_examples='random',
    )
        
    # ~ OPTICS Clustering
    # The optics clustering does this by 
    e.log('starting optics clustering...')
    optics = OPTICS(
        min_samples=OPTICS_MIN_SAMPLES,
        min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
        max_eps=OPTICS_MAX_EPS,
        metric=OPTICS_METRIC,
        xi=OPTICS_XI,
    )
    optics.fit(encoded_all)
    num_clusters = len(set(optics.labels_))
    e.log(f'identified {num_clusters} clusters...')
    
    labels = optics.labels_
    if USE_LABEL_PROPAGATION:
        label_prop = LabelPropagation(
            kernel='knn',
            n_neighbors=25,
        )
        label_prop.fit(encoded_all, labels)
        labels[labels == -1] = label_prop.transduction_[labels == -1]
    
    e.apply_hook(
        'clustering_results',
        clustering_name='optics',
        clustering_labels=labels,
        index_tuples=index_tuples,
        index_data_map=index_data_map,
        encoded=encoded_all,
        mapper=mapper,
        choose_examples='random',
    )
    
    e.log('fitting semi-supervised UMAP with cluster assignments...')
    mapper_optics = UMAP(
        random_state=1,
        n_neighbors=e.UMAP_NUM_NEIGHBORS,
        min_dist=e.UMAP_MIN_DIST,
        metric=e.UMAP_METRIC,
        repulsion_strength=e.UMAP_REPULSION_STRENGTH,
    )
    mapper_optics.fit(encoded_all, y=labels)
    e.apply_hook(
        'clustering_results',
        clustering_name='optics_semi',
        clustering_labels=labels,
        index_tuples=index_tuples,
        index_data_map=index_data_map,
        encoded=encoded_all,
        mapper=mapper_optics,
        choose_examples='random',
    )
    
    # ~ Spectral clustering
    # In this section we use a different clustering method, namely kmeans clustering. The reason for this is that unlike DBSCAN 
    # kmeans clustering is a method where the number of clusters has to be pre-defined as a parameter and is thus just a 
    # different clustering principle. The idea is to hopefully identify an appropriate number of clusters using DBSCAN and then 
    # use this method to determine the clusters (because one problem of DBSCAN is that it won't actually assign every element to 
    # a cluster!)
    e.log(f'starting spectral clustering with {KMEANS_NUM_CLUSTERS} clusters...')
    
    kmeans = KMeans(
        n_clusters=KMEANS_NUM_CLUSTERS,
        max_iter=KMEANS_MAX_ITER,
    )
    kmeans.fit(encoded_all)
    
    e.apply_hook(
        'clustering_results',
        clustering_name='kmeans',
        clustering_labels=kmeans.labels_,
        index_tuples=index_tuples,
        index_data_map=index_data_map,
        encoded=encoded_all,
        mapper=mapper,
        choose_examples='random',
    )
    
    

@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()