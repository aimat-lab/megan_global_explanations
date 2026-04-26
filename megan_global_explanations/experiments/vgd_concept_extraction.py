"""
This is the base experiment for the generation of a concept clustering from a visual graph dataset and an
already pre-trained Megan model. After the model and the dataset have been loaded, the HDBSCAN algorithm will 
be used to find the concept clusters in the model's latent space for each of the explanation channels.

Additionally, there is the option to optimize prototype graphs for each of the clusters and to generate
natural language descriptions and hypotheses for the prototypes using GPT-4 API.

All information created for each of the clusters is then saved into a persistent format on the disk and 
additionally a concept cluster report PDF is generated which can be used by human users to understand 
the concept explanations.
"""
import os
import json
import pickle
import random
import pathlib
import traceback
import typing as t
from copy import deepcopy
from collections import defaultdict

import umap
import hdbscan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cosine, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.config import Config
from visual_graph_datasets.graph import graph_expand_mask
from visual_graph_datasets.graph import graph_find_connected_regions
from visual_graph_datasets.graph import extract_subgraph
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.colors import ColorProcessing
from graph_attention_student.utils import array_normalize
from graph_attention_student.torch.megan import Megan

from megan_global_explanations.visualization import create_concept_cluster_report
from megan_global_explanations.prototype.optimize import genetic_optimize
from megan_global_explanations.prototype.optimize import embedding_distance_fitness
from megan_global_explanations.prototype.colors import sample_from_cogiles
from megan_global_explanations.prototype.colors import mutate_add_edge
from megan_global_explanations.prototype.colors import mutate_remove_edge
from megan_global_explanations.prototype.colors import mutate_modify_node
from megan_global_explanations.prototype.colors import mutate_add_node
from megan_global_explanations.prototype.colors import mutate_remove_node
from megan_global_explanations.gpt import describe_color_graph
from megan_global_explanations.data import ConceptWriter
from megan_global_explanations.data import ConceptReader
from megan_global_explanations.data import select_representatives
from megan_global_explanations.data import sharpen_scores
from megan_global_explanations.utils import EXPERIMENTS_PATH

mpl.use('Agg')

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
VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or 
#       "classification"
DATASET_TYPE: str = 'regression'
# :param CHANNEL_INFOS:
#       This dictionary can optionally be given to supply additional information about the individual 
#       explanation channels. The key should be the index of the channel and the value should again be 
#       a dictionary that contains the information for the corresponding channel.
CHANNEL_INFOS: t.Dict[int, dict] = defaultdict(lambda: {
    'name': 'n/a',
    'color': 'lightgray'
})
# :param SUBSET:
#       Optionally can be used to define an integer number of elements that should be randomly sampled from 
#       the dataset to be used for the clustering. If this is None, the entire dataset will be used.
SUBSET: t.Optional[int] = None

# == MODEL PARAMETERS == 
# These parameters determine the details related to the model that should be used for the 
# concept extraction. For this experiment, the model should already be trained and only 
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the 
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'rb_dual_motifs.ckpt')

# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a 
#       fidelity higher than this will be used as possible candidates for the clustering.
FIDELITY_THRESHOLD: float = 0.5
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially 
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
MIN_CLUSTER_SIZE: int = 20
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the 
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while 
#       lower values tend to result in more clusters.
MIN_SAMPLES: int = 5
# :param CLUSTER_SELECTION_METHOD:
#       This string value determines the method that is used to select the clusters from the HDBSCAN
#       algorithm. The default value is 'leaf' which is the most conservative method. Other possible
#       values are 'eom' and 'leaf'.
CLUSTER_SELECTION_METHOD: str = 'leaf'
# :param CLUSTERING_METRIC:
#       This string value determines the distance metric that is used for the HDBSCAN clustering,
#       the centroid distance calculations, and the UMAP visualization. Common options include
#       'manhattan', 'euclidean', and 'cosine'.
CLUSTERING_METRIC: str = 'manhattan'
# :param CLUSTER_REPRESENTATIVE:
#       This string value determines how the representative point for each cluster is calculated.
#       'centroid' computes the mean of all cluster embeddings. 'medoid' selects the actual
#       cluster member whose total distance to all other members is minimal. The medoid is more
#       robust to outliers and is always a real data point, but is more expensive to compute.
CLUSTER_REPRESENTATIVE: str = 'centroid'

# == REPRESENTATIVE SELECTION PARAMETERS ==
# These parameters control which cluster members are stored as "representatives"
# inside the Clustering archive. Representatives carry full explanation data
# (SMILES, importances, predictions, deviations) so the report PDF can be
# generated without needing the model or dataset at render time.

# :param NUM_REPRESENTATIVES:
#       Number of representative members to store per cluster.
NUM_REPRESENTATIVES: int = 16
# :param REPRESENTATIVE_STRATEGY:
#       How to select representatives from the cluster members.
#         - 'closest': deterministic — the N nearest members to the centroid.
#         - 'temperature': stochastic — softmax over negative centroid-distances
#           with temperature auto-scaled by ``REPRESENTATIVE_TEMPERATURE * median_distance``.
#           Low temperature biases toward the centroid; high temperature gives diversity.
REPRESENTATIVE_STRATEGY: str = 'temperature'
# :param REPRESENTATIVE_TEMPERATURE:
#       Temperature multiplier for the 'temperature' strategy. Expressed as a
#       fraction of the median intra-cluster centroid distance.
#       0.5 = strong centroid bias, 1.0 = moderate, 2.0+ = near-uniform.
REPRESENTATIVE_TEMPERATURE: float = 0.7

# == SCORE SHARPENING PARAMETERS ==
# These parameters control optional post-processing of cluster distances that
# forces more decisive cluster assignments. Sharpening maps a distance vector
# through softmax/sparsemax of negative distances, producing near-binary scores
# when one cluster is clearly closest while staying ambiguous when the input is
# genuinely uncertain. Elements whose per-channel fidelity falls below
# ``FIDELITY_THRESHOLD`` are left unsharpened so that out-of-cluster samples
# don't get forced into a random cluster.

# :param SHARPEN_SCORES:
#       When True, sharpen cluster distances in the sample_element_distances
#       heatmap via softmax/sparsemax. Does not affect the raw scoring hook
#       (only the visualization).
SHARPEN_SCORES: bool = True
# :param SHARPEN_METHOD:
#       'sparsemax' (produces exact zeros for weak matches when a clear winner
#       exists) or 'softmax' (smoother, never exactly zero).
SHARPEN_METHOD: str = 'sparsemax'
# :param SHARPEN_TEMPERATURE:
#       Temperature multiplier passed to ``sharpen_scores``. Auto-scaled by
#       the median per-row distance. Lower = more discrete, higher = smoother.
SHARPEN_TEMPERATURE: float = 0.5

# :param SORT_SIMILARITY:
#       This boolean flag determines whether the clusters should be sorted by their similarity.
#       If this is True, the clusters will be sorted by their similarity which means that the order 
#       of the clusters will be determined by the similarity with each other. Having this enables makes 
#       the concept report a bit more readable because similar clusters will appear close to each other 
#       in the report PDF.
SORT_SIMILARITY: bool = True

# == PROTOTYPE OPTIMIZATION PARAMETERS ==
# These parameters configure the process of optimizing the cluster prototype representatation

# :param OPTIMIZE_CLUSTER_PROTOTYPE:
#       This boolean flag determines whether the prototype optimization should be executed at 
#       all or not. If this is False, the entire optimization routine will be skipped during the 
#       cluster discovery.
OPTIMIZE_CLUSTER_PROTOTYPE: bool = True
# :param INITIAL_POPULATION_SAMPLE:
#       This integer number determines the number of initial samples that are drawn from the cluster 
#       members as the initial population of the prototype optimization GA procedure.
INITIAL_POPULATION_SAMPLE: int = 200
# :param OPTIMIZE_PROTOTYPE_POPSIZE:
#       This integer number determines the population size of the genetic optimization algorithm
#       that is used to optimize the prototype representation.
OPTIMIZE_PROTOTYPE_POPSIZE: int = 1000
# :param OPTIMIZE_PROTOTYPE_EPOCHS:
#       This integer number determines the number of epochs that the genetic optimization algorithm
#       will be executed for the prototype optimization.
OPTIMIZE_PROTOTYPE_EPOCHS: int = 50
# :param OPENAI_KEY:
#       This string value has to be the OpenAI API key that should be used for the GPT-4 requests
#       that will be needed to generate the natural language descriptions of the prototypes.
OPENAI_KEY: str = os.getenv('OPENAI_KEY')
# :param DESCRIBE_PROTOTYPE:
#       This boolean flag determines whether the prototype description should be generated at all
#       or not. If this is False, the entire description routine will be skipped during the
#       cluster discovery.
DESCRIBE_PROTOTYPE: bool = True
# :param HYPOTHESIZE_PROTOTYPE:
#       This boolean flag determines whether the prototype hypothesis should be generated at all
#       or not. If this is False, the entire hypothesis routine will be skipped during the
#       cluster discovery.
HYPOTHESIZE_PROTOTYPE: bool = True
# :param CONTRIBUTION_THRESHOLDS:
#       This dictionary determines the thresholds to be used when converting the contribution values 
#       of classification tasks into the strings such that they can be passed to the language model 
#       for the hypothesis generation. The keys are the contribution values and the values are the 
#       strings that will be used to describe the impact of these contributions in words.
#       Note that this will only be used for classification problems since for classification problems 
#       the contribution values are measured in classification logits which do not have a direct meaning 
#       to the language model. In contrast, regression contributions are measured directly in the 
#       target space and therefore do not need to be converted.
CONTRIBUTION_THRESHOLDS: dict = {
    10: 'small',
    20: 'high'
}

# == VISUALIZATION PARAMETERS ==
# These parameters determine the details of the visualizations that will be created as part of the 
# artifacts of this experiment.

# :param PLOT_UMAP:
#       This boolean flag determines whether the UMAP visualization of the graph embeddings should be
#       created or not. If this is True, the UMAP visualization will be created and saved as an additional 
#       artifact of the experiment.
PLOT_UMAP: bool = False
# :param NUM_SAMPLE_ELEMENTS:
#       The number of random dataset elements to sample for the sample-to-cluster distance heatmap.
#       Set to 0 to skip this visualization.
NUM_SAMPLE_ELEMENTS: int = 25

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('get_dataset_path')
def get_dataset_path(e: Experiment) -> str:
    """
    This hook is responsible for returning the path to the visual graph dataset that is to be used
    for the concept clustering. This may either be an absolute string path to a visual graph dataset
    folder on the local system. Otherwise this may also be a valid string identifier for a vgd in
    which case it will be downloaded from the remote file share instead.
    """
    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
        
    else:
        config = Config()
        config.load()
        
        dataset_path = ensure_dataset(
            dataset_name=e.VISUAL_GRAPH_DATASET,
            config=config,
            logger=e.logger,
        )
        
    return dataset_path


@experiment.hook('load_dataset', replace=False)
def load_dataset(e: Experiment,
                 path: str,
                 ) -> dict:
    """
    This hook takes a local path to a (visual graph) dataset as the only argument and is then 
    responsibe for loading and returning that dataset as a index_data_map.
    
    Additionally, this function has to set up the experiment values "node_dim", "edge_dim" and "out_dim"
    based on the dataset that has been loaded.
    
    This default implementation uses the default VisualGraphDatasetReader to load the dataset from the disk.
    """
    reader = VisualGraphDatasetReader(
        path=path,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    processing = reader.read_process().processing
    
    example_graph = list(index_data_map.values())[0]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]
    e['out_dim'] = example_graph['graph_labels'].shape[0]
    e.log(f'loaded dataset with {e["node_dim"]} node features and {e["edge_dim"]} edge features')
    
    return index_data_map, processing


@experiment.hook('load_model')
def load_model(e: Experiment,
               path: str
               ) -> Megan:
    """
    This hook receives a local file system path as the only argument and is supposed to load the 
    MEGAN model from that path and return the instance.
    
    This standard implementation just loads the default Megan torch implementation
    """
    model = Megan.load(path)
    return model


@experiment.hook('optimize_prototype', default=False, replace=False)
def optimize_prototype(e: Experiment,
                       model: Megan,
                       channel_index: int,
                       processing: ProcessingBase,
                       cluster_graphs: t.List[tv.GraphDict],
                       cluster_embeddings: np.ndarray,
                       **kwargs,
                       ) -> dict:
    """
    This hook receives the model, the channel index, processing a list of graphs and a list of cluster embeddings 
    as parameters and the purpose is to use all that information for somehow derive a cluster prototype in the format 
    of a single graph dict element.
    
    The default implementation of this hook is to simply return None which indicates that no prototype was or could be 
    created.
    """
    return None


@experiment.hook('prototype_hypothesis', replace=False, default=False)
def prototype_hypothesis(e: Experiment,
                         value: str,
                         image_path: str,
                         channel_index: int,
                         **kwargs,
                         ) -> t.Optional[str]:
    """
    This hook takes various information about the prototype and the concept cluster as parameters and 
    is supposed to generate some kind of natural language hypothesis about the causal structure property 
    relationships that could be underlying to this concept.
    
    This generation is usually accomplished by a large language model such as OpenAI's GPT.
    
    The standard implementation of this hook just returns None, which indicates that no suitable hypothesis
    could be generated for the target concept cluster. This is because the generation of the hypothesis is 
    heavily domain dependent and a generic implementation is not possible.
    
    :returns: Either None (in which case it is ignored) or a string to be included as the hypothesis.
    """
    e.log(' * skipping hypothesis generation for prototype')
    return None


@experiment.hook('describe_prototype', replace=False, default=False)
def describe_prototype(e: Experiment,
                       value: str,
                       image_path: str,
                       ) -> str:
    
    try:
        description, _ = describe_color_graph(
            api_key=e.OPENAI_KEY,
            image_path=image_path,
        )
        print(description)
        return (
            f'Prototoype Representation: {value}\n'
            f'GPT-4 Description: {description}'
        )
        
    except Exception as exc:
        e.log(f'error "{exc}" while describing the prototype - skipping!')
        # traceback.print_exc()
        
        return 'No description generated.'


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # ~ loading the dataset
    # The dataset is either loaded from the local file system as a path or it is downloaded 
    # from the remote file share first by providing it's unique string identifier.
    
    # :hook get_dataset_path:
    #       This hook will return the absolute string path to the visual graph dataset.
    dataset_path = e.apply_hook('get_dataset_path')
    
    e.log('loading dataset...')
    # :hook load_dataset:
    #       This hook is responsible for loading the dataset from the given path and returning it as a
    #       index_data_map. Additionally, this function has to set up the experiment values "node_dim",
    #       "edge_dim" and "out_dim" based on the dataset that has been loaded.
    #       It also returns the processing object that was stored alongside the dataset.
    index_data_map, processing = e.apply_hook(
        'load_dataset',
        path=dataset_path,
    )
    num_graphs = len(index_data_map)
    e.log(f'loaded dataset with {num_graphs} elements')
    indices = list(index_data_map.keys())
    # 03.06.24
    # For the particularly large dataset we need a method to reduce the number of graphs for the clustering 
    # because the clustering will become a significant runtime bottleneck otherwise...
    if e.SUBSET and len(indices) > e.SUBSET:
        e.log(f'sub-sampling the graphs to reduce the number of elements to {e.SUBSET}...')
        indices = random.sample(indices, k=e.SUBSET)
    
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    e.log(f'working with {len(graphs)} graphs and {len(indices)} indices...')
    
    # ~ loading the model
    # Besides the dataset we also have to load the model from its persistent representation on the disk
    # so that we can use it for the concept clustering
    
    model: Megan = e.apply_hook(
        'load_model',
        path=e.MODEL_PATH,
    )
    num_channels = model.num_channels
    e['num_channels'] = num_channels
    e.log(f'loaded model of the class: {model.__class__.__name__} '
          f'with {num_channels} explanation channels')
    
    # ~ Concept clustering the latent space
    
    e.log('running the model forward pass for all the graphs...')
    # First of all we need to query the model using all the graphs from the dataset to obtain the model's
    # predictions as well as the explanations and graph embeddings that are required for the clustering
    infos = model.forward_graphs(graphs, batch_size=1000) 
    # We also want to calculate the loo devations aka the channel-specific fidelity values as those will 
    # be part of the criterium that we will use to filter the relevant concept clusters.
    deviations = model.leave_one_out_deviations(graphs, batch_size=1000)
    
    e.log('updating the dataset...')
    # To make it easier going forward we will actually attach all the information gained from this 
    # model forward pass to the dataset structure itself (to the graph dicts)
    for index, graph, info, dev in zip(indices, graphs, infos, deviations):
        
        # 31.01.24
        # Had to add this conditional only due to backwards compatibility issues with the old visual graph 
        # datasets where the "repr" key was not yet part of the metadata.
        metadata = index_data_map[index]['metadata']
        if 'repr' in metadata:
            graph['graph_repr'] = metadata['repr']
            
        # graph_output: (O, )
        graph['graph_output'] = info['graph_output']
        # graph prediction is supposed to be a single value that determines the overall prediction of the 
        # model. This differs for regression and classification datasets where the regression result is the 
        # value itself and the classification result is the predicted class index
        # graph_prediction: (, )
        if e.DATASET_TYPE == 'regression':
            graph['graph_prediction'] = info['graph_output'][0]
        elif e.DATASET_TYPE == 'classification':
            graph['graph_prediction'] = np.argmax(info['graph_output'])
        
        # graph_embeddings: (D, K)
        graph['graph_embedding'] = info['graph_embedding']
        # node_importance: (V, K)
        graph['node_importances'] = array_normalize(info['node_importance'])
        # edge_importance: (E, K)
        graph['edge_importances'] = array_normalize(info['edge_importance'])
    
        # graph_fidelity: (O, K)
        graph['graph_deviation'] = dev

        # The graph fidelity should be an vector with as many elements as there are explanation channels 
        # so basically there should be one fidelity value per explanation channel. The derivation of that 
        # form is differently defined for regression and classification tasks.
        # graph_fidelity: (K, )
        if e.DATASET_TYPE == 'regression':
            graph['graph_fidelity'] = np.array([-dev[0, 0], dev[0, 1]])
        elif e.DATASET_TYPE == 'classification':
            matrix = np.array(dev)
            mask = 1.0 - np.eye(info['graph_output'].shape[0])
            # graph['graph_fidelity'] = np.diag(matrix) - np.sum(matrix * mask, axis=1)
            graph['graph_fidelity'] = np.diag(matrix)
            
    # ~ saving graphs
    # The graphs were just updated with additional information from the prediction. These graph structures might be needed in 
    # the analysis as well so we will save them as a separate experiment artifact.
    e.log('saving the raw graph data as a JSON file...')
    graphs_path = os.path.join(e.path, 'graphs.json')
    with open(graphs_path, 'w') as file:
        json.dump(graphs, file, cls=NumericJsonEncoder)
    
    # Now we calculate the concept clusters separately for each of the explanation channels of the model.
    e.log('starting concept clustering...')
    cluster_infos: t.List[dict] = []
    channel_clusterers: t.Dict[int, object] = {}
    channel_labels: t.Dict[int, np.ndarray] = {}
    cluster_index = 0
    for channel_index in range(num_channels):
        
        e.log(f'> CHANNEL {channel_index}')
        
        # Here we filter according to the fidelity - we only want elements with a certain minumum fidelity to be 
        # eligible for the clustering to begin with. The reasoning here is that there are a lot of elements which 
        # do not have any activation in one of the channels and therefore also have ~0 fidelity for that channel.
        # Those elements are not going to be informative at all.
        channel_indices, channel_graphs = zip(*[
            (index, graph) 
            for index, graph in zip(indices, graphs) 
            if graph['graph_fidelity'][channel_index] > e.FIDELITY_THRESHOLD
        ])
        channel_indices = np.array(channel_indices)
        
        # graph_embeddings: (B, D)
        # This is an array of the actual graph embedding vectors - specifically for the current explanation channel
        graph_embeddings = np.array([graph['graph_embedding'][:, channel_index] for graph in channel_graphs])
        e.log(f' * filtered {len(channel_indices)} elements from {len(indices)}')
        
        # Fit HDBSCAN. For metrics not natively supported by HDBSCAN's tree-based algorithms
        # (e.g. cosine), we use euclidean on the raw embeddings instead. For L2-normalized
        # embeddings (unit sphere), euclidean preserves the same distance ranking as cosine
        # (euclidean^2 = 2 * cosine_distance). This allows HDBSCAN's prediction functions
        # (membership_vector, approximate_predict) to work for all metrics.
        _TREE_METRICS = {'euclidean', 'manhattan', 'minkowski', 'chebyshev', 'l1', 'l2', 'cityblock'}
        if e.CLUSTERING_METRIC in _TREE_METRICS:
            hdbscan_metric = e.CLUSTERING_METRIC
        else:
            hdbscan_metric = 'euclidean'
            e.log(f'  metric {e.CLUSTERING_METRIC} not supported by HDBSCAN trees, '
                  f'using euclidean (equivalent for L2-normalized embeddings)')

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=e.MIN_CLUSTER_SIZE,
            min_samples=e.MIN_SAMPLES,
            metric=hdbscan_metric,
            cluster_selection_method=e.CLUSTER_SELECTION_METHOD,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(graph_embeddings)
        # labels: (B, )
        # This is an array that contains the cluster indices for every element of the dataset. It assigns an integer
        # cluster index to each element, where -1 is a special index indicating that an element does not belong to
        # any cluster.
        # A list of all the possible cluster indices from which we can derive how many clusters there have been found
        # in general.
        channel_clusterers[channel_index] = clusterer
        channel_labels[channel_index] = labels

        clusters = [c for c in set(labels) if c >= 0]
        num_clusters = len(clusters)
        e.log(f' * found {num_clusters} clusters')
   
        for cluster in clusters:
            
            # Now, for each specific cluster we want to get the subset of the dataset elements that actually belong 
            # to that cluster therefore we construct a boolean mask here that defines if an element belongs to 
            # the current cluster.
            # mask: (B, )  example [True, True, False, False, True, ....]
            mask = (labels == cluster)
            
            # cluster_graph_embeddings: (B_cluster, D)
            cluster_graph_embeddings = graph_embeddings[mask]
            # cluster_centroid: (D, )
            if e.CLUSTER_REPRESENTATIVE == 'medoid':
                intra_dists = pairwise_distances(cluster_graph_embeddings, metric=e.CLUSTERING_METRIC)
                medoid_idx = np.argmin(intra_dists.sum(axis=1))
                cluster_centroid = cluster_graph_embeddings[medoid_idx]
            else:
                cluster_centroid = np.mean(cluster_graph_embeddings, axis=0)
            # cluster_indices: (B_cluster, )
            cluster_indices = channel_indices[mask]
            cluster_elements = deepcopy([index_data_map[i] for i in cluster_indices])
            cluster_index_tuples = [(i, channel_index) for i in cluster_indices]
            cluster_graphs = [index_data_map[i]['metadata']['graph'] for i in cluster_indices]
            cluster_image_paths = [index_data_map[i]['image_path'] for i in cluster_indices]
            
            if e.DATASET_TYPE == 'regression':
                cluster_contribution = np.mean([graph['graph_deviation'][0, channel_index] for graph in cluster_graphs])
            elif e.DATASET_TYPE == 'classification':
                cluster_contribution = np.mean([graph['graph_deviation'][channel_index, channel_index] for graph in cluster_graphs])   
            
            info = {
                'channel_index':        channel_index,
                'index':                cluster_index,
                'hdbscan_label':        cluster,
                'embeddings':           cluster_graph_embeddings,
                'centroid':             cluster_centroid,
                'index_tuples':         cluster_index_tuples,
                'elements':             cluster_elements,
                'graphs':               cluster_graphs,
                'image_paths':          cluster_image_paths,
                'name':                 e.CHANNEL_INFOS[channel_index]['name'],
                'color':                e.CHANNEL_INFOS[channel_index]['color'],
            }
            
            # ~ Select representative members and compute member statistics
            # Representatives carry full explanation data (SMILES, importances, etc.)
            # so the clustering report can be generated without the model/dataset.
            num_reps = e.NUM_REPRESENTATIVES
            rep_strategy = e.REPRESENTATIVE_STRATEGY
            rep_temperature = e.REPRESENTATIVE_TEMPERATURE

            centroid_dists = pairwise_distances(
                cluster_graph_embeddings, cluster_centroid.reshape(1, -1),
                metric=e.CLUSTERING_METRIC,
            ).flatten()

            rep_indices = select_representatives(
                centroid_dists, n=num_reps,
                strategy=rep_strategy, temperature=rep_temperature,
            )

            representatives = []
            for ri in rep_indices:
                graph = cluster_graphs[ri]
                representatives.append({
                    'smiles': graph.get('graph_repr', ''),
                    'dataset_index': int(cluster_index_tuples[ri][0]),
                    'node_importances': np.asarray(graph.get('node_importances', [])).tolist(),
                    'edge_importances': np.asarray(graph.get('edge_importances', [])).tolist(),
                    'graph_output': np.asarray(graph.get('graph_output', [])).tolist(),
                    'graph_deviation': np.asarray(graph.get('graph_deviation', [])).tolist(),
                })
            info['representatives'] = representatives

            info['member_stats'] = {
                'graph_outputs': np.array([
                    np.asarray(g.get('graph_output', 0.0)).item()
                    if np.asarray(g.get('graph_output', 0.0)).ndim == 0
                    else np.asarray(g.get('graph_output', [0.0]))[0]
                    for g in cluster_graphs
                ]),
                'graph_deviations': np.array([
                    np.asarray(g.get('graph_deviation', np.zeros((1, 1)))).flatten()
                    for g in cluster_graphs
                ]),
                'mask_sizes': np.array([
                    np.asarray(g.get('node_importances', np.zeros((1, 1)))).sum(axis=0)
                    for g in cluster_graphs
                ]),
                'centroid_distances': centroid_dists,
            }

            e.log(f' ({cluster}/{num_clusters})'
                  f' - cluster index: {cluster_index}'
                  f' - channel index: {channel_index}'
                  f' - contribution: {cluster_contribution:.2f}')
            # Optionally it is also possible to derive an approximation for the prototype of the cluster by doing 
            # an optimization scheme. However, this will require quite some time so it can be skipped as well.
            if e.OPTIMIZE_CLUSTER_PROTOTYPE:

                try:
                    e.log(f' ({cluster}/{num_clusters}) optimizing prototype...')
                    # :hook optimize_prototype:
                    #       Given the model, the channel index, the processing instance, the list of cluster graphs and
                    #       the list of cluster embeddings, this hook is supposed to return a dictionary that describes 
                    #       the optimized prototype for the cluster. This dictionary will have to contain the two keys 
                    #       "graph" (the graph dict representation) and "value" (the string domain representation) of the 
                    #       the prototype.
                    
                    # There are rare cases where this also fails due to the initial elements being empty for example
                    # in that case we 
                    cluster_prototype: dict = e.apply_hook(
                        'optimize_prototype',
                        model=model,
                        channel_index=channel_index,
                        processing=processing,
                        cluster_graphs=cluster_graphs,
                        cluster_embeddings=cluster_graph_embeddings,
                    )
                    
                    prototype_graph = cluster_prototype['graph']
                    prototype_value = cluster_prototype['value']
                    
                    # Additionally the graph representation will have to contain the keys "node_importances" and
                    # "edge_importances" which are the explanation masks for the prototype and can be obtained 
                    # by querying the model with the final prototype graph and updating its attributes.
                    prototype_info = model.forward_graphs([prototype_graph])[0]
                    prototype_graph['node_importances'] = prototype_info['node_importance']
                    prototype_graph['edge_importances'] = prototype_info['edge_importance']
                    
                    # 29.01.24
                    # So actually there is a chance that this visualization step may fail for some very exotic SMILES.
                    fig, _ = processing.visualize_as_figure(
                        value=cluster_prototype['value'],
                        graph=cluster_prototype['graph'],
                        width=1000,
                        height=1000,
                    )
                    prototype_path = os.path.join(e.path, f'prototype__cl{cluster_index:02d}.png')
                    fig.savefig(prototype_path)
                    plt.close(fig)
                    
                    # The prototype that we add to the list needs to be a visual graph element dictionary which means that 
                    # it has to have the following structure consisting of an image path and the metadata dict, which then in 
                    # turn contains the actual graph representation dict.
                    
                    prototype_graph['graph_repr'] = prototype_value
                    
                    prototype = {
                        'image_path': prototype_path,
                        'metadata': {
                            'graph':    prototype_graph, 
                            'repr':     prototype_value,
                        },
                    }
                    info['prototypes'] = [prototype]
                    
                    # It is also possible to specifically disable/enable the description of the prototypes
                    if e.DESCRIBE_PROTOTYPE:
                    
                        # :hook describe_prototype:
                        #       Given the string representation of the prototype and the path to the visualization of the 
                        #       prototype, this hook is supposed to return a string description for the prototype.
                        #       which will be included in the concept report.
                        description = e.apply_hook(
                            'describe_prototype',
                            value=cluster_prototype['value'],
                            image_path=prototype_path,
                        )
                        info['description'] = description
                        
                    if e.HYPOTHESIZE_PROTOTYPE:
                        # :hook prototype_hypothesis:
                        #       Given the string representation of the prototype, the path to the visualization and the 
                        #       description string, this hook is supposed to return a string hypothesis for the prototype.
                        #       This hypothesis is supposed to provide a starting point about the causal structure property 
                        #       relationship of the prototype & the cluster as a whole.
                        hypothesis = e.apply_hook(
                            'prototype_hypothesis', 
                            value=cluster_prototype['value'],
                            image_path=prototype_path,
                            channel_index=channel_index,
                            contribution=cluster_contribution,
                        )
                        # Thers is a chance that the hypothesis generation fails or is not implemented for a specific 
                        # target domain. So only if a textual hypothesis is actually returned we want to include it in
                        # the cluster info.
                        if hypothesis is not None:
                            info['hypothesis'] = hypothesis
                    
                except Exception as exc:
                    e.log(f'error "{exc}" while optimizing the prototype - skipping!')
                    traceback.print_exc()
            
            cluster_index += 1
            cluster_infos.append(info)
    
    print(cluster_infos[0].keys())
            
    # We definitely want to store the cluster infos to the experiment storage so that we can access them 
    # later on during the analysis as well.
    #e['cluster_infos'] = cluster_infos
    
    # Only if configured we are actually going to sort the clusters by their similarity. This similarity sorting 
    # works like this: Within each channel (!) we are going to start with the first cluster and then we are going
    # to find the cluster that is most similar to it. We are going to repeat this process until all clusters are
    # are added to the new list.
    if e.SORT_SIMILARITY:
        
        e.log('sorting clusters by similarity...')
        cluster_infos_sorted = []
        for k in range(e['num_channels']):
            infos = [info for info in cluster_infos if info['channel_index'] == k]
            
            info = infos.pop(0)
            cluster_infos_sorted.append(info)
            
            while len(infos) != 0:
                
                centroid = info['centroid']
                centroid_distances = pairwise_distances(
                    np.expand_dims(centroid, axis=0),
                    [i['centroid'] for i in infos],
                    metric=e.CLUSTERING_METRIC,
                )
                
                index = np.argmin(centroid_distances[0])
                info = infos.pop(index)
                cluster_infos_sorted.append(info)
        
        cluster_infos = cluster_infos_sorted
        
    for index, info in enumerate(cluster_infos):
        info['index'] = index
    
    # ~ Clustering metrics
    
    e.log('calculating clustering metrics...')
    for channel_index in range(e['num_channels']):
            
        infos = [info for info in cluster_infos if info['channel_index'] == channel_index]
        # Clustering metrics cannot be calculated if there are not at least 2 clusters!
        if len(infos) < 2:
            continue
        
        embeddings = []
        labels = []
        for index, info in enumerate(infos):
            embeddings += info['embeddings'].tolist()
            labels += [index for _ in info['embeddings']]
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # calculating the silhouette score
        # The silhouette score is a measure of how similar an object is to its own cluster (cohesion) compared to
        # other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicates that the
        # object is well matched to its own cluster and poorly matched to neighboring clusters.
        sil_value = silhouette_score(embeddings, labels)
        dbi_value = davies_bouldin_score(embeddings, labels)
        
        e[f'{channel_index}/silhouette'] = sil_value
        e[f'{channel_index}/dbi'] = dbi_value
        
        e.log(f'channel {channel_index}'
              f' - silhouette: {sil_value:.3f}'
              f' - dbi: {dbi_value:.3f}')
    
    # ~ Dimensionality reduction
    # In this section we want to create perform a dimensionality reduction on the graph embedding latent space 
    # so that we can get somewhat of a visual understanding of the clustering that is going on there. For 
    # this purpose we are using UMAP - specifically we are using a separate UMAPing process for each of the 
    # explanation channels.
    
    if e.PLOT_UMAP:
        
        e.log(f'starting to create {e["num_channels"]} UMAP visualizations...')
        fig, rows = plt.subplots(
            ncols=e['num_channels'],
            nrows=2,
            figsize=(20, 20),
            squeeze=False,
        )
        
        for channel_index in range(e['num_channels']):
            e.log(f'creating UMAP visualization for channel {channel_index}...')
            
            # As a first step we want to filter the graphs. So we dont actually want to use the embeddings of 
            # all the graphs for the mapping but only a subset of them according to the fidelty threshold.
            # because the embeddings with really low fidelity dont make any sense to look at anyways and would 
            # only "pollute" the visualization.
            channel_graphs = [
                graph
                for graph in graphs
                if graph['graph_fidelity'][channel_index] > e.FIDELITY_THRESHOLD
            ]
            
            # graph_embeddings: (B, D)
            embeddings = np.array([graph['graph_embedding'][:, channel_index] for graph in channel_graphs])
            e.log(f' * filtered {len(channel_graphs)} elements from {len(graphs)}')
            
            mapper = umap.UMAP(
                n_neighbors=100,
                min_dist=0.0,
                n_components=2,
                metric=e.CLUSTERING_METRIC,
                repulsion_strength=1.0,
            )
            mapped = mapper.fit_transform(embeddings)
            
            # Then in the first row, we ware going to just plot the latent space in raw format without indicating the 
            # actual clustering results.
            ax_raw = rows[0][channel_index]
            ax_raw.scatter(
                mapped[:, 0], mapped[:, 1],
                color=e.CHANNEL_INFOS[channel_index]['color'],
                linewidths=0.0,
                s=10,
                alpha=0.25,
            )
            ax_raw.set_title(f'UMAP Reduced Explanation Embeddings\n'
                             f'Channel {channel_index} - {e.CHANNEL_INFOS[channel_index]["name"]}')
            
            # In this second row we are going to plot the clustering results which includes the elements 
            # that were chosen as part of the clusters as well as the centroids of those clusters.
            e.log(' * plotting clustering results')
            ax_cls = rows[1][channel_index]
            ax_cls.set_title('HDBSCAN Clusters and Centroids')
            
            ax_cls.scatter(
                mapped[:, 0], mapped[:, 1],
                color='lightgray',
                linewidths=0.0,
                s=10,
                zorder=-10,
            )
            
            infos = [info for info in cluster_infos if info['channel_index'] == channel_index]
            for info in infos:
                
                embeddings = info['embeddings']
                embeddings_mapped = mapper.transform(embeddings)
                ax_cls.scatter(
                    embeddings_mapped[:, 0], embeddings_mapped[:, 1],
                    color='lightgreen',
                    linewidths=0.0,
                    s=5,
                )
                
                centroid = info['centroid']
                centroid_mapped = mapper.transform(np.expand_dims(centroid, axis=0))
                ax_cls.scatter(
                    centroid_mapped[0, 0], centroid_mapped[0, 1],
                    color='black',
                    marker='x',
                    zorder=10,
                )
                ax_cls.text(
                    centroid_mapped[0, 0], centroid_mapped[0, 1],
                    f'({info["index"]})',
                    zorder=10,
                )
        
        fig_path = os.path.join(e.path, 'umap.png')
        fig.savefig(fig_path, dpi=300)
    
    # ~ writing concepts to disk
    # The ConceptWriter class can be used to write all the concept related information to the disk as a special 
    # self-contained data structure. 
    
    e.log('saving the concept clustering data...')
    concepts_path = os.path.join(e.path, 'concepts')
    os.mkdir(concepts_path)
    
    writer = ConceptWriter(
        path=concepts_path,
        model=model,
        processing=processing,
        logger=e.logger,
    )
    writer.write(cluster_infos)

    # ~ Save centroids as a standalone JSON file for easy downstream consumption
    centroids_list = []
    for info in cluster_infos:
        centroids_list.append({
            'index': info['index'],
            'channel': info['channel_index'],
            'centroid': info['centroid'].tolist(),
        })

    centroids_path = os.path.join(e.path, 'centroids.json')
    with open(centroids_path, 'w') as f:
        json.dump(centroids_list, f, indent=4)
    e.log(f'saved {len(centroids_list)} centroids to {centroids_path}')

    # ~ Save fitted HDBSCAN clusterers for downstream use
    # These contain the condensed tree, the raw training embeddings (at
    # ``prediction_data_.raw_data``), and prediction data needed for membership_vector()
    # on new/unseen points. Downstream tools (predict_and_score.py) reconstruct per-cluster
    # member embeddings via ``raw_data[labels_ == label]`` — no extra artefact needed.
    clusterers_path = os.path.join(e.path, 'clusterers.pkl')
    with open(clusterers_path, 'wb') as f:
        pickle.dump(channel_clusterers, f)
    e.log(f'saved {len(channel_clusterers)} fitted HDBSCAN clusterers to {clusterers_path}')

    # ~ creating the concept report
    # Based on the raw information about the extracted concept clusters we now want to generate a PDF report 
    # file which presents that information to a user in a more structured way.
    # The create_concept_cluster_report function from the visualization module can be used for this purpose. 
    # it will take the concept_infos list as input and then create a PDF report file from that in addition 
    # to other information. 
    
    e.log('creating the concept report...')
    report_path = os.path.join(e.path, 'concept_report.pdf')
    cache_path = os.path.join(e.path, 'cache')
    os.mkdir(cache_path)
    create_concept_cluster_report(
        cluster_data_list=cluster_infos,
        dataset_type=e.DATASET_TYPE,
        logger=e.logger,
        path=report_path,
        cache_path=cache_path,
        examples_type='centroid',
        num_examples=16,
        distance_func=cosine,
        normalize_centroid=True,
    )

    # ~ Per-channel reference statistics
    # Precompute per-channel pairwise distance statistics from a sample of the dataset.
    # These are used as context for the centroid distance matrix and the normalized matrix.
    # We also store the per-channel embeddings for reuse in the distribution plots below.

    PAIRWISE_SAMPLE_SIZE = 2000
    channel_stats: t.Dict[int, dict] = {}
    channel_embeddings_map: t.Dict[int, np.ndarray] = {}

    for channel_index in range(e['num_channels']):
        ch_embeddings = np.array([
            graph['graph_embedding'][:, channel_index]
            for graph in graphs
            if graph['graph_fidelity'][channel_index] > e.FIDELITY_THRESHOLD
        ])
        channel_embeddings_map[channel_index] = ch_embeddings

        # Sample for pairwise distance computation to keep it tractable
        n = len(ch_embeddings)
        if n > PAIRWISE_SAMPLE_SIZE:
            sample_idx = np.random.choice(n, size=PAIRWISE_SAMPLE_SIZE, replace=False)
            sample = ch_embeddings[sample_idx]
        else:
            sample = ch_embeddings

        pw_dists = pairwise_distances(sample, metric=e.CLUSTERING_METRIC)
        # Extract upper triangle (excluding diagonal) for statistics
        triu_idx = np.triu_indices_from(pw_dists, k=1)
        pw_values = pw_dists[triu_idx]

        channel_stats[channel_index] = {
            'mean': float(np.mean(pw_values)),
            'median': float(np.median(pw_values)),
            'std': float(np.std(pw_values)),
            'pairwise_values': pw_values,
        }
        e.log(f'channel {channel_index} pairwise distance stats: '
              f'mean={channel_stats[channel_index]["mean"]:.4f}, '
              f'median={channel_stats[channel_index]["median"]:.4f}, '
              f'std={channel_stats[channel_index]["std"]:.4f}')

    # ~ Centroid distance matrix
    # For each pair of cluster centroids we compute the pairwise distance using the configured metric
    # and annotate with per-channel reference statistics (mean/median pairwise distance).

    if len(cluster_infos) >= 2:
        e.log('creating centroid distance matrix...')
        centroids = np.array([info['centroid'] for info in cluster_infos])
        centroid_dist_matrix = pairwise_distances(centroids, metric=e.CLUSTERING_METRIC)

        cluster_labels = [f"ch{info['channel_index']}_cl{info['index']}" for info in cluster_infos]

        # Build a subtitle with per-channel reference stats
        stats_lines = []
        for ch_idx in sorted(channel_stats.keys()):
            s = channel_stats[ch_idx]
            stats_lines.append(f'Ch{ch_idx} pairwise: mean={s["mean"]:.3f}, median={s["median"]:.3f}, std={s["std"]:.3f}')

        fig_matrix, ax_matrix = plt.subplots(
            figsize=(max(8, len(cluster_infos)), max(6, len(cluster_infos) * 0.8)),
        )
        im = ax_matrix.imshow(centroid_dist_matrix, cmap='coolwarm')
        ax_matrix.set_xticks(range(len(cluster_labels)))
        ax_matrix.set_xticklabels(cluster_labels, rotation=45, ha='right', fontsize=8)
        ax_matrix.set_yticks(range(len(cluster_labels)))
        ax_matrix.set_yticklabels(cluster_labels, fontsize=8)
        ax_matrix.set_title(
            f'Centroid Distance Matrix ({e.CLUSTERING_METRIC})\n'
            + '\n'.join(stats_lines),
            fontsize=9,
        )

        for i in range(len(cluster_labels)):
            for j in range(len(cluster_labels)):
                ax_matrix.text(j, i, f'{centroid_dist_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=6,
                               color='black')

        fig_matrix.colorbar(im)
        fig_matrix.tight_layout()
        fig_matrix.savefig(os.path.join(e.path, 'centroid_distance_matrix.png'), dpi=150)
        plt.close(fig_matrix)

    # ~ Normalized centroid distance matrix
    # Same matrix but each entry is expressed in units of standard deviations of the overall
    # pairwise distance distribution for the corresponding channel pair.

    if len(cluster_infos) >= 2:
        e.log('creating normalized centroid distance matrix...')
        normalized_matrix = np.zeros_like(centroid_dist_matrix)
        for i, info_i in enumerate(cluster_infos):
            for j, info_j in enumerate(cluster_infos):
                # Use the channel of the row cluster for normalization;
                # for cross-channel pairs, average both channels' stats.
                ch_i = info_i['channel_index']
                ch_j = info_j['channel_index']
                if ch_i == ch_j:
                    std = channel_stats[ch_i]['std']
                    mean = channel_stats[ch_i]['mean']
                else:
                    std = (channel_stats[ch_i]['std'] + channel_stats[ch_j]['std']) / 2
                    mean = (channel_stats[ch_i]['mean'] + channel_stats[ch_j]['mean']) / 2

                if std > 0:
                    normalized_matrix[i, j] = (centroid_dist_matrix[i, j] - mean) / std
                else:
                    normalized_matrix[i, j] = 0.0

        fig_norm, ax_norm = plt.subplots(
            figsize=(max(8, len(cluster_infos)), max(6, len(cluster_infos) * 0.8)),
        )
        im_norm = ax_norm.imshow(normalized_matrix, cmap='coolwarm')
        ax_norm.set_xticks(range(len(cluster_labels)))
        ax_norm.set_xticklabels(cluster_labels, rotation=45, ha='right', fontsize=8)
        ax_norm.set_yticks(range(len(cluster_labels)))
        ax_norm.set_yticklabels(cluster_labels, fontsize=8)
        ax_norm.set_title(
            f'Normalized Centroid Distance Matrix ({e.CLUSTERING_METRIC})\n'
            f'Values in std deviations from mean pairwise distance',
            fontsize=9,
        )

        for i in range(len(cluster_labels)):
            for j in range(len(cluster_labels)):
                ax_norm.text(j, i, f'{normalized_matrix[i, j]:.2f}σ',
                             ha='center', va='center', fontsize=6,
                             color='black')

        fig_norm.colorbar(im_norm)
        fig_norm.tight_layout()
        fig_norm.savefig(os.path.join(e.path, 'centroid_distance_matrix_normalized.png'), dpi=150)
        plt.close(fig_norm)

    # ~ Per-channel pairwise distance reference distribution
    # For each channel, plot the sampled pairwise distance distribution as a reference for the
    # overall scale of the embedding space.

    e.log('creating per-channel pairwise distance reference distributions...')
    for channel_index in range(e['num_channels']):
        stats = channel_stats[channel_index]
        pw_values = stats['pairwise_values']

        fig_ref, ax_ref = plt.subplots(figsize=(8, 5))
        ax_ref.hist(pw_values, bins=50, color='lightgray', edgecolor='black', alpha=0.7)
        ax_ref.axvline(stats['mean'], color='blue', linestyle='--', label=f'mean={stats["mean"]:.3f}')
        ax_ref.axvline(stats['median'], color='green', linestyle='--', label=f'median={stats["median"]:.3f}')

        # Mark centroid distances for clusters in this channel
        ch_infos = [info for info in cluster_infos if info['channel_index'] == channel_index]
        for i, info_a in enumerate(ch_infos):
            for j, info_b in enumerate(ch_infos):
                if j <= i:
                    continue
                d = pairwise_distances(
                    np.expand_dims(info_a['centroid'], axis=0),
                    np.expand_dims(info_b['centroid'], axis=0),
                    metric=e.CLUSTERING_METRIC,
                )[0, 0]
                ax_ref.axvline(d, color='red', linestyle='-', linewidth=2,
                               label=f'cl{info_a["index"]}<->cl{info_b["index"]}={d:.3f}')

        ax_ref.set_xlabel(f'Distance ({e.CLUSTERING_METRIC})')
        ax_ref.set_ylabel('Count')
        ax_ref.set_title(
            f'Pairwise Distance Distribution — Channel {channel_index} '
            f'({e.CHANNEL_INFOS[channel_index]["name"]})\n'
            f'(sampled {min(len(channel_embeddings_map[channel_index]), PAIRWISE_SAMPLE_SIZE)} elements)',
            fontsize=9,
        )
        ax_ref.legend(fontsize=8)
        fig_ref.tight_layout()
        fig_ref.savefig(os.path.join(e.path, f'pairwise_distance_distribution__ch{channel_index}.png'), dpi=150)
        plt.close(fig_ref)

    # ~ Per-cluster distance distributions
    # For each cluster we compute the distance from every element in the dataset (that passes the
    # fidelity threshold for the cluster's channel) to the cluster centroid and plot the distribution
    # as a histogram. Distances to other centroids (especially within the same channel) are marked
    # as vertical lines.

    e.log('creating per-cluster score distributions...')

    # We store per-cluster scores and intra/inter splits for reuse in the summary analysis below.
    cluster_score_data: t.List[dict] = []

    for info in cluster_infos:
        ch_idx = info['channel_index']
        cl_idx = info['index']
        centroid = info['centroid']
        hdbscan_label = info.get('hdbscan_label')

        all_channel_emb = channel_embeddings_map[ch_idx]
        ch_labels = channel_labels.get(ch_idx)

        # Compute scores from all fidelity-passing elements to this cluster via the hook
        scores = e.apply_hook(
            'compute_cluster_score',
            embeddings=all_channel_emb,
            cluster_info=info,
            channel_clusterers=channel_clusterers,
        )

        # Split into intra-cluster (elements belonging to this cluster) and inter-cluster (rest)
        if ch_labels is not None and hdbscan_label is not None:
            intra_mask = (ch_labels == hdbscan_label)
        else:
            intra_mask = np.zeros(len(scores), dtype=bool)

        intra_scores = scores[intra_mask]
        inter_scores = scores[~intra_mask]

        cluster_score_data.append({
            'info': info,
            'scores': scores,
            'intra_scores': intra_scores,
            'inter_scores': inter_scores,
        })

        # -- Per-cluster score distribution plot (existing)
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        ax_dist.hist(scores, bins=50, color=info['color'], edgecolor='black', alpha=0.7)
        ax_dist.set_xlabel('Score (lower = better fit)')
        ax_dist.set_ylabel('Count')
        ax_dist.set_title(f'Score Distribution to Cluster {cl_idx} (Channel {ch_idx} - {info["name"]})')
        ax_dist.axvline(np.median(scores), color='red', linestyle='--', label=f'median={np.median(scores):.2f}')

        # Mark distances from other centroids to this cluster's centroid
        for other_info in cluster_infos:
            if other_info['index'] == cl_idx:
                continue
            d = pairwise_distances(
                np.expand_dims(centroid, axis=0),
                np.expand_dims(other_info['centroid'], axis=0),
                metric=e.CLUSTERING_METRIC,
            )[0, 0]
            is_same_channel = other_info['channel_index'] == ch_idx
            ax_dist.axvline(
                d,
                color='darkblue' if is_same_channel else 'gray',
                linestyle='-' if is_same_channel else ':',
                linewidth=2 if is_same_channel else 1,
                alpha=1.0 if is_same_channel else 0.5,
                label=f'{"→" if is_same_channel else "⇢"}cl{other_info["index"]}={d:.3f}',
            )

        ax_dist.legend(fontsize=8)
        fig_dist.tight_layout()
        fig_dist.savefig(os.path.join(e.path, f'distance_distribution__ch{ch_idx}_cl{cl_idx}.png'), dpi=150)
        plt.close(fig_dist)

    # ~ Intra vs inter cluster score analysis
    # For each cluster, compare the score distribution of elements that HDBSCAN assigned to that
    # cluster (intra) versus all other fidelity-passing elements in the same channel (inter).
    # This uses whatever scoring method the compute_cluster_score hook provides.

    if len(cluster_score_data) > 0:
        e.log('creating intra vs inter cluster score analysis...')

        # -- Per-cluster overlaid histograms
        for csd in cluster_score_data:
            info = csd['info']
            ch_idx = info['channel_index']
            cl_idx = info['index']
            intra = csd['intra_scores']
            inter = csd['inter_scores']

            intra_mean = float(np.mean(intra)) if len(intra) > 0 else 0.0
            intra_std = float(np.std(intra)) if len(intra) > 0 else 0.0
            inter_mean = float(np.mean(inter)) if len(inter) > 0 else 0.0
            inter_std = float(np.std(inter)) if len(inter) > 0 else 0.0
            ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')
            gap = inter_mean - intra_mean

            # Store metrics
            e[f'intra_inter/{cl_idx}/intra_mean'] = intra_mean
            e[f'intra_inter/{cl_idx}/intra_std'] = intra_std
            e[f'intra_inter/{cl_idx}/inter_mean'] = inter_mean
            e[f'intra_inter/{cl_idx}/inter_std'] = inter_std
            e[f'intra_inter/{cl_idx}/ratio'] = ratio
            e[f'intra_inter/{cl_idx}/gap'] = gap
            e[f'intra_inter/{cl_idx}/n_intra'] = int(len(intra))
            e[f'intra_inter/{cl_idx}/n_inter'] = int(len(inter))

            e.log(f'  cluster {cl_idx} (ch{ch_idx}): '
                  f'intra={intra_mean:.4f}±{intra_std:.4f} (n={len(intra)}), '
                  f'inter={inter_mean:.4f}±{inter_std:.4f} (n={len(inter)}), '
                  f'ratio={ratio:.2f}, gap={gap:.4f}')

            fig_ii, ax_ii = plt.subplots(figsize=(8, 5))

            # Determine shared bin edges across both distributions
            all_vals = np.concatenate([intra, inter]) if len(intra) > 0 else inter
            bins = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), 50)

            if len(intra) > 0:
                ax_ii.hist(intra, bins=bins, color='green', alpha=0.6, edgecolor='darkgreen',
                           label=f'Intra (n={len(intra)}, μ={intra_mean:.3f})')
                ax_ii.axvline(intra_mean, color='green', linestyle='--', linewidth=2)

            if len(inter) > 0:
                ax_ii.hist(inter, bins=bins, color='red', alpha=0.4, edgecolor='darkred',
                           label=f'Inter (n={len(inter)}, μ={inter_mean:.3f})')
                ax_ii.axvline(inter_mean, color='red', linestyle='--', linewidth=2)

            ax_ii.set_xlabel('Score (lower = better fit)')
            ax_ii.set_ylabel('Count')
            ax_ii.set_title(
                f'Intra vs Inter Cluster Scores — Cluster {cl_idx} (Ch {ch_idx} - {info["name"]})\n'
                f'Gap={gap:.3f}  Ratio={ratio:.2f}'
            )
            ax_ii.legend(fontsize=9)
            fig_ii.tight_layout()
            fig_ii.savefig(os.path.join(e.path, f'intra_inter__ch{ch_idx}_cl{cl_idx}.png'), dpi=150)
            plt.close(fig_ii)

        # -- Summary grouped bar chart across all clusters
        n_clusters = len(cluster_score_data)
        cluster_labels_bar = [f"ch{csd['info']['channel_index']}_cl{csd['info']['index']}" for csd in cluster_score_data]
        intra_means = [float(np.mean(csd['intra_scores'])) if len(csd['intra_scores']) > 0 else 0.0
                       for csd in cluster_score_data]
        intra_stds = [float(np.std(csd['intra_scores'])) if len(csd['intra_scores']) > 0 else 0.0
                      for csd in cluster_score_data]
        inter_means = [float(np.mean(csd['inter_scores'])) if len(csd['inter_scores']) > 0 else 0.0
                       for csd in cluster_score_data]
        inter_stds = [float(np.std(csd['inter_scores'])) if len(csd['inter_scores']) > 0 else 0.0
                      for csd in cluster_score_data]

        x = np.arange(n_clusters)
        bar_width = 0.35

        fig_summary, ax_summary = plt.subplots(figsize=(max(8, n_clusters * 2), 5))
        bars_intra = ax_summary.bar(x - bar_width / 2, intra_means, bar_width,
                                     yerr=intra_stds, capsize=4,
                                     color='green', alpha=0.7, label='Intra-cluster')
        bars_inter = ax_summary.bar(x + bar_width / 2, inter_means, bar_width,
                                     yerr=inter_stds, capsize=4,
                                     color='red', alpha=0.7, label='Inter-cluster')

        # Annotate ratios above each group
        for i in range(n_clusters):
            r = inter_means[i] / intra_means[i] if intra_means[i] > 0 else float('inf')
            y_max = max(inter_means[i] + inter_stds[i], intra_means[i] + intra_stds[i])
            ax_summary.text(x[i], y_max + 0.02, f'{r:.2f}x',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax_summary.set_xlabel('Cluster')
        ax_summary.set_ylabel('Mean Score (lower = better fit)')
        ax_summary.set_title('Intra vs Inter Cluster Scores — Summary')
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels(cluster_labels_bar, fontsize=9)
        ax_summary.legend()
        fig_summary.tight_layout()
        fig_summary.savefig(os.path.join(e.path, 'intra_inter_summary.png'), dpi=150)
        plt.close(fig_summary)

        e.log(f'created intra vs inter analysis for {n_clusters} clusters')

    # ~ Post-hoc hierarchical clustering of clusters (per channel)
    # Agglomerative hierarchical clustering run over the already-found leaf clusters,
    # one dendrogram per explanation channel. The inter-cluster distance between any
    # two clusters is the mean pairwise distance between all their members under
    # CLUSTERING_METRIC (average linkage over the full member point clouds). This
    # exposes super-cluster structure — e.g. multiple leaf clusters that all capture
    # the same functional motif in different local contexts — without requiring any
    # predefined SMARTS patterns. Scoring is not affected; this is an analysis-only
    # artefact controlled by ANALYZE_CLUSTER_HIERARCHY.

    cluster_hierarchy: t.Dict[int, np.ndarray] = {}

    if getattr(e, 'ANALYZE_CLUSTER_HIERARCHY', False) and len(cluster_infos) >= 2:
        e.log('running hierarchical clustering over leaf clusters...')

        channel_to_clusters: t.Dict[int, t.List[dict]] = {}
        for info in cluster_infos:
            channel_to_clusters.setdefault(info['channel_index'], []).append(info)

        for ch_idx, infos in channel_to_clusters.items():
            if len(infos) < 2:
                continue

            infos_sorted = sorted(infos, key=lambda i: i['index'])
            K = len(infos_sorted)

            inter = np.zeros((K, K), dtype=float)
            for i in range(K):
                for j in range(i + 1, K):
                    d = pairwise_distances(
                        infos_sorted[i]['embeddings'],
                        infos_sorted[j]['embeddings'],
                        metric=e.CLUSTERING_METRIC,
                    ).mean()
                    inter[i, j] = d
                    inter[j, i] = d

            condensed = squareform(inter, checks=False)
            Z = linkage(condensed, method=e.CLUSTER_HIERARCHY_LINKAGE)
            cluster_hierarchy[ch_idx] = Z

            leaf_labels = [
                f"cl{info['index']} (n={len(info['embeddings'])})"
                for info in infos_sorted
            ]

            fig_h, ax_h = plt.subplots(figsize=(max(8, K * 0.9), 5))
            dendrogram(Z, labels=leaf_labels, ax=ax_h, leaf_rotation=45, color_threshold=0)
            ax_h.set_title(
                f'Cluster Hierarchy — Channel {ch_idx} '
                f"({e.CHANNEL_INFOS[ch_idx]['name']})\n"
                f'linkage={e.CLUSTER_HIERARCHY_LINKAGE}, metric={e.CLUSTERING_METRIC}'
            )
            ax_h.set_ylabel(f'Mean pairwise distance ({e.CLUSTERING_METRIC})')
            fig_h.tight_layout()
            fig_h.savefig(os.path.join(e.path, f'dendrogram_ch{ch_idx}.png'), dpi=150)
            plt.close(fig_h)

            max_d = float(Z[:, 2].max()) if len(Z) > 0 else 0.0
            for frac in np.arange(0.1, 0.91, 0.1):
                t_cut = float(frac) * max_d
                grouping = fcluster(Z, t=t_cut, criterion='distance')
                groups: t.Dict[int, t.List[int]] = {}
                for leaf_idx, g in enumerate(grouping):
                    groups.setdefault(int(g), []).append(infos_sorted[leaf_idx]['index'])
                summary = ' | '.join(
                    '[' + ', '.join(f'cl{ci}' for ci in sorted(members)) + ']'
                    for members in groups.values()
                )
                e.log(f'  ch{ch_idx} cut={frac:.0%} (d={t_cut:.4f}): {summary}')

        hierarchy_path = os.path.join(e.path, 'cluster_hierarchy.pkl')
        with open(hierarchy_path, 'wb') as f:
            pickle.dump(cluster_hierarchy, f)
        e.log(f'saved cluster hierarchy ({len(cluster_hierarchy)} channel(s)) to {hierarchy_path}')

    # ~ Sample element distance heatmap
    # Pick a configurable number of random elements from the dataset and compute their distances
    # to every cluster centroid/medoid. Visualized as a heatmap with rows sorted by nearest cluster.

    if e.NUM_SAMPLE_ELEMENTS > 0 and len(cluster_infos) > 0:
        e.log(f'creating sample element score heatmap ({e.NUM_SAMPLE_ELEMENTS} elements)...')

        n_samples = min(e.NUM_SAMPLE_ELEMENTS, len(graphs))
        sample_indices = np.random.choice(len(graphs), size=n_samples, replace=False)
        sample_graphs = [graphs[i] for i in sample_indices]

        # Compute scores from each sampled element to each cluster via the hook.
        # membership_vector() handles arbitrary points (even unseen ones), so no
        # fidelity-position mapping is needed.
        cluster_labels_hm = [f"ch{info['channel_index']}_cl{info['index']}" for info in cluster_infos]
        dist_data = np.zeros((n_samples, len(cluster_infos)))

        for j, info in enumerate(cluster_infos):
            ch_idx = info['channel_index']
            sample_embeddings = np.array([g['graph_embedding'][:, ch_idx] for g in sample_graphs])

            dist_data[:, j] = e.apply_hook(
                'compute_cluster_score',
                embeddings=sample_embeddings,
                cluster_info=info,
                channel_clusterers=channel_clusterers,
            )

        # Optional sharpening: for each sample row, sharpen the distances per
        # channel (clusters from different channels are not directly comparable)
        # and gate by that sample's per-channel fidelity against FIDELITY_THRESHOLD.
        # Below-threshold channels get their cells set to 1.0 (the max sharpened
        # "distance"), keeping the whole heatmap on a consistent [0, 1] scale.
        if e.SHARPEN_SCORES:
            e.log(f'sharpening heatmap scores ({e.SHARPEN_METHOD}, gate=fidelity>={e.FIDELITY_THRESHOLD})')
            channel_to_cols: t.Dict[int, t.List[int]] = {}
            for j, info in enumerate(cluster_infos):
                channel_to_cols.setdefault(info['channel_index'], []).append(j)
            for i, g in enumerate(sample_graphs):
                fidelity = np.asarray(g.get('graph_fidelity', np.zeros(e['num_channels'])))
                for ch_idx, cols in channel_to_cols.items():
                    if float(fidelity[ch_idx]) < e.FIDELITY_THRESHOLD:
                        dist_data[i, cols] = 1.0  # not assigned to any cluster
                        continue
                    raw = dist_data[i, cols]
                    dist_data[i, cols] = sharpen_scores(
                        raw,
                        method=e.SHARPEN_METHOD,
                        temperature=e.SHARPEN_TEMPERATURE,
                    )

        # Sort rows by their nearest cluster for visual grouping
        nearest_cluster = np.argmin(dist_data, axis=1)
        sort_order = np.lexsort((dist_data[np.arange(n_samples), nearest_cluster], nearest_cluster))
        dist_data_sorted = dist_data[sort_order]
        sample_indices_sorted = sample_indices[sort_order]
        sample_graphs_sorted = [sample_graphs[i] for i in sort_order]
        nearest_cluster_sorted = nearest_cluster[sort_order]

        # Build row labels (use graph_repr/SMILES if available, otherwise dataset index)
        row_labels = []
        for idx in sample_indices_sorted:
            repr_str = graphs[idx].get('graph_repr', '')
            if repr_str and len(repr_str) <= 30:
                row_labels.append(f'{idx}: {repr_str}')
            elif repr_str:
                row_labels.append(f'{idx}: {repr_str[:27]}...')
            else:
                row_labels.append(str(idx))

        # Fidelity check for styling: rows whose nearest-cluster channel has
        # fidelity >= FIDELITY_THRESHOLD are rendered bold ("trustworthy
        # assignment"); those below threshold are italic ("likely noise").
        row_above_threshold: t.List[bool] = []
        for i, g in enumerate(sample_graphs_sorted):
            nearest_ch = cluster_infos[int(nearest_cluster_sorted[i])]['channel_index']
            fidelity = np.asarray(g.get('graph_fidelity', np.zeros(e['num_channels'])))
            row_above_threshold.append(float(fidelity[nearest_ch]) >= e.FIDELITY_THRESHOLD)

        fig_hm, ax_hm = plt.subplots(
            figsize=(max(8, len(cluster_infos) * 1.5), max(8, n_samples * 0.35)),
        )
        im_hm = ax_hm.imshow(dist_data_sorted, cmap='coolwarm', aspect='auto')

        ax_hm.set_xticks(range(len(cluster_labels_hm)))
        ax_hm.set_xticklabels(cluster_labels_hm, rotation=45, ha='right', fontsize=8)
        ax_hm.set_yticks(range(n_samples))
        y_tick_labels = ax_hm.set_yticklabels(row_labels, fontsize=7)
        for tick, above in zip(y_tick_labels, row_above_threshold):
            if above:
                tick.set_fontweight('bold')
            else:
                tick.set_fontstyle('italic')
        ax_hm.set_title(
            f'Sample Element Scores to Cluster Representatives\n'
            f'Sorted by nearest cluster — representative: {e.CLUSTER_REPRESENTATIVE}',
            fontsize=9,
        )

        # Annotate values and mark the minimum per row
        for i in range(n_samples):
            min_j = np.argmin(dist_data_sorted[i])
            for j in range(len(cluster_infos)):
                fontweight = 'bold' if j == min_j else 'normal'
                ax_hm.text(j, i, f'{dist_data_sorted[i, j]:.2f}',
                           ha='center', va='center', fontsize=6,
                           fontweight=fontweight, color='black')

        fig_hm.colorbar(im_hm)
        fig_hm.tight_layout()
        fig_hm.savefig(os.path.join(e.path, 'sample_element_distances.png'), dpi=150)
        plt.close(fig_hm)

    # ~ Export Clustering archive (.clu)
    # Build a self-contained Clustering object from the in-memory experiment data and
    # save it as a .clu zip archive. This single file replaces the separate
    # clusterers.pkl / centroids.json / cluster_hierarchy.pkl artefacts for downstream
    # consumption (predict_and_score.py, interactive visualizations).
    from megan_global_explanations.data import Clustering as ClusteringData

    clustering_obj = ClusteringData.from_experiment(
        cluster_infos=cluster_infos,
        channel_infos=e.CHANNEL_INFOS,
        clustering_metric=e.CLUSTERING_METRIC,
        channel_embeddings_map=channel_embeddings_map,
        channel_labels=channel_labels,
        cluster_hierarchy=cluster_hierarchy or None,
    )
    clustering_clu_path = os.path.join(e.path, 'clustering.clu')
    clustering_obj.save(clustering_clu_path)
    e.log(f'saved Clustering archive to {clustering_clu_path}')

    # :hook post_clustering:
    #       This hook is called at the very end of the experiment after all clustering, prototype optimization,
    #       and report generation is complete. It receives all the relevant data structures and can be used
    #       to perform additional custom processing or export tasks.
    e.apply_hook(
        'post_clustering',
        cluster_infos=cluster_infos,
        graphs=graphs,
        indices=indices,
        model=model,
        processing=processing,
        index_data_map=index_data_map,
        channel_clusterers=channel_clusterers,
        channel_labels=channel_labels,
    )


# :hook compute_cluster_score:
#       Compute scores indicating how well elements fit a given cluster. Returns an (N,)
#       array of scores where lower values mean better fit (like a distance). Override this
#       hook to use alternative scoring methods such as HDBSCAN membership probability.
#       The hook receives ``channel_clusterers`` (dict of fitted HDBSCAN objects per channel)
#       which can be used with ``hdbscan.prediction.membership_vector()`` to compute soft
#       cluster membership for any set of embeddings.
@experiment.hook('compute_cluster_score', default=True, replace=False)
def compute_cluster_score(e: Experiment,
                          embeddings: np.ndarray,
                          cluster_info: dict,
                          **kwargs) -> np.ndarray:
    """
    Default: pairwise distance to the cluster centroid/medoid.
    Convention: lower score = better fit.
    """
    centroid = cluster_info['centroid']
    return pairwise_distances(
        embeddings, centroid.reshape(1, -1), metric=e.CLUSTERING_METRIC
    ).flatten()


@experiment.hook('post_clustering', default=False, replace=False)
def post_clustering(e: Experiment, **kwargs) -> None:
    """
    Default implementation of post_clustering hook - does nothing.
    Override in sub-experiments to add custom post-processing.
    """
    pass


@experiment.analysis
def analysis(e: Experiment):
    
    return
    e.log('starting analysis...')
    
    # ~ loading everything
    # Before we can do any analysis we need to load all the actual data that was created or used during the experiment
    # we mainly want to load the persistent representation of the the extracted concepts. However, this requires that 
    # we also load the dataset and the model first!
    
    e.log('loading the dataset...')
    dataset_path = e.apply_hook('get_dataset_path')
    index_data_map, processing = e.apply_hook(
        'load_dataset',
        path=dataset_path,
    )
    e.log(f'loaded dataset with {len(index_data_map)} elements...')
    
    e.log('loading model...')
    model = e.apply_hook(
        'load_model',
        path=e.MODEL_PATH,
    ) 
    e.log(f'loaded model of type {model.__class__.__name__}')

    e.log('loading concepts...')
    concepts_path = os.path.join(e.path, 'concepts')
    reader = ConceptReader(
        path=concepts_path,
        model=model,
        dataset=index_data_map,
        logger=e.logger,
    )
    concepts = reader.read()
    e.log(f'loaded {len(concepts)} concepts...')
    
    for concept in concepts:
        concept['image_paths'] = [element['image_path'] for element in concept['elements']]
        concept['graphs'] = [element['metadata']['graph'] for element in concept['elements']]
    
    # ~ creating the concept report
    # After everything is loaded we can then create the concept report PDF itself from the loaded 
    # concepts
    
    # report_path = os.path.join(e.path, 'concept_report.pdf')
    # cache_path = os.path.join(e.path, 'cache')
    # create_concept_cluster_report(
    #     cluster_data_list=concepts,
    #     dataset_type=e.DATASET_TYPE,
    #     logger=e.logger,
    #     path=report_path,
    #     cache_path=cache_path,
    #     examples_type='centroid',
    #     num_examples=16,
    #     distance_func=cosine,
    #     normalize_centroid=True,
    # )

experiment.run_if_main()