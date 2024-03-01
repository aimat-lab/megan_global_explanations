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
from scipy.spatial.distance import cosine
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
PLOT_UMAP: bool = True

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
    model = Megan.load_from_checkpoint(path)
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
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    indices = np.array([index for index in index_data_map.keys()])
    
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
    infos = model.forward_graphs(graphs) 
    # We also want to calculate the loo devations aka the channel-specific fidelity values as those will 
    # be part of the criterium that we will use to filter the relevant concept clusters.
    deviations = model.leave_one_out_deviations(graphs)
    
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
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=e.MIN_CLUSTER_SIZE,
            min_samples=e.MIN_SAMPLES,
            metric='manhattan',
            cluster_selection_method=e.CLUSTER_SELECTION_METHOD,
        )
        # labels: (B, )
        # This is an array that contains the cluster indices for every element of the dataset. It assigns an integer 
        # cluster index to each element, where -1 is a special index indicating that an element does not belong to 
        # any cluster.
        labels = clusterer.fit_predict(graph_embeddings)
        # A list of all the possible cluster indices from which we can derive how many clusters there have been found 
        # in general.
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
                'embeddings':           cluster_graph_embeddings,
                'centroid':             cluster_centroid,
                'index_tuples':         cluster_index_tuples,
                'elements':             cluster_elements,
                'graphs':               cluster_graphs,
                'image_paths':          cluster_image_paths,
                'name':                 e.CHANNEL_INFOS[channel_index]['name'],
                'color':                e.CHANNEL_INFOS[channel_index]['color'],
            }
            
            e.log(f' ({cluster}/{num_clusters}) - contribution: {cluster_contribution:.2f}')
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
                    metric='manhattan',
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
                metric='manhattan',
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
    
    
    
    
@experiment.analysis
def analysis(e: Experiment):
    
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
    
    report_path = os.path.join(e.path, 'concept_report.pdf')
    cache_path = os.path.join(e.path, 'cache')
    create_concept_cluster_report(
        cluster_data_list=concepts,
        dataset_type=e.DATASET_TYPE,
        logger=e.logger,
        path=report_path,
        cache_path=cache_path,
        examples_type='centroid',
        num_examples=16,
        distance_func=cosine,
        normalize_centroid=True,
    )

experiment.run_if_main()