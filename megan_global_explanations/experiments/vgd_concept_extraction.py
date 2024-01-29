"""
This experiment can be used to generate a concept extraction report for a visual graph dataset 
and an already trained MEGAN model. 


"""
import os
import random
import pathlib
import typing as t
from collections import defaultdict

import hdbscan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.config import Config
from visual_graph_datasets.graph import graph_expand_mask
from visual_graph_datasets.graph import graph_find_connected_regions
from visual_graph_datasets.graph import extract_subgraph
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
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

mpl.use('Agg')


PATH = pathlib.Path(__file__).parent.absolute()

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
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'rb_dual_motifs.ckpt')

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


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset', replace=False)
def load_dataset(e: Experiment,
                 path: str,
                 ) -> dict:
    
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
    
    # For the embedding objective function we need some kind of anchor location to which we want to 
    # minimize the distance. For this we are simply going to use the cluster centroid.
    anchor = np.mean(cluster_embeddings, axis=0)
    
    # In this section we assemble the initial population for the optimization of the prototype. In fact, 
    # in this use case, it is possible to already assemble a very good initial population by simply using 
    # the subgraphs which are already highlighted by the explanation masks of the cluster members. These 
    # should by themselves already be very close to the ideal cluster prototype and most likely only 
    # need minor refinements through the GA optimization.
    num_initial = min(e.INITIAL_POPULATION_SAMPLE, len(cluster_embeddings))
    elements_initial = []
    for graph in random.sample(cluster_graphs, k=num_initial):
        
        # In the very first step we need to binarize the node explanation mask by using a simple 
        # threshold.
        node_mask = (graph['node_importances'][:, channel_index] > 0.5).astype(int)
        
        # Since the explanation mask tends to be too sparse in many situations we perform a one-hop expansion 
        # of this mask here. So that function will propagate the mask label to all the nodes that are currently 
        # adjacent to at least one other mask node.
        node_mask = graph_expand_mask(graph, node_mask, num_hops=2)
        
        # Then we only want to extract connected subgraphs. But by the explanation mask alone it is not certain 
        # that all the masked nodes are connected. So here we use this function which determines all the 
        # connected regions for the masked part of the graph and ultimately decide to only use the largest 
        # of those.
        region_mask = graph_find_connected_regions(graph, node_mask)
        regions = [v for v in set(region_mask) if v > -1]
        # Although it should not happen, it is still possible that the explanation mask for individual 
        # elements are empty which will also lead this regions list to be empty. To avoid errors we 
        # need to skip the elements for which that is the case.
        if not regions:
            continue
        
        # Here we sort the regions descending by size so that we can later select only the largest 
        # region for the subgraph extraction.
        regions.sort(key=lambda r: np.sum((region_mask == r).astype(int)), reverse=True)

        mask = (region_mask == regions[0]).astype(int)
        if np.sum(mask) < 2:
            continue
        
        sub_graph, _ , _ = extract_subgraph(graph, mask)
    
        elements_initial.append({
            'graph': sub_graph,
            'value': processing.unprocess(sub_graph)
        })
    
    # This function will execute the actual genetic optimization algorithm to create graph prototypes
    # that are as close to the cluster centroid (==anchor) as possible.
    element, history = genetic_optimize(
        fitness_func=lambda graphs: embedding_distance_fitness(
            graphs=graphs,
            model=model,
            channel_index=channel_index,
            anchor=anchor,
            node_factor=0.002,
            edge_factor=0.002,
        ),
        sample_func=lambda: random.choice(elements_initial),
        mutation_funcs=[
            #mutate_add_node,
            mutate_modify_node,
            mutate_remove_node,
            mutate_remove_node,
            #mutate_add_edge,
            mutate_remove_edge,
        ],
        num_epochs=50,
        population_size=1_000,
        elite_ratio=0.1,
        refresh_ratio=0.2,
        logger=e.logger,
    )
    
    return element


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    # ~ loading the dataset
    # The dataset is either loaded from the local file system as a path or it is downloaded 
    # from the remote file share first by providing it's unique string identifier.
    
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
    
    e.log('loading dataset...')
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
    for graph, info, dev in zip(graphs, infos, deviations):
        
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
            graph['graph_fidelity'] = np.diag(dev)
    
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
            metric=cosine,
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
            # cluster_indices: (B_cluster, )
            cluster_indices = channel_indices[mask]
            cluster_index_tuples = [(i, channel_index) for i in cluster_indices]
            cluster_graphs = [index_data_map[i]['metadata']['graph'] for i in cluster_indices]
            cluster_image_paths = [index_data_map[i]['image_path'] for i in cluster_indices]
            
            info = {
                'index':                cluster_index,
                'embeddings':           cluster_graph_embeddings,
                'index_tuples':         cluster_index_tuples,
                'graphs':               cluster_graphs,
                'image_paths':          cluster_image_paths,
                'name':                 e.CHANNEL_INFOS[channel_index]['name'],
                'color':                e.CHANNEL_INFOS[channel_index]['color'],
            }
            
            # Optionally it is also possible to derive an approximation for the prototype of the cluster by doing 
            # an optimization scheme. However, this will require quite some time so it can be skipped as well.
            if e.OPTIMIZE_CLUSTER_PROTOTYPE:
                e.log(f' ({cluster}/{num_clusters}) optimizing prototype...')
                cluster_prototype: dict = e.apply_hook(
                    'optimize_prototype',
                    model=model,
                    channel_index=channel_index,
                    processing=processing,
                    cluster_graphs=cluster_graphs,
                    cluster_embeddings=cluster_graph_embeddings,
                )

                fig, _ = processing.visualize_as_figure(
                    value=cluster_prototype['value'],
                    graph=cluster_prototype['graph'],
                    width=1000,
                    height=1000,
                )
                prototype_path = os.path.join(e.path, f'prototype__cl{cluster_index:02d}.png')
                fig.savefig(prototype_path)
                
                info['prototype'] = {
                    'path': prototype_path,
                    'description': 'No description generated.', 
                }
            
            cluster_index += 1
            cluster_infos.append(info)
            
    # We definitely want to store the cluster infos to the experiment storage so that we can access them 
    # later on during the analysis as well.
    #e['cluster_infos'] = cluster_infos
    
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
    )


experiment.run_if_main()