import os
import logging
import typing as t
from collections import defaultdict

import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from sklearn.metrics import pairwise_distances
from graph_attention_student.torch.megan import Megan
from graph_attention_student.utils import array_normalize
from visual_graph_datasets.processing.base import ProcessingBase

from megan_global_explanations.utils import NULL_LOGGER


DEFAULT_CHANNEL_INFOS = defaultdict(lambda: {
    'name': 'channel',
    'color': 'lightgray',
})


def extract_concepts(model: Megan,
                     index_data_map: t.Dict[int, dict],
                     processing: ProcessingBase,
                     dataset_type: t.Literal['regresssion', 'classification'] = 'regression',
                     fidelity_threshold: float = 0.0,
                     min_samples: int = 0,
                     min_cluster_size: int = 0,
                     cluster_metric: str = 'manhattan',
                     cluster_selection_method: str = 'leaf',
                     channel_infos: t.Dict[int, dict] = DEFAULT_CHANNEL_INFOS,
                     optimize_prototypes: bool = False,
                     sort_similarity: bool = True,
                     logger: logging.Logger = NULL_LOGGER,
                     ) -> t.Dict[int, dict]:
    """
    This function uses the given MEGAN ``model``, the dataset in the format of the given ``index_data_map`` 
    to extract the concept explanations based on the models latent space representations.
    
    The concepts are returned in the format of a list of dictionaries, where every dict contains all the 
    relevant information about the concept.
    """
    num_channels = model.num_channels

    # ~ updating the dataset
    # In the first step we put the entire dataset through the model to obtain all the model outputs for all 
    # the elements of the dataset. This includes the output predictions, the explanation masks, the fidelity 
    # values, but also the latent space representations of the explanation channels.
    
    indices = list(index_data_map.keys())
    graphs = [index_data_map[index]['metadata']['graph'] for index in indices]
    
    logger.info(f'running model forward pass for the dataset with {len(graphs)} elements...')
    infos = model.forward_graphs(graphs)
    devs = model.leave_one_out_deviations(graphs)
    
    # We are attaching all this additional information that we obtain from the dataset here as additional 
    # attributes of the graphs dict objects themselves so that later on all the necessary information can 
    # be accessed from those.
    for index, graph, info, dev in zip(indices, graphs, infos, devs):
        
        graph['graph_output'] = info['graph_output']
        # besides the raw output vector for the prediction, we also want to store the actual prediction 
        # outcome. This differs based on what kind of task we are dealing with here. 
        if dataset_type == 'regression':
            graph['graph_prediction'] = info['graph_output'][0]
        elif dataset_type == 'classification':
            graph['graph_prediction'] = np.argmax(info['graph_output'])
        
        # correspondingly, the calculation of the fidelity is also different for regression and classification
        graph['graph_deviation'] = dev
        if dataset_type == 'regression':
            graph['graph_fidelity'] = np.array([-dev[0, 0], +dev[0, 1]])
        elif dataset_type == 'classification':
            matrix = np.array(dev)
            graph['graph_fidelity'] = np.diag(matrix)
        
        # Also we want to store all the information about the explanations channels, which includes the 
        # explanations masks themselves, but also the embedding vectors  
        graph['graph_embeddings'] = info['graph_embedding']
        graph['node_importances'] = array_normalize(info['node_importance'])
        graph['edge_importances'] = array_normalize(info['edge_importance'])

    # ~ concept clustering
    
    # As the concepts are generated we are going to store them in this list. Each concept is essentially 
    # representated as a dictionary which has certain keys that describe some aspect of it.
    concepts: t.List[dict] = []
    
    logger.info('starting the concept clustering...')
    for channel_index in range(num_channels):
        
        logger.info(f'for channel {channel_index}')
        
        # The first thing we do is to filter the dataset so that we only have those elements that meet 
        # the given fidelity threshold. Only if samples show a certain minimal fidelity we can be sure that 
        # those explanations are actually meaningful for the predictions.
        
        indices_channel = [index for index, graph in zip(indices, graphs) if graph['graph_fidelity'][channel_index] > fidelity_threshold]
        indices_channel = np.array(indices_channel)
        graphs_channel = [index_data_map[index]['metadata']['graph'] for index in indices_channel]
        
        # channel_embeddings: (num_graphs, embedding_dim)
        graph_embeddings_channel = np.array([graph['graph_embeddings'][channel_index] for graph in graphs_channel])
        clusterer = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            metric=cluster_metric,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(graph_embeddings_channel)
        
        clusters = [label for label in set(labels) if label > 0]
        num_clusters = len(clusters)
        logger.info(f'found {num_clusters} from {len(graph_embeddings_channel)} embeddings')
        
        for cluster_index in clusters:
            
            mask_cluster = (labels == cluster_index)
            graph_embeddings_cluster = graph_embeddings_channel[mask_cluster]
            
            cluster_centroid = np.mean(graph_embeddings_cluster, axis=0)
            indices_cluster = indices_channel[mask_cluster]
            elements_cluster = [index_data_map[index] for index in indices_cluster]
            graphs_cluster = [data['metadata']['graph'] for data in elements_cluster]
            index_tuples_cluster = [(index, channel_index) for index in indices_cluster]
            
            if dataset_type == 'regression':
                contribution_cluster = np.mean([graph['graph_deviation'][0, channel_index] for graph in graphs_cluster])
            elif dataset_type == 'classification':
                contribution_cluster = np.mean([graph['graph_deviation'][channel_index, channel_index] for graph in graphs_cluster])
                
            concept: dict = {
                'index': cluster_index,
                'channel_index': channel_index,
                'index_tuples': index_tuples_cluster,
                'embeddings': graph_embeddings_cluster,
                'centroid': cluster_centroid,
                'elements': elements_cluster,
                'graphs': graphs_cluster,
                'name': channel_infos[channel_index]['name'],
                'color': channel_infos[channel_index]['color'],
            }
            concepts.append(concept)
            
    if sort_similarity:
        
        logger.info('sorting the concepts by similarity...')
        concepts_sorted = []
        for channel_index in range(num_channels):
            
            concepts_channel = [concept for concept in concepts if concept['channel_index'] == channel_index]
            
            # We will just randomly start with the first cluster and then iteratively traverse the list of 
            # the clusters by always selecting the next cluster according to which one is the closest to the 
            # the current one - out of the remaining clusters.
            concept = concepts_channel.pop(0)
            
            while len(concepts_channel) != 0:
                
                centroid = concept['centroid']
                centroid_distances = pairwise_distances(
                    np.expand_dims(centroid, axis=0),
                    [c['centroid'] for c in concepts_channel],
                    metric=cluster_metric,
                )
                
                index = np.argmin(centroid_distances[0])
                concept = concepts_channel.pop(index)
                concepts_sorted.append(concept)
                
        concepts = concepts_sorted
            
    return concepts
            