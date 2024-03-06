import os
import random
import pathlib
import traceback
import typing as t
from copy import deepcopy

import numpy as np
import visual_graph_datasets.typing as tv
from rdkit import Chem
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.torch.megan import Megan
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.graph import copy_graph_dict
from visual_graph_datasets.graph import extract_subgraph
from visual_graph_datasets.graph import graph_expand_mask
from visual_graph_datasets.graph import graph_find_connected_regions

from megan_global_explanations.prototype.optimize import genetic_optimize
from megan_global_explanations.prototype.optimize import embedding_distance_fitness
from megan_global_explanations.prototype.optimize import embedding_distances_fitness_mse
from megan_global_explanations.prototype.optimize import graph_matching_fitness
from megan_global_explanations.prototype.optimize import graph_matching_embedding_fitness
from megan_global_explanations.prototype.colors import mutate_remove_node
from megan_global_explanations.prototype.colors import mutate_remove_edge
from megan_global_explanations.gpt import query_gpt
from megan_global_explanations.gpt import describe_molecule


PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# The parameters determine the details related to the dataset that should be used as the basis 
# of the concept extraction

# :param VISUAL_GRAPH_DATASETS:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either 
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this 
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote 
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'ba2motifs'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or 
#       "classification"
DATASET_TYPE: str = 'classification'
# :param CHANNEL_INFOS:
#       This dictionary can optionally be given to supply additional information about the individual 
#       explanation channels. The key should be the index of the channel and the value should again be 
#       a dictionary that contains the information for the corresponding channel.
CHANNEL_INFOS: t.Dict[int, dict] = {
    0: {
        'name': 'cycle',
        'color': 'navajowhite',
    },
    1: {
        'name': 'house',
        'color': 'paleturquoise',
    }
}

# == MODEL PARAMETERS == 
# These parameters determine the details related to the model that should be used for the 
# concept extraction. For this experiment, the model should already be trained and only 
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the 
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'ba2motifs.ckpt')

# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a 
#       fidelity higher than this will be used as possible candidates for the clustering.
FIDELITY_THRESHOLD: float = 5.0
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially 
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
MIN_CLUSTER_SIZE: int = 10
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the 
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while 
#       lower values tend to result in more clusters.
MIN_SAMPLES: int = 30

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
INITIAL_POPULATION_SAMPLE: int = 100
# :param OPTIMIZE_PROTOTYPE_POPSIZE:
#       This integer number determines the population size of the genetic optimization algorithm
#       that is used to optimize the prototype representation.
OPTIMIZE_PROTOTYPE_POPSIZE: int = 1000
# :param OPTIMIZE_PROTOTYPE_EPOCHS:
#       This integer number determines the number of epochs that the genetic optimization algorithm
#       will be executed for the prototype optimization.
OPTIMIZE_PROTOTYPE_EPOCHS: int = 20
# :param OPENAI_KEY:
#       This string value has to be the OpenAI API key that should be used for the GPT-4 requests
#       that will be needed to generate the natural language descriptions of the prototypes.
OPENAI_KEY: str = os.getenv('OPENAI_KEY')
# :param DESCRIBE_PROTOTYPE:
#       This boolean flag determines whether the prototype description should be generated at all
#       or not. If this is False, the entire description routine will be skipped during the
#       cluster discovery.
DESCRIBE_PROTOTYPE: bool = False
# :param HYPOTHESIZE_PROTOTYPE:
#       This boolean flag determines whether the prototype hypothesis should be generated at all
#       or not. If this is False, the entire hypothesis routine will be skipped during the
#       cluster discovery.
HYPOTHESIZE_PROTOTYPE: bool = False


__DEBUG__ = True

experiment = Experiment.extend(
    'vgd_concept_extraction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('optimize_prototype', default=False, replace=True)
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
        
        # elements_initial.append({
        #     'graph': graph,
        #     'value': processing.unprocess(graph)
        # })
        # continue
        
        # In the very first step we need to binarize the node explanation mask by using a simple 
        # threshold.
        node_mask = (graph['node_importances'][:, channel_index] > 0.5).astype(int)
        
        # Since the explanation mask tends to be too sparse in many situations we perform a one-hop expansion 
        # of this mask here. So that function will propagate the mask label to all the nodes that are currently 
        # adjacent to at least one other mask node.
        node_mask = graph_expand_mask(graph, node_mask, num_hops=4)
        
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
        fitness_func=lambda elements: embedding_distances_fitness_mse(
            elements=elements,
            model=model,
            channel_index=channel_index,
            anchors=[anchor],
            violation_radius=0.2,
        ),
        sample_func=lambda: random.choice(elements_initial),
        mutation_funcs=[
            lambda *args, **kwargs: mutate_remove_node(*args, **kwargs, processing=processing),
            lambda *args, **kwargs: mutate_remove_edge(*args, **kwargs, processing=processing),
        ],
        num_epochs=e.OPTIMIZE_PROTOTYPE_EPOCHS,
        population_size=e.OPTIMIZE_PROTOTYPE_POPSIZE,
        elite_ratio=0.1,
        refresh_ratio=0.5,
        logger=e.logger,
    )
    
    return element


@experiment.hook('describe_prototype', default=False, replace=True)
def describe_prototype(e: Experiment,
                       value: str,
                       image_path: str,
                       ) -> str:
    
    return ''


@experiment.hook('prototype_hypothesis', default=False, replace=True)
def prototype_hypothesis(e: Experiment,
                         value: str,
                         image_path: str,
                         description: str,
                         channel_index: int,
                         ) -> t.Optional[str]:
    
    return None


experiment.run_if_main()