import os
import json
import random
import typing as t
import numpy as np

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.processing.colors import graph_to_cogiles
from visual_graph_datasets.generation.colors import make_grid_motif, make_star_motif, make_ring_motif
from visual_graph_datasets.generation.colors import BLUE, RED, GREEN, YELLOW, CYAN, MAGENTA
from visual_graph_datasets.data import NumericJsonEncoder
from vgd_counterfactuals.base import CounterfactualGenerator
from vgd_counterfactuals.generate.colors import get_neighborhood


# == CLUSTER CREATION PARAMETERS ==
NUM_DIMENSIONS: int = 5
CLUSTER_DATA: dict = {
    0: {
        'seed': graph_to_cogiles(make_ring_motif(CYAN, BLUE, 9)),
        'variance': 2,
        'labels': [1, 0, 0, 0, 0]
    },
    1: {
        'seed': graph_to_cogiles(make_star_motif(MAGENTA, RED, 9)),
        'variance': 2,
        'labels': [0, 1, 0, 0, 0]
    },
    2: {
        'seed': graph_to_cogiles(make_ring_motif(RED, MAGENTA, 9)),
        'variance': 2,
        'labels': [0, 0, 1, 0, 0]
    }, 
    3: {
        'seed': graph_to_cogiles(make_grid_motif(YELLOW, GREEN, 3, 3)),
        'variance': 2,
        'labels': [0, 0, 0, 1, 0]
    },
    4: {
        'seed': graph_to_cogiles(make_grid_motif(RED, BLUE, 3, 3)),
        'variance': 2,
        'labels': [0, 0, 0, 0, 1]
    }
}

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    e.log('creating the processing instance...')
    processing = ColorProcessing()
    
    e.log('generating the dataset by applying random edits on the seeds...')
    dataset_path = os.path.join(e.path, 'dataset')
    os.mkdir(dataset_path)
    index: int = 0
    index_data_map = {}
    for cluster_index, cluster_data in e.CLUSTER_DATA.items():
        
        # First of all we would like to visualize the seed graphs so that the user can visually verify that 
        # they are the correct ones in the end for that purpose we create a seperate folder into which we 
        # save the corresponding visual graph element representation
        cluster_path = os.path.join(e.path, f'{cluster_index:02d}_graph_cluster')
        os.mkdir(cluster_path)
        
        processing.create(
            value=cluster_data['seed'],
            index=0,
            output_path=cluster_path,
        )
        
        # Then we need to actually create the dataset here by applying all possible single graph edits on 
        # the seed graphs to basically create graph clusters.
        neighbors: t.List[dict] = [{'value': cluster_data['seed']}]
        for k in range(cluster_data['variance']):
            num_neighbors = min(10, len(neighbors))
            for data in random.sample(neighbors, k=num_neighbors):
                neighbors += get_neighborhood(
                    value=data['value'],
                    processing=processing,
                    # colors=COLORS,
                )
            
        
        # The previous function only returns a list of dicts, where each dict contains the domain specific 
        # representation of a graph and now we need to convert those into actual graphs to attach them to the 
        # dataset.
        for data in neighbors:
            processing.create(
                value=data['value'],
                index=index,
                output_path=dataset_path,
                additional_metadata={
                    'seed_index': cluster_index,
                    'seed_name': cluster_index,
                    'targets': cluster_data['labels']
                }
            )
            index += 1
            
        e.log(f' * seed "{cluster_index}" done - num elements: {len(neighbors)}')


experiment.run_if_main()