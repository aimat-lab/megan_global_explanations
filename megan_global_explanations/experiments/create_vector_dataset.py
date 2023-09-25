import os
import json

import numpy as np

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.data import NumericJsonEncoder


# == CLUSTER CREATION PARAMETERS ==
NUM_DIMENSIONS: int = 5
CLUSTER_DATA: dict = {
    0: {
        'centroid': [0, 1, -1, 0, 0],
        'variance': 1,
        'labels': [1, 0, 0]
    },
    1: {
        'centroid': [-2, 5, -3, 0, 1],
        'variance': 0.2,
        'labels': [0, 1, 0]
    }, 
    2: {
        'centroid': [5, 10, 0, 1, 15],
        'variance': 3,
        'labels': [0, 0, 1]
    }
}

NUM_ELEMENTS: int = 500

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    
    e.log('starting experiment...')

    index = 0
    dataset = {}
    for cluster_index, cluster_data in e.CLUSTER_DATA.items(): 
        
        centroid = np.array(cluster_data['centroid'])
        variance = cluster_data['variance']
        
        cluster_elements = centroid + variance * np.random.randn(e.NUM_ELEMENTS, e.NUM_DIMENSIONS)
        for vec in cluster_elements:
            
            data = {
                'index': index,
                'cluster': cluster_index,
                'labels': cluster_data['labels'],
                'vector': vec
            }
            dataset[index] = data
            index += 1
            
    # ~ Saving the dataset
    # Now that the dataset has been generated we can save it as a JSON file

    e.log('writing dataset...')
    dataset_path = os.path.join(e.path, 'dataset.json')
    with open(dataset_path, mode='w') as file:
        content = json.dumps(dataset, cls=NumericJsonEncoder)
        file.write(content)


experiment.run_if_main()