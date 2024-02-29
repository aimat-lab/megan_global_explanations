import os
import sys
import json
import pathlib
import random
import logging
import typing as t
from copy import deepcopy

import numpy as np
from visual_graph_datasets.data import VisualGraphDatasetReader
from decouple import config

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')

LOG_TESTING = config('LOG_TESTING', cast=bool, default=True)
LOG = logging.getLogger('Testing')
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.NullHandler())
if LOG_TESTING:
    LOG.addHandler(logging.StreamHandler(sys.stdout))
    
# For some functions we require an OpenAI key to be able to query the OpenAI API, so we load it from the
# environment variables here. Specifically from the .env file that is located in the same folder as this
# module.
OPENAI_KEY = config('OPENAI_KEY', default=None)    

    
def load_mock_clusters(num_clusters: int = 3,
                       num_channels: int = 2,
                       embedding_dim: int = 10,
                       num_prototypes: int = 0,
                       ) -> t.List[dict]:
    """
    This function loads a cluster data list from the file system. This cluster data list is 
    a mock data structure mainly intended for testing the creation of the concept cluster reports.
    
    :returns: A list of dicts which each contain the information about one cluster such that 
        it can be used to create the cluster report
    """
    # The mock clusters will be based on the mock dataset which is also part of the testing assets
    index_data_map: dict = deepcopy(load_mock_vgd())
    indices = list(index_data_map.keys())
    num_elements = len(index_data_map)
    
    folder_path = os.path.join(ASSETS_PATH, 'mock_clusters')
    
    data_path = os.path.join(folder_path, 'data.json')
    with open(data_path) as file:
        content = file.read()
        cluster_data_list = json.loads(content)
    
    cluster_data_list = []
    for i in range(num_clusters):
        channel_index = random.randint(0, num_channels - 1)
        cluster_indices = random.sample(indices, k=int(num_elements / num_clusters))
        cluster_datas = deepcopy([index_data_map[index] for index in cluster_indices])
                
        for data in cluster_datas:
            graph = data['metadata']['graph']
            
            graph['node_importances'] = np.random.random(size=(len(graph['node_indices']), num_channels))
            graph['edge_importances'] = np.random.random(size=(len(graph['edge_indices']), num_channels))
            graph['graph_deviation'] = np.random.random(size=(1, num_channels))
            graph['graph_prediction'] = np.random.random()
            
            graph['graph_repr'] = data['metadata']['repr'] if 'repr' in data['metadata'] else data['metadata']['value']
            
        cluster_embeddings = [np.random.random(size=(embedding_dim, )) for _ in cluster_indices]
        cluster_centroid = np.mean(cluster_embeddings, axis=0)
            
        cluster_data = {
            'index':            i,
            'channel_index':    channel_index,
            'elements':         cluster_datas,
            'image_paths':      [data['image_path'] for data in cluster_datas],
            'graphs':           [data['metadata']['graph'] for data in cluster_datas],
            'index_tuples':     [(index, channel_index) for index in cluster_indices],
            'embeddings':       cluster_embeddings,
            'centroid':         cluster_centroid,
            'name':             'name',
        }
        
        if num_prototypes > 0:
            cluster_data['prototypes'] = deepcopy(random.choices(cluster_datas, k=num_prototypes))
        
        cluster_data_list.append(cluster_data)
        
    return cluster_data_list


def load_mock_vgd() -> dict:
    """
    This function loads a visual graph dataset from the file system. This dataset is a mock
    dataset mainly intended for testing the concept extraction and the concept cluster report
    creation.
    
    :returns: A dictionary which contains the visual graph dataset data
    """
    path = os.path.join(ASSETS_PATH, 'mock_vgd')
    reader = VisualGraphDatasetReader(path=path)
    index_data_map = reader.read()
    return index_data_map


# Here I want to load the .env file in the same folder as this module and expose all the env variables
# as a dict ENV_VARS

