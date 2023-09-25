import os
import json
import random
import shutil

import numpy as np
from visual_graph_datasets.data import VisualGraphDatasetReader, NumericJsonEncoder

from util import ASSETS_PATH

# == PARAMETERS ==

VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
NUM_CHANNELS: int = 2
NUM_CLUSTERS: int = 3
NUM_ELEMENTS: int = 30

# The number of dimensions which the randomly generated embedding vectors should have
NUM_DIMENSIONS: int = 32

# == SCRIPT ==

channel_indices = list(range(NUM_CHANNELS))

path = os.path.join(ASSETS_PATH, 'mock_clusters')
if os.path.exists(path):
    shutil.rmtree(path)
    
os.mkdir(path)

print('reading dataset...')
reader = VisualGraphDatasetReader(
    path=VISUAL_GRAPH_DATASET_PATH,
)
index_data_map = reader.read()
indices = list(index_data_map.keys())

print('generating random cluster assignments...')
cluster_data_list = []
for cluster_index in range(NUM_CLUSTERS):
    
    # first we randomly sample some graphs from the dataset and determine a channel index
    cluster_indices = random.sample(indices, k=NUM_ELEMENTS)
    channel_index = random.choice(channel_indices)
    
    # To make it a bit more realsitic we will actually generate all the random embedding vectors 
    # around an also randomly generated centroid vector for each of the clusters.
    centroid = np.random.rand(NUM_DIMENSIONS)
    embeddings = np.random.normal(centroid, 0.1, size=(NUM_ELEMENTS, NUM_DIMENSIONS))
    embeddings = embeddings.tolist()
    
    graphs = []
    index_tuples = []
    image_paths = []
    for index, emb in zip(cluster_indices, embeddings):
        index_tuples.append([index, channel_index])
        
        # first we need to copy the image file
        image_path = os.path.join(path, f'{cluster_index}_{index}.png')
        shutil.copy(index_data_map[index]['image_path'], image_path)
        image_paths.append(image_path)
        
        graph = index_data_map[index]['metadata']['graph']
        graph['node_importances'] = graph['node_importances_2']
        graph['edge_importances'] = graph['edge_importances_2']
        graph['graph_deviation'] = [[random.uniform(-2, 2)] for i in range(NUM_CHANNELS)]
        graph['graph_prediction'] = random.uniform(-10, 10)
        
        graphs.append(graph)
    
    cluster_data = {
        'index': cluster_index,
        'graphs': graphs,
        'index_tuples': index_tuples,
        'embeddings': embeddings,
    }
    cluster_data_list.append(cluster_data)
    
    
data_path = os.path.join(path, 'data.json')
with open(data_path, mode='w') as file:
    content = json.dumps(cluster_data_list, cls=NumericJsonEncoder)
    file.write(content)