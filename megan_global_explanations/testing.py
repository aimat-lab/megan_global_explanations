import os
import json
import typing as t

import numpy as np
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.colors import ColorProcessing

import megan_global_explanations.typing as tg


class MockModel():
    """
    This is a mock implementation for testing purposes. This class is supposed to emulate the external interface of a 
    ``Megan`` model. Most importantly the ``forward_graphs`` method is supposed to be implemented such that it returns
    a list of dictionaries where each dictionary contains keys that describe the output of the model regarding the 
    output of the model, the embeddings and the explanations. 
    
    Additionally to be used for testing the class also implements the ``save`` and ``load_from_checkpoint`` methods.
    These simply save the parameters of the constructor as a json format and then load it accordingly again.
    """
    def __init__(self,
                 num_channels: int = 2,
                 embedding_dim: int = 64,
                 ) -> None:
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        
        self.params = {
            'num_channels': self.num_channels,
            'embedding_dim': self.embedding_dim,
        }
        
    def forward_graphs(self, graphs: t.List[dict]):
        infos = []
        for graph in graphs:
            info = {
                'graph_output':     np.random.random(),
                'graph_embedding':  np.random.random((self.embedding_dim, self.num_channels)),
                'node_importance':  np.random.random((len(graph['node_indices']), self.num_channels)),
                'edge_importance':  np.random.random((len(graph['edge_indices']), self.num_channels)),
            }
            infos.append(info)
            
        return infos
    
    def save(self, path: str):
        with open(path, mode='w') as file:
            json.dump(self.params, file)
            
    @classmethod
    def load_from_checkpoint(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError('The model checkpoint file does not exist!')
        
        with open(path, mode='r') as file:
            params = json.load(file)
            return cls(**params)


def create_mock_concepts(num: int,
                         dim: int = 64,
                         prototype_value: t.Optional[str] = None,
                         num_graphs: int = 0,
                         processing: ProcessingBase = ColorProcessing(),
                         ) -> t.List[tg.ConceptDict]:
    
    concept_data = []
    for index in range(num):
        concept = {
            'index': index,
            'centroid': np.random.rand(dim),
            'num': np.random.randint(10, 100),
        }
        
        if prototype_value:
            
            graph = processing.process(prototype_value)
            concept['prototypes'] = [{
                'image_path': None,
                'graph': graph,
            }]
            
        # Another thing that a concept may be associated with is the exact graphs that make up the 
        # concept. If a number greater 0 is requested, these graphs are simply approximated as copies 
        # of the prototype graphs which was already created anyways.
        if prototype_value and num_graphs > 0:
            graphs = [graph for _ in range(num_graphs)]
            concept['graphs'] = graphs
        
        concept_data.append(concept)
    
    return concept_data