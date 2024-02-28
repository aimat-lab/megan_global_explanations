import os
import typing as t

import numpy as np
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.colors import ColorProcessing

import megan_global_explanations.typing as tg


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