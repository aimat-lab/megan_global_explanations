import os
import random
import typing as t

import networkx as nx
from visual_graph_datasets.graph import copy_graph_dict
from visual_graph_datasets.graph import nx_from_graph
from visual_graph_datasets.graph import graph_add_edge
from visual_graph_datasets.graph import graph_remove_edge
from visual_graph_datasets.graph import graph_node_adjacency
from visual_graph_datasets.graph import graph_attach_node
from visual_graph_datasets.graph import graph_remove_node
from visual_graph_datasets.graph import graph_is_connected
from visual_graph_datasets.graph import graph_has_isolated_node
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.generation.colors import *
from vgd_counterfactuals.generate.colors import get_valid_node_replace
from vgd_counterfactuals.generate.colors import get_valid_add_edge
from vgd_counterfactuals.generate.colors import get_valid_remove_edge


COLOR_PROCESSING = ColorProcessing()

COLORS = [
    RED,
    GREEN,
    BLUE,
    MAGENTA,
    CYAN,
    YELLOW,
    GRAY,
]


def mutate_remove_edge(element: dict,
                       processing: ProcessingBase = COLOR_PROCESSING,
                       max_tries: int = 5,
                       ) -> dict:
    graph = copy_graph_dict(element['graph'])

    node_adjacency = graph_node_adjacency(graph)
    graph_degree = np.sum(node_adjacency, axis=-1)
    node_indices = [index for index in graph['node_indices'] if graph_degree[index] > 2]
    if len(node_indices) < 2:
        return element
    
    for i in range(max_tries):
        node_index_1 = random.choice(node_indices)
        node_index_2 = random.choice(node_indices)
       
        graph_ = graph_remove_edge(
           graph=copy_graph_dict(graph),
           node_index_1=node_index_1,
           node_index_2=node_index_2,
           directed=False,
        )
       
        # if graph_is_connected(graph_) and len(graph_['node_indices']) > 1:
        if not graph_has_isolated_node(graph_):
            return {
                'graph': graph_,
                'value': processing.unprocess(graph_)
            }
       
    # If we cant manage to make a valid edit within the max number of tries we will just 
    # return the original unmodified element. 
    return element


def mutate_add_edge(element: dict,
                    processing: ProcessingBase = COLOR_PROCESSING,
                    max_tries: int = 5
                    ) -> dict:
    graph = copy_graph_dict(element['graph'])
    # Here we simply pick two nodes that do not already have an edge between them and then 
    # insert that edge.
    
    node_adjacency = graph_node_adjacency(graph)
    for i in range(max_tries):
        node_index_1 = random.choice(graph['node_indices'])
        node_index_2 = random.choice(graph['node_indices'])
        
        if node_index_1 == node_index_2 or node_adjacency[node_index_1][node_index_2]:
            continue
        
        graph = graph_add_edge(
            graph=graph,
            node_index_1=node_index_1,
            node_index_2=node_index_2,
            directed=False,
            attributes={'edge_attributes': np.ndarray([1])},
        )
        
        return {
            'graph': graph,
            'value': processing.unprocess(graph)
        }
        
    # If we cant manage to make a valid edit within the max number of tries we will just 
    # return the original unmodified element.
    return element


def mutate_modify_node(element: dict,
                       colors: t.List[t.Any] = COLORS,
                       processing: ProcessingBase = COLOR_PROCESSING
                       ) -> dict:

    graph = copy_graph_dict(element['graph'])
    # Here we simply select a random node and change it's color to another 
    # random color.
    node_index = random.choice(graph['node_indices'])
    node_attributes = np.array(random.choice(colors), dtype=float)
    graph['node_attributes'][node_index] = node_attributes
    
    return {
        'graph': graph,
        'value': processing.unprocess(graph),   
    }
    

def mutate_add_node(element: dict,
                    colors: t.List[t.Any] = COLORS,
                    processing: ProcessingBase = COLOR_PROCESSING,
                    ) -> dict:
    graph = copy_graph_dict(element['graph'])
    # We randomly choose a node from the graph to which we are going to attach 
    # the new node to and the new node will be randomly picked from the 
    # list of possible colors
    node_index = random.choice(graph['node_indices'])
    node_attributes = np.array(random.choice(colors), dtype=float)

    graph_ = graph_attach_node(
        graph,
        anchor_index=node_index,
        node_attribute=node_attributes,
        edge_attribute=[1.0],
    )
    
    return {
        'graph': graph_,
        'value': processing.unprocess(graph_),
    }


def mutate_remove_node(element: dict,
                       processing: ProcessingBase = COLOR_PROCESSING,
                       max_tries: int = 5,
                       ) -> dict:
    
    if len(element['graph']['node_indices']) <= 2:
        return element
    
    graph = copy_graph_dict(element['graph'])
    
    for _ in range(max_tries):
    
        try:
            node_index = random.choice(graph['node_indices'])
            graph_ = graph_remove_node(
                graph=graph,
                node_index=node_index,
            )
        except:
            continue
    
        
        #if graph_is_connected(graph_) and len(graph_['node_indices'] > 1):
        if not graph_has_isolated_node(graph_):
            return {
                'graph': graph_,
                'value': processing.unprocess(graph_)
            }

    return element


def sample_from_cogiles(cogiles_list: t.List[str],
                        processing: ProcessingBase = COLOR_PROCESSING,
                        ) -> dict:
    cogiles = random.choice(cogiles_list)
    graph = processing.process(cogiles)
    
    return {
        'value': cogiles,
        'graph': graph,
    }