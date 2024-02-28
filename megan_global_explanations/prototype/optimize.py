import os
import random
import logging
import typing as t
from copy import deepcopy

import numpy as np
import visual_graph_datasets.typing as tv
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from graph_attention_student.utils import array_normalize
from graph_attention_student.torch.megan import Megan

from megan_global_explanations.utils import NULL_LOGGER

# ~ Fitness Functions


def graph_matching_embedding_fitness(graphs: t.List[tv.GraphDict],
                                     model: Megan,
                                     channel_index: int,
                                     anchor_graphs: t.List[tv.GraphDict],
                                     processing: MoleculeProcessing,
                                     check_edges: bool = False,
                                     ratio: float = 0.5,
                                     ) -> np.ndarray:
    num_anchors = len(anchor_graphs)
    cutoff = int(num_anchors * ratio)
    
    infos_anchors = model.forward_graphs(anchor_graphs)
    centroid = np.mean([info['graph_embedding'][:, channel_index] for info in infos_anchors], axis=0)
    
    infos = model.forward_graphs(graphs)
    
    fitness = np.zeros(shape=(len(graphs), ))
    for index, (info, graph) in enumerate(zip(infos, graphs)):
        semantic_violation = int(cosine(info['graph_embedding'][:, channel_index], centroid) > 0.5)
        match_violations = [int(not processing.contains(anchor, graph, check_edges=check_edges)) for anchor in anchor_graphs]
        fitness[index] = (
            1000 * semantic_violation
            + 100 * np.sum(match_violations)
            - len(graph['node_indices'])
        )

    return fitness


def graph_matching_fitness(graphs: t.List[tv.GraphDict],
                           anchor_graphs: t.List[tv.GraphDict],
                           processing: MoleculeProcessing,
                           check_edges: bool = False,
                           ratio: float = 0.5,
                           ) -> np.ndarray:
    num_anchors = len(anchor_graphs)
    cutoff = int(num_anchors * ratio)
    
    fitness = np.zeros(shape=(len(graphs), ))
    for index, graph in enumerate(graphs):
        violations = [100 * int(not processing.contains(anchor, graph, check_edges=check_edges)) for anchor in anchor_graphs]
        fitness[index] = np.sort(violations)[::-1][cutoff] - len(graph['node_indices'])

    return fitness


def embedding_distances_fitness_mse(elements: t.List[dict],
                                    model: Megan,
                                    channel_index: int,
                                    anchors: t.List[np.ndarray],
                                    distance_func: t.Callable = cosine,
                                    node_factor: float = 0.1,
                                    edge_factor: float = 0.02,
                                    violation_radius: float = 0.05,
                                    ) -> np.ndarray:
    
    # infos = model.forward_graphs(graphs)
    # fitness = np.zeros(shape=(len(graphs), ))
    graphs = [element['graph'] for element in elements]
    
    infos = model.forward_graphs(graphs)
    fitness = np.zeros(shape=(len(graphs), ))
    
    for index, (info, graph, element) in enumerate(zip(infos, graphs, elements)):
        embedding = info['graph_embedding'][:, channel_index]
        
        node_importance = array_normalize(info['node_importance'])
        
        distances = np.array([distance_func(embedding, anchor) for anchor in anchors])
        violations = [int(dist > violation_radius) for dist in distances]
        num_violations = np.sum(violations)
        fitness[index] = (
            # The first term for the objective is the actual distance of the graph embedding to 
            # the anchor location.
            #100 * max(0, num_violations - int(0.5 * len(anchors)))
            100 * num_violations
            # This second term here penalizes graphs that are too big because usually we want to 
            # find the minimally matching graph.
            + len(graph['node_indices'])
            #- np.sum(node_importance[:, channel_index])
            + 100 * int('damaged' in element and element['damaged'])
            #+ 0.1 * np.mean(info['node_importance'][:, other_channels])
            #- 0.1 * np.mean(info['node_importance'][:, channel_index])
            # + edge_factor * 0.5 * len(graph['edge_indices'])
        )
        
    return fitness


def embedding_distance_fitness(graphs: t.List[tv.GraphDict],
                               model: Megan,
                               channel_index: int,
                               anchor: np.ndarray,
                               distance_func: t.Callable = cosine,
                               node_factor: float = 0.1,
                               edge_factor: float = 0.02,
                               ) -> np.ndarray:
    infos = model.forward_graphs(graphs)
    fitness = np.zeros(shape=(len(graphs), ))
    for index, (info, graph) in enumerate(zip(infos, graphs)):
        embedding = info['graph_embedding'][:, channel_index]
        fitness[index] = (
            # The first term for the objective is the actual distance of the graph embedding to 
            # the anchor location.
            distance_func(embedding, anchor) 
            # This second term here penalizes graphs that are too big because usually we want to 
            # find the minimally matching graph.
            + node_factor * len(graph['node_indices'])
            + edge_factor * 0.5 * len(graph['edge_indices'])
        )
    
    return fitness

# ~ Selection Functions

def tournament_select(elements: t.List[dict],
                      tournament_size: int = 5,
                      ) -> dict:
    contestants = random.sample(elements, k=tournament_size)
    contestants.sort(key=lambda element: element['fitness'])

    return contestants[0]


# ~ Actual Genetic Algorithm

def genetic_optimize(fitness_func: t.Callable,
                     sample_func: t.Callable,
                     mutation_funcs: t.List[t.Callable],
                     select_func: t.Callable = tournament_select,
                     num_epochs: int = 100,
                     population_size: int = 1000,
                     refresh_ratio: float = 0.1,
                     elite_ratio: float = 0.1,
                     logger: logging.Logger = NULL_LOGGER, 
                     ) -> tv.GraphDict:
    """
    

    :param fitness_func: This is supposed to be a function which accepts a list of B graph dict 
        representations and outputs a numpy array of the shape (B, ) which contains a single float 
        fitness value for each of the input graphs.
    :param sample_func: This is a function which is supposed to implement a random element sampling. 
        the function should not accept any parameters 
    """
    
    num_refresh = int(population_size * refresh_ratio)
    num_elite = int(population_size * elite_ratio)
    num_rest = population_size - num_elite - num_refresh
    
    def update_fitness(elements: t.List[dict]) -> None:
        graphs = [element['graph'] for element in elements]
        # fitness: (B, )
        fitness = fitness_func(elements)
        
        for element, fit in zip(elements, fitness):
            element['fitness'] = fit
            
        return elements
    
    def sample_element() -> dict:
        # "sample_func" is supposed to sample an element randomly from the initial graphs 
        # distribution. It returns a dictionary which should contain the following mandatory entries:
        # - graph: The full graph dict
        # - value: The string representation of the graph
        element = sample_func()
        element['fitness'] = None
        return element
    
    # ~ Creating the initial population
    population = [sample_element() for _ in range(population_size)]
    update_fitness(population)
    population.sort(key=lambda element: element['fitness'])
    
    # ~ optimizing with the genetic algorithm
    for epoch in range(num_epochs):
        
        # First we need to create the candidates from the population process
        #candidates = [select_func(population) for element in population]
        candidates = [deepcopy(element) for element in population]
        candidates_mutated = []
        for element in candidates:
            mutation_func = random.choice(mutation_funcs)
            mutated = mutation_func(element)
            candidates_mutated.append(mutated)
            
        candidates = candidates_mutated
        update_fitness(candidates)
        candidates.sort(key=lambda element: element['fitness'])
        
        refreshments = [sample_element() for i in range(num_refresh)]
        update_fitness(refreshments)
        
        # Now we have to create the new population 
        population = (
            candidates[:num_rest] + 
            population[:num_elite] + 
            #[population[0]] * num_elite +
            refreshments
        )
        population.sort(key=lambda element: element['fitness'])
        
        fitness = [element['fitness'] for element in population]
        best_fitness = np.min(fitness)
        mean_fitness = np.mean(fitness) 
        logger.info(f' * epoch {epoch:03d}/{num_epochs}'
                    f' - pop size: {len(population)}'
                    f' - best: {best_fitness:.4f}'
                    f' - mean: {mean_fitness:.4f}')
        
    population.sort(key=lambda element: element['fitness'])
    best = population[0]
    history = {}
    
    return best, history