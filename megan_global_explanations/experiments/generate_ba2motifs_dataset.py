"""
This experiment creates the "visual graph dataset" version of the BA2Motifs dataset from the Pytorch Geometric library.
"""
import os
import pathlib
import tempfile
import typing as t

import numpy as np
from torch_geometric.datasets import BA2MotifDataset
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.generic import GenericProcessing

PATH = pathlib.Path(__file__).parent.absolute()

# We only create a sub class of the GenericProcessing class here for the sake of creating the 
# indepdent processing module, which becomes easier like this.
class MotifProcessing(GenericProcessing):
    pass


# == EXPERIMENT PARAMETERS ==
# The parameters for the experiment.

# :param DATASET_NAME:
#       The name of the dataset to be generated. This will be the folder name of the resulting  
#       visual graph dataset folder.
DATASET_NAME: str = 'ba2motifs'

__DEBUG__ = True


experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment to generate "BA2Motifs" dataset')
    
    with tempfile.TemporaryDirectory() as tempdir:
        
        e.log('loading the dataset from torch geometric...')    
        dataset = BA2MotifDataset(
            root=tempdir,
        )
        e.log(f'loaded dataset: {dataset}')
        
        e.log('creating visual graph dataset folder...')
        dataset_folder = os.path.join(e.path, e.DATASET_NAME)
        os.mkdir(dataset_folder)
        
        e.log('setting up the VisualGraphDatasetWriter...')
        writer = VisualGraphDatasetWriter(
            path=dataset_folder,
        )
        
        e.log('setting up GenericProcessing...')
        processing = MotifProcessing()
        
        e.log('saving the processing instance as indepdent module...')
        content = create_processing_module(processing)
        module_path = os.path.join(dataset_folder, 'process.py')
        with open(module_path, mode='w') as file:
            file.write(content)
        
        e.log('processing the dataset into the visual graph dataset...')
        graphs: list[dict] = []
        for index, data in enumerate(dataset):
            
            num_nodes = len(data.x)
            node_indices = np.arange(num_nodes)
            node_attributes = data.x.numpy()
            
            edge_indices = data.edge_index.T.numpy()
            num_edges = len(edge_indices)
            edge_attributes = np.zeros((num_edges, 1))
            
            label = data.y.numpy()[0]
            graph_labels = np.array([int(label == 0), int(label == 1)])
            
            graph = {
                'node_indices': node_indices,
                'node_attributes': node_attributes,
                'edge_indices': edge_indices,
                'edge_attributes': edge_attributes,
                'graph_labels': graph_labels,
            }
            graphs.append(graph)
            
            processing.create(
                index=index,
                value=None,
                graph=graph,
                writer=writer,
            )
            
            if index % 100 == 0:
                e.log(f' * {index} done')
            
        e.log(f'processed {len(graphs)} graphs')
        
experiment.run_if_main()