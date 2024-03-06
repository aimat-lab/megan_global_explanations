"""
This module defines the data structures and classes that are used to represent the custom concept data.
Beyond the data structures themselves, this module also defines the methods that are used to load and 
save the data from and to the file system. The data is stored in a JSON file format and also partially 
based on the visual graph dataset format to represent the concept prototypes for example.
"""
import os
import json
import shutil
import logging
import collections
import typing as t
import visual_graph_datasets.typing as tv
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.base import create_processing_module
from graph_attention_student.torch.megan import Megan

import megan_global_explanations.typing as tg
from megan_global_explanations.utils import NULL_LOGGER
from megan_global_explanations.utils import safe_int

# ~ Implementations

def resolve_path(path: str, base_path: str):
    expanded_path = os.path.expanduser(path)
    if os.path.isabs(expanded_path):
        return expanded_path
    else:
        return os.path.join(base_path, expanded_path)


def update_dict(original: dict, update: dict) -> dict:
    """
    This function updates the original dictionary with the update dictionary. This is done recursively 
    such that the original dictionary is updated in place.
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            update_dict(original[key], value)
        else:
            original[key] = value
            
    return original


def strip_graph_data(data: dict,
                     data_keys: t.List[str] = ['image_path'],
                     graph_keys: t.List[str] = ['node_']):
    
    if 'image_path' in data:
        del data['image_path']
        
    metadata = data['metadata']

    if 'graph' in metadata:
        del metadata['graph']


class ConceptWriter():
    
    def __init__(self, 
                 path: str,
                 processing: ProcessingBase,
                 model: t.Optional[Megan] = None,
                 logger: logging.Logger = NULL_LOGGER,
                 writer_cls: type = VisualGraphDatasetWriter,
                 ):
        self.path = path
        self.processing = processing
        self.model = model
        self.logger = logger
        self.writer_cls = writer_cls
        
        # This attribute will later on hold the absolute path of where the model was actually saved 
        # to. This will be set in the self.write_model method.
        self.model_path: t.Optional[str] = None
        
    def write(self,
              concepts: tg.ConceptData,
              ) -> None:
        
        # This will persistently save the model to a file in the folder.
        self.write_model()

        # This method will write the concept metadata as a json file to the folder
        # It is important that this is called after the model writing, since we need to save the model path as part 
        # of the metadata!
        reduced_concepts = []
        for concept in concepts:
            reduced_concepts.append({
                'index': concept['index'],
                'centroid': concept['centroid'],
                'channel_index': concept['channel_index'],
            })
        
        self.write_metadata(data={
            'concepts': reduced_concepts
        })
        
        for index, concept in enumerate(deepcopy(concepts)):
            self.logger.info(f' * writing concept {index:03d}/{len(concepts)}')
            self.write_concept(index, concept)
            
        # self.logger.info(' * writing concept processing')
        # self.write_processing()
            
    def write_model(self) -> None:
        
        if self.model is not None:
            self.model_path = os.path.join(self.path, 'model.ckpt')
            self.model.save(self.model_path)
            
    def write_processing(self) -> None:
        content = create_processing_module(self.processing)
        processing_path = os.path.join(self.path, 'process.py')
        with open(processing_path, mode='w') as file:
            file.write(content)
            
    def write_metadata(self, data: dict) -> None:
        metadata_path = os.path.join(self.path, 'metadata.json')
        
        metadata = {
            # Here we use the basename of the model path instead of the absolute making it a relative path. The reader 
            # class will be able to correctly understand this which makes it invariant to moving the folder.
            'model_path': os.path.basename(self.model_path),
            'dataset_path': None,
            **data,
        }
        
        with open(metadata_path, mode='w') as file:
            json.dump(metadata, file, cls=NumericJsonEncoder)
                
    def write_graph(self,
                    graph: tv.GraphDict,
                    index: int,
                    path: str,
                    additional_metadata: t.Optional[dict] = {},
                    ) -> None:
        
        if 'repr' in additional_metadata:
            value = additional_metadata['repr']
        elif 'graph_repr' in graph:
            value = graph['graph_repr']
        else:
            raise ValueError('No domain graph representation found for the given graph!')
                
        writer: VisualGraphDatasetWriter = self.writer_cls(
            path=path,
        )
                
        # To write the graph we need to care about two things: The visualization image and the 
        self.processing.create(
            value=value,
            graph=graph,
            output_path=path,
            index=index,
            additional_metadata=additional_metadata,
            writer=writer,
        )
                
    def write_concept(self,
                      index: int,
                      concept: tg.ConceptDict,
                      ) -> None:
        
        concept_path = os.path.join(self.path, f'{index:03d}')
        os.mkdir(concept_path)
        
        # ~ prototypes
        # Each cluster is *optionally* associated with one or more prototype elements. These are also graphs which 
        # are meant to represent the underlying pattern of the graph.
        # 
        # The prototypes we need to actually save in the format of visual graph elements because the prototypes cannot 
        # be implicitely derived from the dataset. They are the result of an optimization process on the concept cluster 
        # or some other difficult to repeat process and thus we want to save all the data regarding those fully.
        if 'prototypes' in concept:
            
            # We need to remove the prototype from the concept dict and handle it separately as this cannot 
            # just be written to the disk as a json file but needs to be handled as a visual graph dataset
            # element.
            prototypes = concept['prototypes']
            del concept['prototypes']
            
            prototypes_path = os.path.join(concept_path, 'prototypes')
            os.mkdir(prototypes_path)
                
            for index, prototype in enumerate(prototypes):
                graph = prototype['metadata']['graph']
                del prototype['metadata']['graph']
                
                self.write_graph(
                    graph=graph,
                    index=index,
                    path=prototypes_path,
                    additional_metadata=prototype,
                )
                
        # ~ graph elements
        # Each concept mainly consists of a number of graphs which make up that concept. As a unity those 
        # graphs are representative of the underlying pattern which that concept represents.
        # 
        # We dont want to save all of those graphs directly though as that may cause a memory problem if there 
        # are too many graphs or too many concepts. Instead we implicitly save the graphs by *reference*. Since a 
        # visual graph dataset has to be referenced for the creation of a concept folder, there is no need to load 
        # the dataset here.
        if 'elements' in concept:
            
            for data in concept['elements']:
                # This function removes all the redundant information from the visual graph element dict aka all the 
                # information that is already contained in the dataset anyways. So that after this function the resulting 
                # leftover dict only contains the information that was added during the concept creation process.
                strip_graph_data(data)
                
        if 'graphs' in concept:
            del concept['graphs']
        
        if 'image_paths' in concept:
            del concept['image_paths']
                
        # ~ concept metadata
        metadata_path = os.path.join(concept_path, 'metadata.json')
        with open(metadata_path, 'w') as file:
            json.dump(concept, file, cls=NumericJsonEncoder)
        
    
class ConceptReader():
    
    def __init__(self, 
                 path: str,
                 dataset: t.Union[str, dict, None] = None,
                 model: t.Union[Megan, None] = None,
                 logger: logging.Logger = NULL_LOGGER,
                 reader_cls: type = VisualGraphDatasetReader,
                 model_cls: type = Megan,
                 ):
        
        self.path = path
        self.dataset = dataset
        self.model = model
        self.logger = logger
        self.reader_cls = reader_cls
        self.model_cls = model_cls
        
        # This will later hold the dictionary structure of the global concept clustering metadata. This 
        # will be metadata that is not attached to any particular concept but rather additional information about 
        # all the clusters together.
        # This will be populated in the "read_metadata" method.
        self.metadata: t.Optional[str] = None
        
        # This will later hold the index data map of the visual graph dataset that is used as the basis for the 
        # concept clustering. This will be populated in the "load_dataset" method.
        self.index_data_map: t.Optional[dict] = None
        
        # In this dictionary we are creating a map where the keys are the integer indices of the concepts and the 
        # values are the corresponding absolute paths to the concept folders.
        self.index_path_map: t.Dict[int, str] = {}
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path) and (index := safe_int(file)) is not None:
                self.index_path_map[index] = file_path
        
    def read_metadata(self) -> dict:
        
        metadata_path = os.path.join(self.path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as file:
                self.metadata = json.load(file)
                
        return self.metadata
        
    def load_dataset(self) -> None:
        # If the given "dataset" is a dict, it will be assumed that this is directly the already loaded 
        # index_data_map representation of the dataset.
        if isinstance(self.dataset, dict):
            self.index_data_map = self.dataset
            return
        
        # If the value is instead a string, then it will be assumed to be the absolute string path to the 
        # dataset, in which case we can create a new reader instance to load it into memory
        elif isinstance(self.dataset, str):
            dataset_path = self.dataset
        
        # The last option is that no dataset has been given as a parameter, in this case we will assume that 
        # the dataset is referenced in the metadata of the concept clustering itself.
        elif isinstance(self.dataset, None):
            dataset_path = self.metadata['dataset_path']
            dataset_path = resolve_path(dataset_path, self.path)

        assert os.path.exists(dataset_path), f'dataset path "{dataset_path}" does not exist!'
        assert os.path.isdir(dataset_path), f'dataset path "{dataset_path}" is not a directory!'

        reader = self.reader_cls(
            path=self.dataset,
            logger=self.logger,   
        )
        self.index_data_map = reader.read()
        
    def load_model(self) -> None:
        
        # We only need to do something if the model is not None. In that case we will assume that a model has been 
        # passed as a parameter externally and we will just use that. Otherwise we will assume that the model is
        # referenced in the metadata of the concept clustering itself.
        if self.model is None:
            
            model_path = resolve_path(self.metadata['model_path'])
            assert model_path and os.path.exists(model_path), 'The saved model path does not exist!'
            
            self.model_cls.load_from_checkpoint(model_path)
        
    def read(self) -> tg.ConceptData:
        
        assert os.path.exists(self.path), f'concept data path does not exist!'
        assert os.path.isdir(self.path), f'concept data path is not a directory!'
        assert os.listdir(self.path) != 0, f'concept data path is empty directory!'
        
        # This method will read the metadata.json file which contains metadata for the global concept 
        # clustering. This metadata will be saved in the self.metadata attribute.
        self.read_metadata()
        
        # This method will load the dataset which this concept clustering references. There are multiple options 
        # of how this is done either by passing it directly or by passing only a string path. However, after this 
        # method completes successfully, the dataset will be loaded into the self.index_data_map attribute.
        self.load_dataset()
        
        # This method will load the model from the disk so that the self.model attribute is populated with the actual 
        # model on which the concepts are based on.
        self.load_model()
        
        # In this list we will store all the concept dicts that we read from the file system and 
        # this will also be the result of the loading process.        
        concepts: t.List[int] = []

        # "safe_int" is a utility function that will convert a string to an integer but does not raise
        # an exception if the string is not a valid integer. Instead it will return None in that case.
        
        # This list contains the string names of the direct members of the given directory which are 
        # valid integers in their correct integer order. This is because as a soft condition all concept 
        # folders' names should be their intger indices. 
        elements = [element for element in os.listdir(self.path) 
                    if os.path.isdir(os.path.join(self.path, element)) and safe_int(element) is not None]
        elements.sort(key=lambda element: int(element))
        
        index: int = 0
        for element in elements:
            element_path = os.path.join(self.path, element)
            if os.path.isdir(element_path):
                self.logger.info(f' * reading concept {index}')
                concept = self.read_concept_from_path(
                    concept_path=element_path,
                )
                concepts.append(concept)
                index += 1
                
        return concepts
            
    def read_concept(self, index: int) -> tg.ConceptData:
        concept_path = self.index_path_map[index]
        return self.read_concept_from_path(concept_path)
            
    def read_concept_from_path(self, concept_path: str) -> tg.ConceptDict:
        
        # ~ required: metadata file     
        # The one thing that this concept path folder should absolutely contain is a metadata.json file.
        # This json file should contain the concept dictionary itself - or at least the basic structure of 
        # it with all the elements that can actually be JSON encoded.
        
        metadata_path = os.path.join(concept_path, 'metadata.json')
        assert os.path.exists(metadata_path), f'concept metadata file for {concept_path} does not exist!'
        
        with open(metadata_path, 'r') as file:
            concept: tg.ConceptDict = json.load(file)
            
        # ~ loading graph data
        # The main amount of the graph data is stored in the "elements" list. This list contains one dict entry 
        # for every element of the concept cluster. Each of these dicts are stripped down versions of the original 
        # visual graph elements that represent the graphs. All the actual data regarding the graph structure 
        # had been removed and now we load that again from the visual graph dataset.
        
        self.logger.info(f'   querying the model with concept elements...')
        elements = concept['elements']
        indices = [element['metadata']['index'] for element in elements]
        graphs = deepcopy([self.index_data_map[index]['metadata']['graph'] for index in indices])
        
        self.logger.info(f'   updating graph information...')
        self.update_graphs(graphs)
        for index, element, graph in zip(indices, elements, graphs):
            element.update(deepcopy(self.index_data_map[index]))
            element['metadata']['graph'] = graph
            
        # ~ optional
        # Everything else is optional and not guaranteed to be contained within the folder.
        
        prototypes_path = os.path.join(concept_path, 'prototypes')
        if os.path.exists(prototypes_path):
            self.logger.info('   loading prototypes...')
            reader = self.reader_cls(prototypes_path)
            index_data_map = reader.read()
            
            prototypes = [data for data in index_data_map.values()]
            prototype_graphs = [data['metadata']['graph'] for data in prototypes]
            self.update_graphs(prototype_graphs)
            
            concept['prototypes'] = prototypes
            
            
        return concept
        
    def update_graphs(self, graphs: t.List[dict]) -> t.List[dict]:
        
        infos = self.model.forward_graphs(graphs)
        devs = self.model.leave_one_out_deviations(graphs)
        
        for graph, info, dev in zip(graphs, infos, devs):
            graph['node_importances'] = info['node_importance']
            graph['edge_importances'] = info['edge_importance']
            graph['graph_prediction'] = info['graph_output']
            graph['graph_embedding'] = info['graph_embedding']
            graph['graph_deviation'] = dev
            
        return graphs