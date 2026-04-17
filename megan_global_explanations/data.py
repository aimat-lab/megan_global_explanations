"""
This module defines the data structures and classes that are used to represent the custom concept data.
Beyond the data structures themselves, this module also defines the methods that are used to load and 
save the data from and to the file system. The data is stored in a JSON file format and also partially 
based on the visual graph dataset format to represent the concept prototypes for example.
"""
import io
import os
import json
import shutil
import logging
import zipfile
import collections
import typing as t
import typing as typ
import visual_graph_datasets.typing as tv
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import fcluster
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


"""
A concept dict contains information about a single concept (cluster). This information includes 
the following keys:

- index: The integer index of the concept within the scope of the other concepts.
- channel_index: The integer index of the Megan model's channel that this concept was extracted from.
- centroid: A numpy float array of shape (D, ) representing the cluster centroid (the average embedding of the cluster), 
    where D is the dimensionality of the latent space.
- embeddings: A numpy float array of shape (N, D) containing the embeddings of all the elements that are associated 
    with this concept, where N is the number of elements and D is the dimensionality of the latent space.
- elements: A list of all the elements associated with the concept where each element is a GraphDict representation.
- prototypes: (optional) A list of all the prototypes associated with the concept where each prototype is a 
    GraphDict representation.
"""
ConceptDict = t.Dict[str, t.Any]


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
                 write_elements: bool = False,
                 ):
        self.path = path
        self.processing = processing
        self.model = model
        self.logger = logger
        self.writer_cls = writer_cls
        self.write_elements = write_elements
        
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
        
        value = str(value)
                
        writer: VisualGraphDatasetWriter = self.writer_cls(
            path=path,
        )
                
        # To actually write the given graph as a visual graph element (a visualization PNG file and a 
        # metadata JSON file) we use the corresponding Processing instance. The "create" method will 
        # take care of this.
        try:
            self.processing.create(
                value=value,
                graph=graph,
                output_path=path,
                index=index,
                additional_metadata=additional_metadata,
                writer=writer,
            )
        except TypeError as exc:
            print(f'Processing of the graph "{value}"({type(value)}) failed with exception: {type(exc)}{exc}')
                
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
                    additional_metadata=prototype['metadata'],
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
            
            # However! Optionally, if the corresponding flag is set, we DO want to save the graphs in their 
            # entirety in the form of a visual graph dataset. This is mainly required for cases where we want 
            # to export the concepts but really cant get access to the original dataset...
            if self.write_elements:
                
                elements_path = os.path.join(concept_path, 'elements')
                os.mkdir(elements_path)
                for index, element in enumerate(concept['elements']):
                    if 'graph' in element['metadata']:
                        graph = element['metadata']['graph']
                        self.write_graph(
                            graph=graph,
                            index=index,
                            path=elements_path,
                            additional_metadata=element['metadata'],
                        ) 
            
            concept['elements'] = deepcopy(concept['elements'])
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
        
        dataset_path = None
        
        # If the value is instead a string, then it will be assumed to be the absolute string path to the 
        # dataset, in which case we can create a new reader instance to load it into memory
        if isinstance(self.dataset, str):
            dataset_path = self.dataset

        if dataset_path is not None:
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
            
            self.model_cls.load(model_path)
        
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
        
        # 13.09.24 - we'll allow the dataset/index_data_map to be none, BUT if that is the case we expect that 
        # there exists a "elements" folder in the concept folder which contains the elements directly in the 
        # format of a visual graph dataset.
        if self.index_data_map is not None:
            self.logger.info(f'   loading graph information from dataset...')
            elements = concept['elements']
            indices = [element['metadata']['index'] for element in elements]
            graphs = deepcopy([self.index_data_map[index]['metadata']['graph'] for index in indices])
        
            for index, element, graph in zip(indices, elements, graphs):
                element.update(deepcopy(self.index_data_map[index]))
                element['metadata']['graph'] = graph
        
        else:
            self.logger.info('    no dataset. loading graphs directly...')

            elements_path = os.path.join(concept_path, 'elements')
            assert os.path.exists(elements_path), f'no dataset given and concept is missing "elements" folder!'
            
            # With this folder we can simply use the reader instance to construct our graph instances.
            reader = self.reader_cls(elements_path)
            index_data_map = reader.read()
            concept['elements'] = list(index_data_map.values())
            graphs = [data['metadata']['graph'] for data in index_data_map.values()]
        
        # The "update_graphs" method will use the loaded model to attach additional information to the graphs
        # such as the predictions, the embeddings etc.
        self.logger.info(f'   updating graph information with model...')
        self.update_graphs(graphs)
            
        # ~ optional: prototypes
        # Optionally (not always) each concept folder may also contain an additional "prototypes" folder. This 
        # folder contains the prototype elements of the concept cluster. These are stripped down elements which 
        # are meant to represent the underlying pattern of the cluster directly.
        
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
            graph['graph_output'] = info['graph_output']
            graph['graph_embedding'] = info['graph_embedding']
            graph['graph_deviation'] = dev

        return graphs


# =====================================================================
# === Clustering archive (.clu) ======================================
# =====================================================================


def select_representatives(
    distances: np.ndarray,
    n: int,
    strategy: str = 'closest',
    temperature: float = 0.5,
) -> np.ndarray:
    """Select representative member indices from a cluster.

    :param distances: (N,) array of distances from each member to the centroid.
    :param n: Number of representatives to select (capped at ``len(distances)``).
    :param strategy:
        - ``'closest'``: deterministic — the *n* nearest members.
        - ``'temperature'``: stochastic — softmax over negative distances with
          temperature auto-scaled as ``temperature * median(distances)``.
          Low values (e.g. 0.5) bias heavily toward the centroid; high values
          (e.g. 2.0+) approach uniform sampling.
    :param temperature: Temperature multiplier (used only for ``'temperature'``
        strategy). Multiplied by the median distance to produce the actual
        softmax temperature.
    :returns: Integer index array of selected members.
    """
    n = min(n, len(distances))
    if n <= 0:
        return np.array([], dtype=int)

    if strategy == 'closest':
        return np.argsort(distances)[:n]

    if strategy == 'temperature':
        median_dist = float(np.median(distances))
        t = temperature * max(median_dist, 1e-9)
        logits = -np.asarray(distances, dtype=np.float64) / t
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        return np.random.choice(len(distances), size=n, replace=False, p=probs)

    raise ValueError(f"Unknown representative strategy: {strategy!r}")


class ClusterView:
    """Read-only dict-like + attribute-access view over a single cluster's data.

    Returned by ``Clustering[cluster_id]``. Supports both ``cluster['centroid']``
    and ``cluster.centroid`` access patterns, as well as iteration over keys.
    """

    def __init__(self, data: dict):
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, '_data', data)

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"ClusterView has no attribute '{name}'")

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self) -> str:
        return f"ClusterView('{self._data.get('id', '?')}')"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays (small ones only)."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray) and obj.size < 100:
            return obj.tolist()
        return super().default(obj)


def _npy_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _bytes_to_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


class Clustering:
    """Self-contained clustering result with scoring, hierarchy, and serialization.

    Holds per-channel embeddings, HDBSCAN labels, cluster centroids & members,
    and optional agglomerative linkage matrices. Serializes to a ``.clu`` zip
    archive with human-readable JSON metadata and binary ``.npy``/``.npz`` arrays.

    Quick start::

        # Load from archive
        clustering = Clustering.load('clustering.clu')

        # Score a channel embedding
        scores = clustering.score(embedding, channel=0, method='knn', k=5)

        # Access individual clusters
        cl = clustering['ch0_cl3']
        print(cl.centroid.shape, cl.annotations)

        # Merge at 80% of max linkage distance
        merged = clustering.at_linkage(80)
        merged_scores = merged.score(embedding, channel=0)

        # Get the hierarchy as a networkx DiGraph
        tree = clustering.get_tree(channel=0)
    """

    SCHEMA_VERSION = 1

    def __init__(self):
        self.channels: t.Dict[int, dict] = {}
        self.embedding_dim: int = 0
        self.schema_version: int = self.SCHEMA_VERSION

        # Populated by _build_index()
        self.cluster_index: t.Dict[str, dict] = {}

        # For at_linkage() views
        self.parent: t.Optional['Clustering'] = None
        self.merged_map: t.Optional[t.Dict[str, t.List[str]]] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_experiment(
        cls,
        cluster_infos: t.List[dict],
        channel_infos: t.Dict[int, dict],
        clustering_metric: str,
        channel_embeddings_map: t.Dict[int, np.ndarray],
        channel_labels: t.Dict[int, np.ndarray],
        cluster_hierarchy: t.Optional[t.Dict[int, np.ndarray]] = None,
    ) -> 'Clustering':
        """Build a :class:`Clustering` from in-memory experiment data.

        This is the primary construction path used at the end of the
        ``vgd_concept_extraction`` experiment, where all required data
        structures are already in scope.
        """
        obj = cls()

        # Derive embedding_dim from the first available channel
        for ch_idx, emb in channel_embeddings_map.items():
            obj.embedding_dim = emb.shape[1]
            break

        # Group cluster_infos by channel
        ch_clusters: t.Dict[int, t.List[dict]] = {}
        for info in cluster_infos:
            ch_clusters.setdefault(info['channel_index'], []).append(info)

        for ch_idx in sorted(set(channel_embeddings_map.keys()) | set(ch_clusters.keys())):
            ch_info = channel_infos.get(ch_idx, {})
            infos = sorted(ch_clusters.get(ch_idx, []), key=lambda i: i['index'])

            clusters = []
            for info in infos:
                cl_dict: dict = {
                    'index': info['index'],
                    'hdbscan_label': info['hdbscan_label'],
                    'centroid': np.asarray(info['centroid']),
                    'members': np.asarray(info['embeddings']),
                    'metadata': {
                        'name': info.get('name', ''),
                        'color': info.get('color', ''),
                    },
                    'annotations': {},
                }
                if 'representatives' in info:
                    cl_dict['representatives'] = info['representatives']
                if 'member_stats' in info:
                    cl_dict['member_stats'] = info['member_stats']
                clusters.append(cl_dict)

            obj.channels[ch_idx] = {
                'name': ch_info.get('name', f'channel_{ch_idx}'),
                'color': ch_info.get('color', 'gray'),
                'metric': clustering_metric,
                'embeddings': np.asarray(channel_embeddings_map.get(ch_idx, np.empty((0, obj.embedding_dim)))),
                'labels': np.asarray(channel_labels.get(ch_idx, np.empty(0, dtype=int))),
                'linkage': cluster_hierarchy.get(ch_idx) if cluster_hierarchy else None,
                'clusters': clusters,
            }

        obj._build_index()
        return obj

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str, max_members: int = 2000) -> None:
        """Write the clustering to a ``.clu`` zip archive.

        :param path: Output file path (e.g. ``'clustering.clu'``).
        :param max_members: Per-cluster member cap. Clusters with more members
            are randomly subsampled to this count in the saved archive only;
            the in-memory object is not modified.
        """
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            meta = {
                'schema_version': self.schema_version,
                'embedding_dim': self.embedding_dim,
                'channels': sorted(self.channels.keys()),
            }
            zf.writestr('meta.json', json.dumps(meta, indent=2, cls=NumpyEncoder))

            for ch_idx, ch_data in self.channels.items():
                prefix = f'ch{ch_idx}'

                # channel.json
                zf.writestr(f'{prefix}/channel.json', json.dumps({
                    'name': ch_data['name'],
                    'color': ch_data['color'],
                    'metric': ch_data['metric'],
                }, indent=2))

                # clusters.json (includes representatives as nested lists)
                clusters_json = []
                for cl in ch_data['clusters']:
                    entry = {
                        'index': cl['index'],
                        'hdbscan_label': cl['hdbscan_label'],
                        'metadata': cl.get('metadata', {}),
                        'annotations': cl.get('annotations', {}),
                    }
                    if 'representatives' in cl:
                        entry['representatives'] = cl['representatives']
                    clusters_json.append(entry)
                zf.writestr(f'{prefix}/clusters.json',
                            json.dumps(clusters_json, indent=2, cls=NumpyEncoder))

                # centroids.npy — (K, D) stacked
                if ch_data['clusters']:
                    centroids = np.stack([cl['centroid'] for cl in ch_data['clusters']])
                    zf.writestr(f'{prefix}/centroids.npy', _npy_to_bytes(centroids))

                # members.npz — one key per cluster, subsampled if needed
                members_dict = {}
                for i, cl in enumerate(ch_data['clusters']):
                    m = np.asarray(cl['members'])
                    if len(m) > max_members:
                        idx = np.random.choice(len(m), size=max_members, replace=False)
                        m = m[idx]
                    members_dict[str(i)] = m

                buf = io.BytesIO()
                np.savez_compressed(buf, **members_dict)
                zf.writestr(f'{prefix}/members.npz', buf.getvalue())

                # member_stats.npz — pre-computed per-cluster histogram data
                stats_dict = {}
                for i, cl in enumerate(ch_data['clusters']):
                    ms = cl.get('member_stats')
                    if ms:
                        for key, arr in ms.items():
                            stats_dict[f'cl{i}_{key}'] = np.asarray(arr)
                if stats_dict:
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **stats_dict)
                    zf.writestr(f'{prefix}/member_stats.npz', buf.getvalue())

                # embeddings.npy
                zf.writestr(f'{prefix}/embeddings.npy',
                            _npy_to_bytes(ch_data['embeddings']))

                # labels.npy
                zf.writestr(f'{prefix}/labels.npy',
                            _npy_to_bytes(ch_data['labels']))

                # linkage.npy (optional)
                if ch_data.get('linkage') is not None:
                    zf.writestr(f'{prefix}/linkage.npy',
                                _npy_to_bytes(ch_data['linkage']))

    @classmethod
    def load(cls, path: str) -> 'Clustering':
        """Load a :class:`Clustering` from a ``.clu`` zip archive."""
        obj = cls()

        with zipfile.ZipFile(path, 'r') as zf:
            meta = json.loads(zf.read('meta.json'))
            obj.schema_version = meta['schema_version']
            obj.embedding_dim = meta['embedding_dim']

            for ch_idx in meta['channels']:
                prefix = f'ch{ch_idx}'

                ch_info = json.loads(zf.read(f'{prefix}/channel.json'))
                clusters_json = json.loads(zf.read(f'{prefix}/clusters.json'))

                centroids = _bytes_to_npy(zf.read(f'{prefix}/centroids.npy')) \
                    if f'{prefix}/centroids.npy' in zf.namelist() else np.empty((0, obj.embedding_dim))

                members_npz = np.load(io.BytesIO(zf.read(f'{prefix}/members.npz')),
                                      allow_pickle=False)

                embeddings = _bytes_to_npy(zf.read(f'{prefix}/embeddings.npy'))
                labels = _bytes_to_npy(zf.read(f'{prefix}/labels.npy'))

                linkage_key = f'{prefix}/linkage.npy'
                linkage_mat = _bytes_to_npy(zf.read(linkage_key)) \
                    if linkage_key in zf.namelist() else None

                # member_stats.npz (optional)
                stats_key = f'{prefix}/member_stats.npz'
                stats_npz = np.load(io.BytesIO(zf.read(stats_key)), allow_pickle=False) \
                    if stats_key in zf.namelist() else None

                clusters = []
                for i, cl_json in enumerate(clusters_json):
                    cl_dict = {
                        **cl_json,
                        'centroid': centroids[i] if i < len(centroids) else np.zeros(obj.embedding_dim),
                        'members': members_npz[str(i)],
                    }
                    # Reconstruct member_stats from the npz
                    if stats_npz is not None:
                        ms = {}
                        for stat_name in ('graph_outputs', 'graph_deviations',
                                          'mask_sizes', 'centroid_distances'):
                            npz_key = f'cl{i}_{stat_name}'
                            if npz_key in stats_npz:
                                ms[stat_name] = stats_npz[npz_key]
                        if ms:
                            cl_dict['member_stats'] = ms
                    clusters.append(cl_dict)

                obj.channels[ch_idx] = {
                    'name': ch_info['name'],
                    'color': ch_info['color'],
                    'metric': ch_info['metric'],
                    'embeddings': embeddings,
                    'labels': labels,
                    'linkage': linkage_mat,
                    'clusters': clusters,
                }

        obj._build_index()
        return obj

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Populate ``cluster_index`` mapping cluster IDs to their data."""
        self.cluster_index = {}

        if self.merged_map is not None:
            # Merged view: build super-cluster entries from parent's leaf data
            for super_id, leaf_ids in self.merged_map.items():
                leaf_data = [self.parent.cluster_index[lid] for lid in leaf_ids]
                ch = leaf_data[0]['channel']
                self.cluster_index[super_id] = {
                    'id': super_id,
                    'channel': ch,
                    'index': super_id,
                    'centroid': np.mean([d['centroid'] for d in leaf_data], axis=0),
                    'members': np.concatenate([d['members'] for d in leaf_data]),
                    'leaves': leaf_ids,
                    'metadata': {},
                    'annotations': {},
                }
        else:
            # Base clustering: one entry per leaf cluster
            for ch_idx, ch_data in self.channels.items():
                for cl in ch_data['clusters']:
                    cl_id = f"ch{ch_idx}_cl{cl['index']}"
                    self.cluster_index[cl_id] = {
                        'id': cl_id,
                        'channel': ch_idx,
                        **cl,
                    }

    # ------------------------------------------------------------------
    # Cluster access
    # ------------------------------------------------------------------

    def __getitem__(self, cluster_id: str) -> ClusterView:
        if cluster_id not in self.cluster_index:
            raise KeyError(f"Unknown cluster ID: {cluster_id!r}")
        return ClusterView(self.cluster_index[cluster_id])

    def __iter__(self) -> t.Iterator[str]:
        return iter(self.cluster_index)

    def __len__(self) -> int:
        return len(self.cluster_index)

    def __contains__(self, cluster_id: str) -> bool:
        return cluster_id in self.cluster_index

    def to_cluster_infos(self) -> t.List[dict]:
        """Reconstruct a ``cluster_infos``-style list for ``create_concept_cluster_report``.

        Each entry is a dict with ``index``, ``channel_index``, ``centroid``,
        ``embeddings``, ``index_tuples``, ``graphs``, and ``image_paths`` — the
        shape expected by the visualization module's report function.

        When full graph data is not stored (e.g. after loading from a ``.clu``
        archive), ``graphs`` is built from the stored **representatives**: each
        representative's importances, predictions, and deviations are packed
        into a synthetic graph dict. ``image_paths`` will be empty in that case
        (the report renderer can still produce images from SMILES + importances
        if a processing class is provided).
        """
        result: t.List[dict] = []

        for ch_idx in sorted(self.channels.keys()):
            ch_data = self.channels[ch_idx]
            for cl in ch_data['clusters']:
                idx = cl['index']
                channel = ch_idx
                centroid = cl['centroid']
                members = cl['members']

                # Try to use stored representatives to build graph dicts
                reps = cl.get('representatives', [])
                graphs: t.List[dict] = []
                image_paths: t.List[str] = []
                index_tuples: t.List[t.Tuple[int, int]] = []

                for rep in reps:
                    node_imp = np.array(rep['node_importances'])
                    edge_imp = np.array(rep['edge_importances'])
                    graph_out = np.array(rep['graph_output'])
                    graph_dev = np.array(rep['graph_deviation'])
                    n_nodes = len(node_imp)
                    graphs.append({
                        'graph_repr': rep.get('smiles', ''),
                        'node_importances': node_imp,
                        'edge_importances': edge_imp,
                        'graph_output': graph_out,
                        'graph_deviation': graph_dev,
                        'node_indices': list(range(n_nodes)),
                    })
                    image_paths.append('')
                    index_tuples.append((rep.get('dataset_index', 0), channel))

                result.append({
                    'index': idx,
                    'channel_index': channel,
                    'centroid': centroid,
                    'embeddings': members,
                    'index_tuples': index_tuples,
                    'graphs': graphs,
                    'image_paths': image_paths,
                    'name': cl.get('metadata', {}).get('name', ''),
                    'color': cl.get('metadata', {}).get('color', ''),
                })

        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        embedding: np.ndarray,
        channel: int,
        method: str = 'knn',
        k: int = 5,
    ) -> t.Dict[str, float]:
        """Score an embedding against all clusters in a channel.

        :param embedding: A (D,) embedding vector for the given channel.
        :param channel: Explanation-channel index.
        :param method: ``'knn'`` (mean of k nearest member distances) or
            ``'distance'`` (distance to centroid).
        :param k: Number of neighbours for knn mode.
        :returns: ``{cluster_id: score}`` dict. Lower = better fit.
        """
        # Resolve the metric from the channel data. For merged views the
        # channel data is shared with the parent.
        ch_data = self.channels[channel] if channel in self.channels \
            else self.parent.channels[channel]
        metric = ch_data['metric']
        emb = np.asarray(embedding).reshape(1, -1).astype(np.float64)

        if self.merged_map is not None:
            # Merged view: compute leaf scores first, aggregate via min
            leaf_scores = self.parent.score(embedding, channel, method=method, k=k)
            results: t.Dict[str, float] = {}
            for super_id, leaf_ids in self.merged_map.items():
                relevant = [leaf_scores[lid] for lid in leaf_ids
                            if lid in leaf_scores]
                if relevant:
                    results[super_id] = min(relevant)
            return results

        # Base clustering: score against each leaf cluster in this channel
        results = {}
        for cl_id, cl_data in self.cluster_index.items():
            if cl_data['channel'] != channel:
                continue

            members = np.asarray(cl_data['members'])
            if method == 'knn':
                k_actual = min(k, len(members))
                if k_actual == 0:
                    continue
                dists = pairwise_distances(emb, members, metric=metric)[0]
                score = float(np.partition(dists, k_actual - 1)[:k_actual].mean())
            elif method == 'distance':
                centroid = cl_data['centroid'].reshape(1, -1)
                score = float(pairwise_distances(emb, centroid, metric=metric)[0, 0])
            else:
                raise ValueError(f"Unknown scoring method: {method!r}")

            results[cl_id] = score

        return results

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def at_linkage(self, percent: float) -> 'Clustering':
        """Return a view with clusters merged at *percent* % of max merge distance.

        The returned ``Clustering`` shares the underlying channel data with
        ``self`` — no arrays are copied. Merged super-clusters get IDs like
        ``'ch0_s0'``; single-leaf super-clusters keep their original ID.
        Score aggregation uses **min** across constituent leaves.

        :param percent: Cut height as a percentage of max merge distance (0–100).
        """
        view = Clustering.__new__(Clustering)
        view.channels = self.channels
        view.embedding_dim = self.embedding_dim
        view.schema_version = self.schema_version
        view.parent = self
        view.merged_map = {}

        for ch_idx, ch_data in self.channels.items():
            Z = ch_data.get('linkage')
            clusters = sorted(ch_data['clusters'], key=lambda c: c['index'])

            if Z is None or len(clusters) < 2:
                for cl in clusters:
                    leaf_id = f"ch{ch_idx}_cl{cl['index']}"
                    view.merged_map[leaf_id] = [leaf_id]
                continue

            max_d = float(Z[:, 2].max())
            t_cut = (percent / 100.0) * max_d
            grouping = fcluster(Z, t=t_cut, criterion='distance')

            groups: t.Dict[int, t.List[dict]] = {}
            for leaf_idx, g in enumerate(grouping):
                groups.setdefault(int(g), []).append(clusters[leaf_idx])

            super_idx = 0
            for g_id, member_clusters in groups.items():
                leaf_ids = [f"ch{ch_idx}_cl{c['index']}" for c in member_clusters]
                if len(leaf_ids) == 1:
                    super_id = leaf_ids[0]
                else:
                    super_id = f"ch{ch_idx}_s{super_idx}"
                    super_idx += 1
                view.merged_map[super_id] = leaf_ids

        view._build_index()
        return view

    def get_tree(self, channel: int):
        """Return a ``networkx.DiGraph`` representing the cluster hierarchy.

        Leaf nodes are named ``'ch{N}_cl{M}'`` with attributes ``type``,
        ``channel``, ``index``, ``centroid``, ``size``, ``annotations``.
        Internal (merge) nodes are named ``'ch{N}_m{i}'`` with attributes
        ``type``, ``merge_distance``, ``size``. Edges go from parent to child
        with ``weight = merge_distance``.
        """
        import networkx as nx

        ch_data = self.channels[channel]
        clusters = sorted(ch_data['clusters'], key=lambda c: c['index'])
        Z = ch_data.get('linkage')

        G = nx.DiGraph()

        for cl in clusters:
            cl_id = f"ch{channel}_cl{cl['index']}"
            G.add_node(cl_id,
                       type='leaf',
                       channel=channel,
                       index=cl['index'],
                       centroid=cl['centroid'],
                       size=len(cl['members']),
                       annotations=cl.get('annotations', {}))

        if Z is None or len(clusters) < 2:
            return G

        K = len(clusters)
        for merge_idx, row in enumerate(Z):
            left, right, dist, count = int(row[0]), int(row[1]), float(row[2]), int(row[3])
            merge_id = f"ch{channel}_m{merge_idx}"

            child_a = f"ch{channel}_cl{clusters[left]['index']}" if left < K \
                else f"ch{channel}_m{left - K}"
            child_b = f"ch{channel}_cl{clusters[right]['index']}" if right < K \
                else f"ch{channel}_m{right - K}"

            G.add_node(merge_id,
                       type='merge',
                       channel=channel,
                       merge_distance=dist,
                       size=count)
            G.add_edge(merge_id, child_a, weight=dist)
            G.add_edge(merge_id, child_b, weight=dist)

        return G

    def get_forest(self):
        """Return a ``networkx.DiGraph`` containing all channels as disconnected trees."""
        import networkx as nx

        G = nx.DiGraph()
        for ch_idx in self.channels:
            tree = self.get_tree(ch_idx)
            G = nx.compose(G, tree)
        return G