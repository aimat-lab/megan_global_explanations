"""
This is the base experiment which implements the querying and explaining of a single visual graph element 
based on a combination of a Megan model, a visual graph dataset and a previously constructed concept clustering.

The model, dataset and concept clustering are all loaded from the disk. The visual graph element for the query 
is constructed from a given string representation.

The model is then queried with the graph representation to get the local explanations and the graph embeddings 
are used to select the closest concept cluster which will act as the global explanation. This information will be 
visualized in a single plot at the end.
"""
import os
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from graph_attention_student.torch.megan import Megan

from megan_global_explanations.data import ConceptReader


PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == QUERY PARAMETERS ==
# The query parameters are the parameters that are used to query the model.

# :param QUERY_ELEMENT:
#       The string domain representation of the element with which to query the model. 
#       For this element the explanations will be created.
QUERY_ELEMENT: str = 'Y(R)(R)(R)HHHB-1GG-1'
# :param QUERY_TYPE:
#       This string determines whether the given prediction task is a regression or a classification
#       problem, which in turn determines the details of the experiment such as how certain values 
#       will have to be calculates or what kinds of visualizations will be created.
QUERY_TYPE: str = 'regression' # 'classification'

# == DATASET PARAMETERS ==
# The dataset parameters are the parameters that are used to load the dataset. 

# :param VISUAL_GRAPH_DATASET:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'
# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'rb_dual_motifs.ckpt')
# :param CONCEPTS_PATH:
#       This has to be the absolute string path to the concept clustering data that is to be used for the
#       concept clustering. This data is typically created by the concept clustering process and then
#       stored on the disk as a folder.
CONCEPTS_PATH: str = os.path.join(ASSETS_PATH, 'concepts', 'rb_dual_motifs')

# == VISUALIZATION PARAMETERS
# These parameters determine the details for the visualization of the results.

# :param NUM_EXAMPLES:
#       The number of examples to be shown in the visualization. This is the number of examples that will be
#       shown for each concept.
NUM_EXAMPLES: int = 6
# :param EXAMPLE_MODE:
#       This determines the mode in which the examples are selected. This may either be "centroid" or "random".
#       If "centroid" is selected, the examples will be selected based on the distance to the centroid of the
#       concept. If "random" is selected, the examples will be selected randomly from the concept.
EXAMPLE_MODE: str = 'centroid'  # 'random'

__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset', replace=False)
def load_dataset(e: Experiment,
                 path: str,
                 ) -> dict:
    """
    This hook takes a local path to a (visual graph) dataset as the only argument and is then 
    responsibe for loading and returning that dataset as a index_data_map.
    
    Additionally, this function has to set up the experiment values "node_dim", "edge_dim" and "out_dim"
    based on the dataset that has been loaded.
    
    This default implementation uses the default VisualGraphDatasetReader to load the dataset from the disk.
    """
    reader = VisualGraphDatasetReader(
        path=path,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    processing = reader.read_process().processing
    
    example_graph = list(index_data_map.values())[0]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]
    e['out_dim'] = example_graph['graph_labels'].shape[0]
    e.log(f'loaded dataset with {e["node_dim"]} node features and {e["edge_dim"]} edge features')
    
    return index_data_map, processing


@experiment.hook('load_model')
def load_model(e: Experiment,
               path: str
               ) -> Megan:
    """
    This hook receives a local file system path as the only argument and is supposed to load the 
    MEGAN model from that path and return the instance.
    
    This standard implementation just loads the default Megan torch implementation
    """
    model = Megan.load_from_checkpoint(path)
    return model


@experiment.hook('select_examples')
def select_examples(e: Experiment,
                    concept: dict,
                    **kwargs) -> t.List[dict]:
    """
    This hook receives a concept and is supposed to return a list of the examples that are to be shown
    in the format of visual graph element dicts.
    
    This standard implementation only implements the "random" or the "centroid" strategies.
    """    
    centroid = concept['centroid']
    elements = concept['elements']
    channel_index = concept['channel_index']
    
    num_examples = min(len(elements), e.NUM_EXAMPLES)
    
    if e.EXAMPLE_MODE == 'centroid':
        
        # embeddings: (num_elements, embedding_dim)
        embeddings = [element['metadata']['graph']['graph_embedding'][:, channel_index] for element in elements]

        # Get the index of the NUM_EXAMPLES closest elements to the centroid
        distances = [cosine(centroid, embedding) for embedding in embeddings]
        indices = np.argsort(distances)[:num_examples]
        
        return [elements[index] for index in indices]
    
    elif e.EXAMPLE_MODE == 'random':
        
        return np.random.sample(elements, size=num_examples)


@experiment
def experiment(e: Experiment):
    e.log('staring experiment')

    # ~ loading the dataset
    # The dataset is either loaded from the local file system as a path or it is downloaded 
    # from the remote file share first by providing it's unique string identifier.
    
    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
        
    else:
        config = Config()
        config.load()
        
        dataset_path = ensure_dataset(
            dataset_name=e.VISUAL_GRAPH_DATASET,
            config=config,
            logger=e.logger,
        )
    
    e.log('loading dataset...')
    # :hook load_dataset:
    #       This hook is responsible for loading the dataset from the given path and is supposed to return
    #       the index_data_map and the processing instance.
    index_data_map, processing = e.apply_hook(
        'load_dataset',
        path=dataset_path,
    )
    num_graphs = len(index_data_map)
    e.log(f'loaded dataset with {num_graphs} elements')
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    indices = np.array([index for index in index_data_map.keys()])
    
    # ~ loading the model
    # Here we need to load the model which will then actually do the predictions
    
    e.log('loading model...')
    # :hook load_model:
    #       This hook is responsible for loading the model from the given path and is supposed to return
    #       the model instance.
    model: Megan = e.apply_hook(
        'load_model',
        path=e.MODEL_PATH,
    )
    e.log(f'loaded model of of class {model.__class__.__name__}')
    num_channels = model.num_channels
    
    # ~ loading the concepts
    # Finally we also need to load the persistent folder representation of the concept clustering 
    # results which were obtained for the loaded model and dataset combination.
    
    e.log('loading concepts...')
    concept_reader = ConceptReader(
        path=e.CONCEPTS_PATH,
        logger=e.logger,
        model=model,
        dataset=index_data_map,
    )
    concept_metadata = concept_reader.read_metadata()
    concept_reader.load_dataset()
    
    concepts = concept_metadata['concepts']
    concept_centroids = [concept['centroid'] for concept in concepts]
    e.log(f'loaded {len(concept_centroids)} concepts')
    
    # ~ querying the model
    # At this point we can now query the model for the actual prediction based on the given element.
    # However, for this to happen we first need to convert the given element into a graph representation
    # using the processing instance which we get from the dataset as well.
    
    e.log(f'processing the query element...')
    e.log(f'query: {e.QUERY_ELEMENT}')
    graph = processing.process(e.QUERY_ELEMENT)
    info = model.forward_graphs([graph])[0]
    
    pred = info['graph_output']
    graph['graph_prediction'] = pred
    graph['node_importances'] = info['node_importance']
    graph['edge_importances'] = info['edge_importance']
    
    # dev: (num_outputs, num_channels)
    dev = model.leave_one_out_deviations([graph])[0]
    # fid: (num_channels, )
    fid = np.zeros((num_channels, ))
    # depending on the type of prediction problem, the fidelity is defined differently.
    if e.QUERY_TYPE == 'regression':
        fid[0] = -dev[0, 0]
        fid[1] = +dev[0, 1]
    elif e.QUERY_TYPE == 'classification':
        fid = np.diag(dev)
            
    # ~ creating the explanations
    # With that information we can now create all the explanation artifacts for the given element. 
    # This includes for example the local explanation masks, but also includes a search for which concept 
    # cluster the elements embeddings are closest to!

    e.log('visualizing the query element...')
    fig, node_positions = processing.visualize_as_figure(
        value=e.QUERY_ELEMENT, 
        graph=graph, 
        width=1000, 
        height=1000
    )
    graph['node_positions'] = node_positions
    image_path = os.path.join(e.path, 'element.png')
    fig.savefig(image_path)

    e.log('visualizing local explanations...')
    fig, rows = plt.subplots(
        ncols=num_channels,
        nrows=1,
        figsize=(num_channels * 5, 5),
        squeeze=False,
    )
    fig.suptitle(f'Repr: "{e.QUERY_ELEMENT}"\n'
                 f'Prediction: {pred}')
    
    for channel_index in range(num_channels):
        
        ax = rows[0][channel_index]
        
        draw_image(
            ax=ax,
            image_path=image_path,
        )
        plot_node_importances_background(
            ax=ax,
            g=graph,
            node_positions=node_positions,
            node_importances=graph['node_importances'][:, channel_index]
        )
        plot_edge_importances_background(
            ax=ax,
            g=graph,
            node_positions=node_positions,
            edge_importances=graph['edge_importances'][:, channel_index]
        )
        ax.set_title(f'channel {channel_index} - fidelity: {fid[channel_index]}')
        
    expl_path = os.path.join(e.path, 'local_explanations.pdf')
    fig.savefig(expl_path)
    
    e.log('visualizing concept explanations...')
    fig, rows = plt.subplots(
        ncols=1 + e.NUM_EXAMPLES,
        nrows=num_channels,
        figsize=((e.NUM_EXAMPLES + 1) * 5, num_channels * 5),
        squeeze=False,
    )
    fig.suptitle(f'Explanations\n'
                 f'Prediction: {np.round(pred, 3)}')
    
    for channel_index in range(num_channels):
        
        e.log(f'determine closest concept for channel {channel_index}...')
        # channel_embedding: (embedding_dim, )
        channel_embedding = info['graph_embedding'][:, channel_index]
        
        min_index = None
        min_distance = float('inf')
        for concept_index, centroid in enumerate(concept_centroids):
            distance = cosine(centroid, channel_embedding)
            if distance < min_distance:
                min_distance = distance
                min_index = concept_index

        e.log(f'closest concept is {min_index} with distance {min_distance}')
        
        # Only now we actually load all the information about that particular concept into memory using the concept 
        # reader and the index that we have obtained.
        e.log(f'loading concept {min_index}...')
        concept = concept_reader.read_concept(min_index)
        
        # Each concept is drawin in its own row where the first figure is the concept prototype and the following 
        # figures are the examples from that concept cluster.
        e.log(f'drawing concept...')
        ax_ch = rows[channel_index][0]
        ax_ch.set_title(
            f'Predicted Explanation - Channel {channel_index}\n'
            f'Fidelity/Contribution: {np.round(fid[channel_index], 3)}\n'
            f'concept {min_index} - distance: {min_distance:.2f}'
        )
        # We want to put special emphasis on this first figure by making the border thicker.
        for axis in ['top','bottom','left','right']:
            ax_ch.spines[axis].set_linewidth(2)
        
        draw_image(
            ax=ax_ch,
            image_path=image_path,
        )
        plot_node_importances_background(
            ax=ax_ch,
            g=graph,
            node_positions=node_positions,
            node_importances=graph['node_importances'][:, channel_index],
            radius=50,
        )
        plot_edge_importances_background(
            ax=ax_ch,
            g=graph,
            node_positions=node_positions,
            edge_importances=graph['edge_importances'][:, channel_index]
        )
        
        # :hook select_examples:
        #       This hook is responsible for selecting the examples that are to be shown for the given concept.
        #       It receives the concept and the number of examples that are to be shown and is supposed to return
        #       a list of the examples that are to be shown in the format of visual graph element dicts.
        examples = e.apply_hook(
            'select_examples',
            concept=concept,
        )
        
        # drawing the examples from that concept cluster
        for i in range(e.NUM_EXAMPLES):
            ax = rows[channel_index][i + 1]
            
            example_element = examples[i]
            example_graph = example_element['metadata']['graph']
            
            ax.set_title(f'Example {i} - Index: {example_element["metadata"]["index"]}')
            draw_image(
                ax=ax,
                image_path=example_element['image_path'],
            )
            plot_node_importances_background(
                ax=ax,
                g=example_graph,
                node_positions=example_graph['node_positions'],
                node_importances=example_graph['node_importances'][:, channel_index],
                radius=50,
            )
            plot_edge_importances_background(
                ax=ax,
                g=example_graph,
                node_positions=example_graph['node_positions'],
                edge_importances=example_graph['edge_importances'][:, channel_index]
            )
        
    concept_path = os.path.join(e.path, 'concept_explanations.pdf')
    fig.savefig(concept_path)
        
                
experiment.run_if_main()