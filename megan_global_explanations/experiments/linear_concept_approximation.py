import os
import pathlib
import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.torch.megan import Megan
from graph_attention_student.visualization import plot_regression_fit

from megan_global_explanations.data import ConceptReader

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')


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

# == TRAINING PARAMETERS ==
# These parameters determine the details for the training of the simple interpretable proxy model.

# :param NUM_TEST:
#       The number of test examples to be used for the training of the proxy model. This is the number of examples
#       that will be used to evaluate the performance of the proxy model.
NUM_TEST: int = 1000


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


@experiment.hook('create_encoding')
def create_encoding(e: Experiment,
                    graph: dict,
                    concepts: t.List[dict]
                    ) -> np.ndarray:
    """
    This hook receives a graph and a list of concepts and is supposed to return the one-hot encoding 
    of the graph based on the concepts.
    """
    # ~ finding the closest concepts
    concept_indices: t.List[int] = []
    for channel_index in range(e['num_channels']):
        embedding = graph['graph_embedding'][:, channel_index]

        min_distance = float('inf')
        min_index = None
        for concept_index, concept in enumerate(concepts):#enumerate([concept for concept in concepts if concept['channel_index'] == channel_index]):
            centroid = concept['centroid']
            
            distance = cosine(centroid, embedding)
            if distance < min_distance:
                min_distance = distance
                min_index = concept['index']

        concept_indices.append(min_index)
        
    # ~ creating the one-hot encoding
    encoding = np.zeros((e['num_concepts'], ))
    for concept_index in concept_indices:
        encoding[concept_index] = 1
    
    return encoding

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
    e['num_channels'] = num_channels
    
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
    concepts = concept_reader.read()
    e.log(f'loaded {len(concepts)} concepts')
    num_concepts = len(concepts)
    e['num_concepts'] = num_concepts
    
    # ~ creating concept encoding
    # essentially, the idea here is the following: We want to encode every element of the dataset as a constant 
    # vector which is only based on the concept clustering.
    
    # For this we first need to query the model with the entire dataset to obtain the graph embeddings for the 
    # dataset.
    e.log('model forward pass...')
    infos = model.forward_graphs(graphs)

    e.log('model leave-one-out pass...')
    # devs: (num_graphs, num_outputs, num_channels)
    devs = model.leave_one_out_deviations(graphs)

    e.log('creating the encodings...')
    for c, (index, graph, info, dev) in enumerate(zip(indices, graphs, infos, devs)):
        
        graph['node_importances'] = info['node_importance']
        graph['edge_importances'] = info['edge_importance']
        graph['graph_embedding'] = info['graph_embedding']
        graph['graph_deviation'] = np.array([dev[0, 0], dev[0, 1]])
        
        # encoding: (num_concepts, )
        encoding: np.ndarray = e.apply_hook(
            'create_encoding',
            graph=graph,
            concepts=concepts,
        )
        graph['graph_encoding'] = encoding
        
        if c % 1000 == 0:
            e.log(f' * processed {c}/{num_graphs} graphs')
        
        
    # ~ training the model
    # In this section we now want to train a very simple and interpretable model based on these encodings that we 
    # have just created from the concept information.
    
    e.log('creating train-test split...')
    indices = list(index_data_map.keys())
    test_indices = random.sample(indices, k=e.NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    
    test_encodings = np.array([index_data_map[index]['metadata']['graph']['graph_encoding'] for index in test_indices])
    test_labels = np.array([index_data_map[index]['metadata']['graph']['graph_labels'] for index in test_indices])
    
    train_encodings = np.array([index_data_map[index]['metadata']['graph']['graph_encoding'] for index in train_indices])
    train_labels = np.array([index_data_map[index]['metadata']['graph']['graph_labels'] for index in train_indices])
    
    e.log('training the linear model...')
    model = LinearRegression()
    model.fit(train_encodings, train_labels)
    
    out_pred = model.predict(test_encodings)
    out_true = test_labels
    
    e.log(f'evaluating the linear model...')
    r2_value = r2_score(out_true, out_pred)
    mae_value = mean_absolute_error(out_true, out_pred)
    e['test/r2'] = r2_value
    e['test/mae'] = mae_value
    e.log(f' * r2 value: {r2_value:.3f}')
    e.log(f' * mae value: {mae_value:.3f}')
    
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(5, 5),
    )
    plot_regression_fit(
        ax=ax,
        values_pred=out_pred,
        values_true=out_true,
    )
    ax.set_title(f'Linear Regression Test Results\n'
                 f'R2: {r2_value:.3f} - MAE: {mae_value:.3f}')
    fig_path = os.path.join(e.path, 'regression_fit.pdf')
    fig.savefig(fig_path)
    
    # ~ Correlation between the linear model and the concept contributions
    
    coefficients = model.coef_[0]
    contributions = []
    for coef, concept in zip(coefficients, concepts):
        concept_index = concept['index']
        channel_index = concept['channel_index']
        
        concept_graphs = [index_data_map[element['metadata']['index']]['metadata']['graph'] for element in concept['elements']]
        concept_contributions = [graph['graph_deviation'][channel_index] for graph in concept_graphs]
        concept_contribution = np.mean(concept_contributions)
        contributions.append(concept_contribution)
        
    e.log('evaluating the relationship between the linear model and the concept contributions...')
    res = spearmanr(coefficients, contributions)
    coef_corr_value = res.statistic
    coef_r2_value = r2_score(coefficients, contributions) 
    e.log(f' * spearman correlation: {coef_corr_value:.3f}')
    e.log(f' * r2 value: {coef_r2_value:.3f}')
    
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(5, 5),
    )
    ax.scatter(contributions, coefficients)
    ax.set_xlabel('average contribution of concept')
    ax.set_ylabel('linear regression coefficient')
    ax.set_title(f'Contributions vs. Coefficients\n'
                 f'N.o. Concepts: {num_concepts} - R2: {coef_r2_value:.2f} - Spearman: {coef_corr_value:.2f}')
    fig_path = os.path.join(e.path, 'contribution_coefficient.pdf')
    fig.savefig(fig_path)