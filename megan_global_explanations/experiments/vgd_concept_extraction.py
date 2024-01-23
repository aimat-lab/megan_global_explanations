"""
This experiment can be used to generate a concept extraction report for a visual graph dataset 
and an already trained MEGAN model. 


"""
import os
import pathlib
import typing as t

import hdbscan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.torch.megan import Megan

from megan_global_explanations.visualization import create_concept_cluster_report


PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# The parameters determine the details related to the dataset that should be used as the basis 
# of the concept extraction

# :param VISUAL_GRAPH_DATASETS:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either 
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this 
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote 
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or 
#       "classification"
DATASET_TYPE: str = 'regression'

SUBSET: t.Optional[int] = None

# == MODEL PARAMETERS == 
# These parameters determine the details related to the model that should be used for the 
# concept extraction. For this experiment, the model should already be trained and only 
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the 
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'rb_dual_motifs.ckpt')

# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a 
#       fidelity higher than this will be used as possible candidates for the clustering.
FIDELITY_THRESHOLD: float = 0.5
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially 
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
MIN_CLUSTER_SIZE: int = 25
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the 
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while 
#       lower values tend to result in more clusters.
MIN_SAMPLES: int = 50

# == PROTOTYPE OPTIMIZATION PARAMETERS ==



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
    
    reader = VisualGraphDatasetReader(
        path=path,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    
    example_graph = list(index_data_map.values())[0]['metadata']['graph']
    e['node_dim'] = example_graph['node_attributes'].shape[1]
    e['edge_dim'] = example_graph['edge_attributes'].shape[1]
    e['out_dim'] = example_graph['graph_labels'].shape[0]
    e.log(f'loaded dataset with {e["node_dim"]} node features and {e["edge_dim"]} edge features')
    
    return index_data_map


@experiment.hook('load_model')
def load_model(e: Experiment,
               path: str
               ) -> Megan:
    
    model = Megan.load_from_checkpoint(path)
    return model


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
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
    index_data_map: t.Dict[int, dict] = e.apply_hook(
        'load_dataset',
        path=dataset_path,
    )
    num_graphs = len(index_data_map)
    e.log(f'loaded dataset with {num_graphs} elements')
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    indices = np.array([index for index in index_data_map.keys()])
    
    # ~ loading the model
    # Besides the dataset we also have to load the model from its persistent representation on the disk
    # so that we can use it for the concept clustering
    
    model: Megan = e.apply_hook(
        'load_model',
        path=e.MODEL_PATH,
    )
    num_channels = model.num_channels
    e['num_channels'] = num_channels
    e.log(f'loaded model of the class: {model.__class__.__name__} '
          f'with {num_channels} explanation channels')
    
    # ~ Concept clustering the latent space
    
    e.log('running the model forward pass for all the graphs...')
    # First of all we need to query the model using all the graphs from the dataset to obtain the model's
    # predictions as well as the explanations and graph embeddings that are required for the clustering
    infos = model.forward_graphs(graphs) 
    # We also want to calculate the loo devations aka the channel-specific fidelity values as those will 
    # be part of the criterium that we will use to filter the relevant concept clusters.
    deviations = model.leave_one_out_deviations(graphs)
    
    e.log('updating the dataset...')
    # To make it easier going forward we will actually attach all the information gained from this 
    # model forward pass to the dataset structure itself (to the graph dicts)
    for graph, info, dev in zip(graphs, infos, deviations):
        
        # graph_output: (O, )
        graph['graph_output'] = info['graph_output']
        # graph prediction is supposed to be a single value that determines the overall prediction of the 
        # model. This differs for regression and classification datasets where the regression result is the 
        # value itself and the classification result is the predicted class index
        # graph_prediction: (, )
        if e.DATASET_TYPE == 'regression':
            graph['graph_prediction'] = info['graph_output'][0]
        elif e.DATASET_TYPE == 'classification':
            graph['graph_prediction'] = np.argmax(info['graph_output'])
        
        # graph_embeddings: (D, K)
        graph['graph_embedding'] = info['graph_embedding']
        # node_importance: (V, K)
        graph['node_importances'] = info['node_importance']
        # edge_importance: (E, K)
        graph['edge_importances'] = info['edge_importance']
    
        # graph_fidelity: (O, K)
        graph['graph_deviation'] = dev

        # The graph fidelity should be an vector with as many elements as there are explanation channels 
        # so basically there should be one fidelity value per explanation channel. The derivation of that 
        # form is differently defined for regression and classification tasks.
        # graph_fidelity: (K, )
        if e.DATASET_TYPE == 'regression':
            graph['graph_fidelity'] = np.array([-dev[0, 0], dev[0, 1]])
        elif e.DATASET_TYPE == 'classification':
            graph['graph_fidelity'] = np.diag(dev)
    
    # Now we calculate the concept clusters separately for each of the explanation channels of the model.
    e.log('starting concept clustering...')
    cluster_infos: t.List[dict] = []
    cluster_index = 0
    for channel_index in range(num_channels):
        
        e.log(f'> CHANNEL {channel_index}')
        # Here we filter according to the fidelity - we only want elements with a certain minumum fidelity to be 
        # eligible for the clustering to begin with. The reasoning here is that there are a lot of elements which 
        # do not have any activation in one of the channels and therefore also have ~0 fidelity for that channel.
        # Those elements are not going to be informative at all.
        channel_indices, channel_graphs = zip(*[
            (index, graph) 
            for index, graph in zip(indices, graphs) 
            if graph['graph_fidelity'][channel_index] > e.FIDELITY_THRESHOLD
        ])
        channel_indices = np.array(channel_indices)
        
        graph_embeddings = np.array([graph['graph_embedding'][:, channel_index] for graph in channel_graphs])
        e.log(f' * filtered {len(channel_indices)} elements from {len(indices)}')
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=e.MIN_CLUSTER_SIZE,
            min_samples=e.MIN_SAMPLES,
        )
        labels = clusterer.fit_predict(graph_embeddings)
        clusters = [c for c in set(labels) if c >= 0]
        num_clusters = len(clusters)
        e.log(f' * found {num_clusters} clusters')
   
        for cluster in clusters:
            
            mask = (labels == cluster)
            cluster_graph_embeddings = graph_embeddings[mask]
            cluster_indices = channel_indices[mask]
            cluster_index_tuples = [(i, channel_index) for i in cluster_indices]
            cluster_graphs = [index_data_map[i]['metadata']['graph'] for i in cluster_indices]
            cluster_image_paths = [index_data_map[i]['image_path'] for i in cluster_indices]
            
            info = {
                'index':                cluster_index,
                'embeddings':           cluster_graph_embeddings,
                'index_tuples':         cluster_index_tuples,
                'graphs':               cluster_graphs,
                'image_paths':          cluster_image_paths,
            }
            cluster_index += 1
            cluster_infos.append(info)
            
    # We definitely want to store the cluster infos to the experiment storage so that we can access them 
    # later on during the analysis as well.
    e['cluster_infos'] = cluster_infos
    
    e.log('creating the concept report...')
    report_path = os.path.join(e.path, 'concept_report.pdf')
    cache_path = os.path.join(e.path, 'cache')
    os.mkdir(cache_path)
    create_concept_cluster_report(
        cluster_data_list=cluster_infos,
        dataset_type=e.DATASET_TYPE,
        logger=e.logger,
        path=report_path,
        cache_path=cache_path,
        examples_type='centroid',
        num_examples=16,
    )


experiment.run_if_main()