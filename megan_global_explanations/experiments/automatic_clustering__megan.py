"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

import numpy as np

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan2
from graph_attention_student.models import load_model
from graph_attention_student.util import array_normalize

PATH = pathlib.Path(__file__).parent.absolute()

# == MODEL PARAMETERS ==
MODEL_PATH: str = os.path.join(PATH, 'assets', 'predictors', 'rb_dual_motifs')
FIDELITY_THRESHOLD: float = 0.3

experiment = Experiment.extend(
    'automatic_clustering.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('get_encoded')
def hook(e, index_data_map):
    e.log('loading MEGAN model...')
    model: Megan = load_model(e.MODEL_PATH)
    # We would also like to save the model into the experiment registry for the case that we will need it 
    # again later on.
    e['model'] = model
    
    # We would like to 
    indices, graphs = zip(*[(index, data['metadata']['graph']) for index, data in index_data_map.items()])
    
    e.log('making predictions for all graphs...')
    predictions = model.predict_graphs(graphs)
    e.log('computing leave-one-out deviations...')
    leave_one_out = model.leave_one_out_deviations(graphs)
    
    e.log('attaching results to graphs...')
    for graph, (out, ni, ei), one_out in zip(graphs, predictions, leave_one_out):
        graph['graph_prediction'] = out
        graph['graph_one_out'] = one_out
        if e.DATASET_TYPE == 'regression':
            graph['graph_fidelity'] = np.array([-one_out[0][0], +one_out[1][0]])
        elif e.DATASET_TYPE == 'classification':
            graph['graph_fidelity'] = np.array([one_out[k][k] for k in range(e.NUM_CHANNELS)])
        
        graph['node_importances'] = array_normalize(ni)
        graph['edge_importances'] = array_normalize(ei)
    
    # encoded: (A, K, D)
    encoded = model.embedd_graphs(graphs)
    
    # encoded: (K, A, D)
    encoded = np.concatenate([np.expand_dims(encoded[:, k, :], axis=0) for k in range(e.NUM_CHANNELS)], axis=0) 

    e.log('filtering by fidelity threshold...')
    
    indices_filtered = []
    encoded_filtered = []
    for k, enc in enumerate(encoded):
        indices_filtered.append([])
        encoded_filtered.append([])
        
        for i, arr in enumerate(enc):
            if graphs[i]['graph_fidelity'][k] > e.FIDELITY_THRESHOLD:
                indices_filtered[k].append(indices[i])
                encoded_filtered[k].append(arr)
    
    encoded = encoded_filtered
    indices = indices_filtered
    
    return encoded, indices

experiment.run_if_main()