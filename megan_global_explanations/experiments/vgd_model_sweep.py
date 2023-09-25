"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import VisualGraphDatasetReader

PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True
SEED = 1

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
DATASET_TYPE: int = 'regression'  # classification
NUM_TARGETS: int = 1
NUM_TEST: int = 500

# == SWEEP PARAMETERS ==
SWEEP_KEYS: t.List[str] = []
REPETITIONS: int = 1
DEVICE: str = 'cpu:0'


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log(f'starting experiment with tf device "{DEVICE}"...')
    e['device'] = tf.device(DEVICE)
    e['device'].__enter__()
    
    random.seed(SEED)
    np.random.seed(SEED)

    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset with {len(index_data_map)} elements')
    
    e['sweep_keys'] = []
    e['repetitions'] = []
    
    for sweep_index, key in enumerate(e.SWEEP_KEYS):
        e.log(f'(#) sweep ({sweep_index+1}/{len(SWEEP_KEYS)}): {key}')
        e['sweep_keys'].append(key)
        
        for rep_index in range(e.REPETITIONS):
            e.log(f' > repetition ({rep_index+1}/{e.REPETITIONS})')
            e['repetitions'].append(rep_index)
            
            # ~ DATASET SPLIT
            # the dataset split
            e['indices'] = indices
            test_indices = random.sample(indices, k=NUM_TEST)
            train_indices = list(set(indices).difference(set(test_indices)))
            
            e.log(f'training the model...')
            model = e.apply_hook(
                f'train_model_{key}', 
                index_data_map=index_data_map,
                indices=train_indices,
                rep_index=rep_index,
            )
            
            e.log(f'evluating the model...')
            e.apply_hook(
                f'evaluate_model',
                model=model,
                index_data_map=index_data_map,
                indices=test_indices,
                rep_index=rep_index,
                sweep_key=key,
            )
            e.apply_hook(
                f'evaluate_model_{key}',
                model=model,
                index_data_map=index_data_map,
                indices=test_indices,
                rep_index=rep_index,
            )


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()