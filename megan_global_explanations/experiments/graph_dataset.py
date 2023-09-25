import os
import sys
import json
import pathlib
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.data import VisualGraphDatasetReader


PATH = pathlib.Path(__file__).parent.absolute()


# == DATASET PARAMETERS ==

VISUAL_GRAPH_DATASET_PATH = os.path.join(PATH, 'assets', '5_graph_motifs_large')
NUM_DIMENSIONS: int = 2
NUM_TEST: int = 25

# == TRAINING PARAMETERS ==

DEVICE: str = 'cpu:0'

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment): 
    e.log('starting experiment...')
    
    e['device'] = tf.device(e.DEVICE)
    e['device'].__enter__()
    
    e.log('loading dataset...')
    reader = VisualGraphDatasetReader(
        path=e.VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=50,
    )
    index_data_map = reader.read()
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset of {len(index_data_map)} elements')

    e.log(f'creating the train test split...')
    test_indices = random.sample(indices, k=e.NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    e.log(f'chose {len(train_indices)} train elements and {len(test_indices)} test elements')

    # :hook train_model:
    #       This hook should be implemented to train the model using the training dataset.
    model = e.apply_hook(
        'train_model',
        indices=train_indices,
        index_data_map=index_data_map
    )
    
    # :hook evaluate_model:
    #       This hook should implement all the performance evaluations of the model on testing 
    #       dataset.
    e.apply_hook(
        'evaluate_model',
        model=model,
        indices=test_indices,
        index_data_map=index_data_map,
    )

    
@experiment.analysis
def analysis(e: Experiment):
    
    e.log('starting analysis of the results...')
    
    @e.hook('plot_intermediate_embeddings')
    def plot_intermediate_embeddings(e: Experiment,
                                     epoch_embeddings_map: dict, 
                                     color: str = 'gray',
                                     ):
        # embeddings_all: (E, N, 2)
        embeddings_all = np.array([emb.tolist() for emb in epoch_embeddings_map.values()])
        
        num_frames = len(epoch_embeddings_map)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        
        x_min = np.min(embeddings_all[:, :, 0])
        x_max = np.max(embeddings_all[:, :, 0])
        x_diff = abs(x_max - x_min)
        
        y_min = np.min(embeddings_all[:, :, 1])
        y_max = np.max(embeddings_all[:, :, 1])
        y_diff = abs(y_max - y_min)
            
        def update(frame):
            
            epoch, embeddings = list(epoch_embeddings_map.items())[frame]
            ax.clear()
            ax.set_title(f'Epoch {epoch} - Frame {frame}')
            ax.set_xlim([x_min - 0.1 * x_diff, x_max + 0.1 * x_diff])
            ax.set_ylim([y_min - 0.1 * y_diff, y_max + 0.1 * y_diff])
            
            ax.scatter(
                embeddings[:, 0], embeddings[:, 1],
                color=color,
            )
        
        anim = FuncAnimation(
            fig=fig, 
            func=update, 
            frames=num_frames, 
            interval=100, 
            blit=False
        )
        anim_path = os.path.join(e.path, 'intermediate_embeddings.mp4')
        anim.save(anim_path, writer='ffmpeg', fps=4)
        
        return anim
    
    if 'intermediate_embeddings' in e.data:
        
        epoch_embeddings_map = {int(epoch): np.array(emb) for epoch, emb in e['intermediate_embeddings'].items()}
        
        if e.NUM_DIMENSIONS == 2:
            
            e.log('animating the progression of epochs...')
            _ = e.apply_hook(
                'plot_intermediate_embeddings',
                epoch_embeddings_map=epoch_embeddings_map
            )
            
            e.log('animation created')
    

experiment.run_if_main()