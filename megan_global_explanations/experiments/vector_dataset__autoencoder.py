import os
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from megan_global_explanations.models import DensePredictor
from megan_global_explanations.models import DenseAutoencoder
from megan_global_explanations.utils import RecordIntermediateEmbeddingsCallback

# == EVALUATION PARAMETERS ==

COLOR_PRIMARY: str = 'gray'
FIG_SIZE_BASE: int = 10


# == MODEL PARAEMTERS ==
EPOCHS: int = 200
BATCH_SIZE: int = 64

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'vector_dataset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model')
def train_model(e: Experiment,
                indices: t.List[int],
                index_data_map: dict
                ):
    
    model = DenseAutoencoder(
        encoder_units=[32, 16, 2],
        decoder_units=[2, 16, 32, 5],
        final_activation='linear'
    )
    
    model.compile(
        loss=ks.losses.MeanSquaredError(),
        optimizer=ks.optimizers.Adam(learning_rate=0.001)
    )
    
    x = [index_data_map[i]['vector'] for i in indices]
    
    e.log('starting model training...')
    recorder = RecordIntermediateEmbeddingsCallback(
        elements=x,
        epoch_step=1,
        active=True,
        logger=e.logger,
        embedding_func=lambda mdl, els: mdl.embedd_vectors(els),
    )
    model.fit(
        x, x,
        batch_size=e.BATCH_SIZE,
        epochs=e.EPOCHS,
        callbacks=[
            recorder
        ]
    )
    
    # ~ Saving the embedding recordings
    # After the training is done, the "recorder" callback will contain an internal dictionary
    # "epoch_embeddings_map" whose keys are the epoch indices and the values are the embedding 
    # for all of the elements. We need to transfer this data into the experiment storage so that
    # it can be processed later on.
    for epoch, embeddings in recorder.epoch_embeddings_map.items():
        e[f'intermediate_embeddings/{epoch}'] = embeddings

    return model


@experiment.hook('evaluate_model')
def evaluate_model(e: Experiment,
                   model: t.Any,
                   indices: t.List[int],
                   index_data_map: dict
                   ):

    e.log('evaluating model...')


experiment.run_if_main()