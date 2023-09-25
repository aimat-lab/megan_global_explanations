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
from graph_attention_student.data import tensors_from_graphs

from megan_global_explanations.models import GraphPredictor
from megan_global_explanations.utils import RecordIntermediateEmbeddingsCallback

# == EVALUATION PARAMETERS ==

COLOR_PRIMARY: str = 'gray'
FIG_SIZE_BASE: int = 10


# == MODEL PARAEMTERS ==
EPOCHS: int = 50
BATCH_SIZE: int = 16

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'graph_dataset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model', replace=False, default=True)
def train_model(e: Experiment,
                indices: t.List[int],
                index_data_map: dict
                ):
    
    model = GraphPredictor(
        units=[16, 16, 16],
        embedding_units=[16, 2],
        final_units=[16, 5],
        final_activation='softmax'
    )
    
    model.compile(
        loss=ks.losses.CategoricalCrossentropy(),
        optimizer=ks.optimizers.Adam(learning_rate=0.0001)
    )
    
    graphs = [index_data_map[i]['metadata']['graph'] for i in indices]
    y = np.array([index_data_map[i]['metadata']['targets'] for i in indices])
    x = tensors_from_graphs(graphs)
    
    e.log('starting model training...')
    recorder = RecordIntermediateEmbeddingsCallback(
        elements=graphs,
        epoch_step=1,
        active=True,
        logger=e.logger,
        embedding_func=lambda mdl, els: mdl.embedd_graphs(els),
    )
    model.fit(
        x, y,
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
    
    e.log('making test set predictions...')
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in indices]
    
    out_true = [index_data_map[i]['metadata']['targets'] for i in indices]
    out_pred = model.predict_graphs(graphs_test)
    
    labels_true = np.argmax(out_true, axis=-1)
    labels_pred = np.argmax(out_pred, axis=-1)
    
    acc_value = accuracy_score(labels_true, labels_pred)
    e['prediction/acc'] = acc_value

    e.log(f'acc: {acc_value:0.2f}')


experiment.run_if_main()