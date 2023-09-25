import os
import typing as t
import warnings

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
from megan_global_explanations.deep_ect import DeepEctTrainer
from megan_global_explanations.visualization import animate_deepect_history

# Disable all TensorFlow warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# == EVALUATION PARAMETERS ==

COLOR_PRIMARY: str = 'gray'
FIG_SIZE_BASE: int = 10


# == MODEL PARAEMTERS ==
EPOCHS: int = 500
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
        loss=ks.losses.MeanAbsoluteError(),
        optimizer=ks.optimizers.Adam(learning_rate=0.001)
    )
    
    x = [index_data_map[i]['vector'] for i in indices]
    
    e.log('starting model training...')
    
    trainer = DeepEctTrainer(
        model=model,
        elements=x,
        logger=e.logger,
        save_history=True,
        split_epoch_step=100,
    )
    trainer.initialize()
    
    # recorder = RecordIntermediateEmbeddingsCallback(
    #     elements=x,
    #     epoch_step=1,
    #     active=True,
    #     logger=e.logger,
    #     embedding_func=lambda mdl, els: mdl.embedd_vectors(els),
    # )
    trainer.fit(
        x, x,
        batch_size=e.BATCH_SIZE,
        epochs=e.EPOCHS,
    )
    
    # model.fit(
    #     x, x,
    #     batch_size=e.BATCH_SIZE,
    #     epochs=e.EPOCHS,
    #     callbacks=[
    #         recorder
    #     ]
    # )
    
    # ~ Saving the embedding recordings
    # After the training is done, the "recorder" callback will contain an internal dictionary
    # "epoch_embeddings_map" whose keys are the epoch indices and the values are the embedding 
    # for all of the elements. We need to transfer this data into the experiment storage so that
    # it can be processed later on.
    # for epoch, embeddings in recorder.epoch_embeddings_map.items():
    #     e[f'intermediate_embeddings/{epoch}'] = embeddings
    
    anim_path = os.path.join(e.path, 'history.mp4')
    animate_deepect_history(
        history=trainer.history,
        output_path=anim_path,
    )

    return model


@experiment.hook('evaluate_model')
def evaluate_model(e: Experiment,
                   model: t.Any,
                   indices: t.List[int],
                   index_data_map: dict
                   ):

    e.log('evaluating model...')


experiment.run_if_main()