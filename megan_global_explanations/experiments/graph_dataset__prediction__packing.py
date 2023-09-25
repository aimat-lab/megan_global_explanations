import os
import typing as t
import warnings
import random

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from umap import UMAP
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from graph_attention_student.data import tensors_from_graphs
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from hdbscan import HDBSCAN

from megan_global_explanations.models import GraphPredictor
from megan_global_explanations.models import load_model
from megan_global_explanations.utils import RecordIntermediateEmbeddingsCallback
from megan_global_explanations.deep_ect import DeepEctTrainer
from megan_global_explanations.pack import ClusterPackingTrainer
from megan_global_explanations.visualization import animate_deepect_history
from megan_global_explanations.visualization import generate_contrastive_colors

# Disable all TensorFlow warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# == DATASET PARAMETERS ==
NUM_TEST: int = 500

# == EVALUATION PARAMETERS ==
COLOR_PRIMARY: str = 'gray'
FIG_SIZE_BASE: int = 10

# == MODEL PARAEMTERS ==
EPOCHS: int = 15
BATCH_SIZE: int = 32

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'graph_dataset__prediction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model', replace=True, default=False)
def train_model(e: Experiment,
                indices: t.List[int],
                index_data_map: dict
                ):
    
    e['dataset_path'] = e.VISUAL_GRAPH_DATASET_PATH
    
    e.log('setting up the model...')
    model = GraphPredictor(
        units=[16, 16, 16],
        embedding_units=[16, 16, 8],
        final_units=[5],
        final_activation='softmax'
    )
    
    graphs = [index_data_map[i]['metadata']['graph'] for i in indices]
    y = np.array([index_data_map[i]['metadata']['targets'] for i in indices])
    x = tensors_from_graphs(graphs)
        
    e.log('starting model training...')
    
    trainer = ClusterPackingTrainer(
        model=model,
        logger=e.logger,
        min_samples=200,
        cluster_batch_size=2500,
        factor=0.0,
    )
    
    recorder = RecordIntermediateEmbeddingsCallback(
        elements=graphs,
        epoch_step=1,
        active=True,
        logger=e.logger,
        embedding_func=lambda mdl, els: mdl.embedd_graphs(els),
    )
    
    e.log('starting training...')
    trainer.compile(
        loss=ks.losses.CategoricalCrossentropy(),
        optimizer=ks.optimizers.Adam(learning_rate=0.0001),
        #run_eagerly=True,
    )
    trainer.fit(
        x, y,
        #batch_size=e.BATCH_SIZE,
        batch_size=32,
        #epochs=e.EPOCHS,
        epochs=100,
        callbacks=[recorder]
    )
    
    # ~ Saving the embedding recordings
    # After the training is done, the "recorder" callback will contain an internal dictionary
    # "epoch_embeddings_map" whose keys are the epoch indices and the values are the embedding 
    # for all of the elements. We need to transfer this data into the experiment storage so that
    # it can be processed later on.
    for epoch, embeddings in recorder.epoch_embeddings_map.items():
        e[f'intermediate_embeddings/{epoch}'] = embeddings
        
    # ~ Saving the embedding recordings
    # 
    
    # ~ Saving the model and the trainer
    # Here we need to save the model and the trainer object persistently to the disk so that we can later on 
    # load them again to perform analyses
    
    # The model is easily saveable as a folder as any other keras model
    e.log('saving the model to disk...')
    model_path = os.path.join(e.path, 'model')
    model.save(model_path)
    
    # All the state information contained in the trainer object can also be easily saved as a json file
    e.log('saving the deepect trainer to disk...')
    trainer_path = os.path.join(e.path, 'trainer.json')
    # trainer.save(trainer_path)
    
    return model


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')
    
    # ~ loading all the relevant info from the disk
    # First of all we need to load the model, the trainer object and the dataset back into memory
    
    e.log('loading the model...')
    model_path = os.path.join(e.path, 'model')
    model = load_model(model_path)
    
    e.log('loading the deepect trainer...')
    trainer_path = os.path.join(e.path, 'trainer.json')
    # trainer = DeepEctTrainer.load(trainer_path)
    # trainer.set_model(model)
    
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=e['dataset_path'],
        logger=e.logger,
    )
    index_data_map = reader.read()
    indices = list(index_data_map.keys())
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    
    # ~ Visualizing the latent space
    # We want to use the cluster tree information embedded in the trainer to assign a cluster label to all the 
    # elements of the full dataset and then create a visualization of the latent space with the cluster labels.
    
    e.log('creating embeddings for the entire dataset...')
    embeddings = model.embedd_graphs(graphs)
    e.log(f'embeddings shape: {embeddings.shape}')
    
    e.log('predicting labels for the entire dataset...')
    # labels = trainer.predict_elements(graphs)
    clusterer = HDBSCAN(min_samples=200)
    clusterer.fit(embeddings)
    labels = clusterer.labels_

    # If the embeddings are not 2D already we need to use a dimensionality reduction to map it into 2 dimensions
    # first
    mapped = embeddings
    if embeddings.shape[-1] != 2:
        mapper = UMAP(
            n_components=2,
            n_neighbors=200,
            min_dist=0.0,
            metric='euclidean'
        )
        e.log('fitting umap transformation...')
        mapped = mapper.fit_transform(embeddings)
        
    else:
        class IdentityMapper:
            
            def transform(x):
                return x
            
        mapper = IdentityMapper()
        
    clusters = list(set(labels))
    num_clusters = len(clusters)
    cluster_color_map = dict(zip(clusters, generate_contrastive_colors(num_clusters)))
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    for label in clusters:
        indices_cluster = [i for i, l in enumerate(labels) if l == label]
        mapped_cluster = mapped[indices_cluster]
        ax.scatter(
            mapped_cluster[:, 0], mapped_cluster[:, 1],
            c=cluster_color_map[label],
            label=f'cluster {label}'
        )
        
    ax.set_title('Latent Space of all Elements')
    ax.legend()
    fig_path = os.path.join(e.path, 'latent_space.pdf')
    fig.savefig(fig_path)
    
    # ~ Showing example graphs
    # We also want to look at some example graphs from each of those clusters
    
    e.log('drawing some examples...')
    pdf_path = os.path.join(e.path, 'examples.pdf')
    with PdfPages(pdf_path) as pdf:
        
        for label in clusters:
            member_indices = [i for i, l in zip(indices, labels) if l == label]

            num_examples = min(len(member_indices), 20)
            example_indices = random.sample(member_indices, k=num_examples)
            fig, rows = plt.subplots(
                ncols=num_examples,
                nrows=1,
                figsize=(10 * num_examples, 10),
                squeeze=False,
            )
            
            for c, index in enumerate(example_indices):
                
                ax = rows[0][c]
                ax.set_title(f'cluster: {label} - element: {index}')
                data = index_data_map[index]
                draw_image(ax, data['image_path'])
                
            pdf.savefig(fig)
            plt.close(fig)


experiment.run_if_main()