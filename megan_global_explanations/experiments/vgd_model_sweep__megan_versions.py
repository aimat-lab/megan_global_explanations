"""
This experiment is used to perform a sweep comparsion specifically over different versions of the 
MEGAN model family. In a certain number of independent repetitions each model will be trained and 
then evaluated, mainly towards assessing the latent space clustering capabilities.

The following models are implemented for comparison in this experiment:

- *MEGAN*: The initial version of the MEGAN model as conceptionalized in the paper.
- *MEGAN2*: An architectural and procedural extension of the initial megan model. This model includes 
  additional projection networks that are inserted right after the message passing part and which produce 
  the latent explanation embeddings. Furthermore the training process is modified with an additional 
  fidelity training step which actively trains the model to produce explanations that are faithful to the
  intended interpretations.
- *PackMEGAN*: An extension to the Megan2 model which includes an additonal "packing" training loss. 
  this loss is activated only after a certain warm up training time. At that point a clusterng on 
  the embedding space is performed and the density within the found clusters is increased by this loss.

Since all of the models are from the same basic MEGAN family they share all a the hyperparameters that 
are not specific to that particular variation.
"""
import os
import pathlib
import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import tensorflow.keras as ks
from umap import UMAP
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.neighbors import LocalOutlierFactor
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan
from graph_attention_student.models.megan import Megan2
from graph_attention_student.models.utils import ChangeVariableCallback
from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.data import mock_importances_from_graphs
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.fidelity import leave_one_out_analysis

from megan_global_explanations.utils import RecordIntermediateEmbeddingsCallback
from megan_global_explanations.visualization import plot_distributions
from megan_global_explanations.models import EctMegan
from megan_global_explanations.deep_ect import DeepEctTrainer
from megan_global_explanations.pack import ClusterPackingTrainer
from megan_global_explanations.visualization import animate_deepect_history


PATH = pathlib.Path(__file__).parent.absolute()


CHANNEL_COLORS = [
    'lightskyblue',
    'lightsalmon',
]

# == SWEEP PARAMETERS ==
# These parameters are for the model sweep that will be executed in this experiment. The purpose of 
# this exerperiment is to run a comparison between different models and this is being configured here

# :param SWEEP_KEYS:
#       This list should contain unique string identifiers where each one represents one of the models that 
#       is being compared in the experiment. These string names have to correlate to one hook implementation 
#       in the experiment which provided the implementation just for the training of that model.
SWEEP_KEYS: t.List[str] = ['megan_2', 'megan_1']
# :param REPETITIONS:
#       The number of independent repetitions of the training process for each of the models. This is used 
#       to then average the result over all these repetitions at the end.
REPETITIONS: int = 1

# == MEGAN MODEL PARAMETERS ==
UNITS = [64, 64, 64]
IMPORTANCE_CHANNELS = 2
IMPORTANCE_FACTOR = 1.0
IMPORTANCE_MULTIPLIER = 0.5
REGRESSION_REFERENCE = 0.0
REGRESSION_WEIGHTS = [1.0, 1.0]
FINAL_UNITS = [32, 16, 1]
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 1.0

LOSS_CB = lambda: [ks.losses.MeanSquaredError(), NoLoss(), NoLoss()]
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(
    learning_rate=0.001,
)

# == TRAINING PARAMETERS ==
# These parameters control the training process of the models. All the models will use the same training 
# hyperparameters defined here. This includes for example the number of epochs and the batch size.

# :param EPOCHS:
#       The number of epochs which the training is running for. In each epoch the model will be trained 
#       on the entire training dataset, split into multiple batches.
EPOCHS: int = 50
# :param BATCH_SIZE:
#       The number of elements from the training dataset which constitute a single batch that the model 
#       will be trained on. For each batch, the model performs one parameter update
BATCH_SIZE: int = 32
# :param RUN_EAGERLY:
#       A boolean flag which determines if the tensorflow model will be run in eager mode or not. Eager mode 
#       is essentially meant for debugging. In this mode, instead of building a static graph, all the tensor 
#       operations will be computed as the pure python code. This means that the content of each tensor can
#       actually be evaluated and printed during the runtime. However this mode is *significantly* slower 
#       by at least a factor of 100 and thus not recommended expect for debugging purposes.
RUN_EAGERLY: bool = False
# :param DEVICE:
#       This specifies which device to use for the training of the model. The default is main system CPU. 
#       alternative would be "gpu:0" but for that a GPU with a decent amount of VRAM will be required.
DEVICE = 'cpu:0'


# == MEGAN 2 PARAMETERS ==
# These parameters are specific to the Megan2 model which is an extension of the original model and one 
# of the different versions that is being compared in this experiment. All he parameters not specific 
# to this model will be chosen the same as the base version.

# :param FIDELITY_FACTOR:
#       
FIDELITY_FACTOR = 0.2
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
# :param EMBEDDING_UNITS:
#       This list determines the structure of the embedding projection networks. These projection networks 
#       are inserted directly after the message passing step in the MEGAN model. Each item in this list 
#       defines the number of hidden units in of the layers of these networks.
EMBEDDING_UNITS: t.List[int] = [64, 32, 16]


# == CLUSTER MEGAN PARAMETERS ==
# These parameters are specific to the ClusterMegan model which is one of the models that is being compared 
# here. All the parameters that are not specific to this model will be used the same as regular megan

# :param EPOCHS_WARMUP_RATIO:
#       The ratio of the total number of epochs to be considered as the warmup period for the clustering 
#       and packing process. Only after that number of epochs will the packing loss be activated
EPOCHS_WARMUP_RATIO: float = 0.5
# :param CLUSTER_BATCH_SIZE:
#       This is the number of elements that will be sampled from the entire training dataset to perform 
#       the clustering for the identification of the cluster centroids on.
CLUSTER_BATCH_SIZE: int = 6000
# :param MIN_SAMPLES:
#       This is the min_samples parameter for the hdbscan clustering method that will be used for the clustering. 
#       this parameteter essentially controls the level of sensitivity for the clustering process. If this value 
#       is lower then generally more smaller clusters will be found and if it is bigger then lesser larger clusters 
#       will be found. This value has to be chosen relative to the previously given cluster batch size.
MIN_SAMPLES: int = 100


# == UMAP MAPPER PARAMETERS ==
# All used models have a very high-dimensional latent space and to visualize this space a dimensionality reduction 
# has to be performed first. In this case a UMAP will be used for that. The following parameters determine the 
# the parameters of this umap mapper.
# https://umap-learn.readthedocs.io/en/latest/parameters.html
 
# :param OUTLIER_NEIGHBORS:
#       Before applying the umap, we actually apply a local outlier factor (LOF) outlier detection to hopefully 
#       improve the visualization a bit. This determines the number of local nearest neighbors to be used for 
#       this LOF process.
OUTLIER_NEIGHBORS: int = 100
# :param OUTLIER_FACTOR:
#       This is the relative ratio of all elements to be removed through the outlier detection. Note that his 
#       cannot be higher than 0.5
OUTLIER_FACTOR: float = 0.3

# :param UMAP_METRIC:
#       The string key for the metric to be used to perform the umap. This metric will be used to assess the 
#       pairwise distances of all elements in the high dimensional space.
UMAP_METRIC: str = 'cosine'
# :param UMAP_MIN_DIST:
#       a umap parameter. Best to leave it at zero
UMAP_MIN_DIST: float = 0.0
# :param UMAP_NUM_NEIGHBORS:
#       The number of neighbors to be considered for the umap
UMAP_NUM_NEIGHBORS: int = 200
UMAP_REPULSION_STRENGTH: float = 1.0


# == EVALUATION PARAMETERS ==
# The following parameters are mainly important for different kinds of evaluation purposes such as the 
# plotting of the results.

# :param VALUE_RANGE:
#       This is the value range of the predicted target value.
VALUE_RANGE: tuple = (-4, +4)
# :param NUM_BINS:
#       This is the number of bins to use in all cases where some sort of binning is being applied. This 
#       is mainly the case for the creation of histograms to approximate value distributions.
NUM_BINS: int = 50
# :param RECORD_INTERMEDIATE_EMBEDDINGS:
#       Whether or not to record intermediate embedding snapshots during training. If this is enabled 
#       then in regular intervals during the training, the model will be used to produce the embeddings 
#       for the test set samples and those will be recorded into the experiment storage so that they 
#       can be evaluated later on durin ghe analysis.
RECORD_INTERMEDIATE_EMBEDDINGS: bool = True
# :param INTERMEDIATE_EMBEDDINGS_STEP: 
#       The number of epochs during traing after which a new set of intermediate embeddings is 
#       will be recoreded - if it is enabled.
INTERMEDIATE_EMBEDDINGS_STEP: int = 5
# :param FIG_SIZE_BASE:
#       The base fig size for matplotlib plots
FIG_SIZE_BASE: int = 10
# :param COLOR_PRIMARY:
#       The primary color which will be used for the plotting of the evaluation results
COLOR_PRIMARY: str = 'gray'
# :param COLOR_SECONDARY: 
#       The secondary color which will be used for the plotting of the evaluation results.
#       This color will be used whenever a second color is needed within a plot.
COLOR_SECONDARY: str = 'lightgray'

__DEBUG__ = True
__TESTING__ = True

experiment = Experiment.extend(
    'vgd_model_sweep.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('train_model_megan_1')
def train_model_megan_1(e: Experiment, 
                        index_data_map: dict, 
                        indices: list,
                        rep_index: int,
                        sweep_key: str = 'megan_1',
                        **kwargs):
    
    if e.__TESTING__:
        e.EPOCHS = 1
        e.RUN_EAGERLY = True
    
    e.log(f'training MEGAN v1 model...')
    model = Megan(
        units=e.UNITS,
        final_units=e.FINAL_UNITS,
        importance_channels=e.IMPORTANCE_CHANNELS,
        importance_factor=e.IMPORTANCE_FACTOR,
        importance_multiplier=e.IMPORTANCE_MULTIPLIER,
        final_activation=e.FINAL_ACTIVATION,
        regression_reference=e.REGRESSION_REFERENCE,
        regression_weights=e.REGRESSION_WEIGHTS,
        sparsity_factor=e.SPARSITY_FACTOR,
    )
    model.compile(
        optimizer=e.OPTIMIZER_CB(),
        loss=e.LOSS_CB(),
        run_eagerly=e.RUN_EAGERLY,
    )
    
    graphs = [index_data_map[i]['metadata']['graph'] for i in indices]
    targets = [index_data_map[i]['metadata']['target'] for i in indices]
    
    x = tensors_from_graphs(graphs)
    y = (np.array(targets), *mock_importances_from_graphs(graphs, e.IMPORTANCE_CHANNELS))
    
    # This call back can be used to record snapshots of the graph embedding latent space representations 
    # at different points during the training.
    embeddings_recorder_callback =  RecordIntermediateEmbeddingsCallback(
        epoch_step=e.INTERMEDIATE_EMBEDDINGS_STEP,
        active=e.RECORD_INTERMEDIATE_EMBEDDINGS,
        elements=graphs,
        logger=e.logger,
    )
    
    hist = model.fit(
        x, y,
        epochs=e.EPOCHS,
        batch_size=e.BATCH_SIZE,
        callbacks=[
            embeddings_recorder_callback,
        ]
    )
    
    # After the training is done, the recorder callback has now internally stored the snapshots of the 
    # graph embeddings during the training and now we need to store those into the experiment memory so 
    # that they can be analyzed later
    for epoch, embeddings in embeddings_recorder_callback.epoch_embeddings_map.items():
        e[f'intermediate_embeddings/{rep_index}/{sweep_key}/{epoch}'] = embeddings
    
    return model


@experiment.hook('train_model_megan_2')
def train_model_megan_2(e: Experiment,
                        index_data_map: dict,
                        indices,
                        rep_index: int,
                        sweep_key: str = 'megan_2',
                        **kwargs):
    
    if e.__TESTING__:
        e.EPOCHS = 3
        e.RUN_EAGERLY = True
            
    e.log(f'training MEGAN v2 model...')
    model = Megan2(
        units=e.UNITS,
        final_units=e.FINAL_UNITS,
        importance_channels=e.IMPORTANCE_CHANNELS,
        importance_factor=e.IMPORTANCE_FACTOR,
        importance_multiplier=e.IMPORTANCE_MULTIPLIER,
        final_activation=e.FINAL_ACTIVATION,
        regression_reference=e.REGRESSION_REFERENCE,
        regression_weights=e.REGRESSION_WEIGHTS,
        sparsity_factor=e.SPARSITY_FACTOR,
        # v2 specific
        fidelity_factor=e.FIDELITY_FACTOR,
        fidelity_funcs=e.FIDELITY_FUNCS,
        embedding_units=e.EMBEDDING_UNITS,
    )
    model.compile(
        optimizer=e.OPTIMIZER_CB(),
        loss=e.LOSS_CB(),
        run_eagerly=e.RUN_EAGERLY,
    )
    
    indices_support = random.sample(indices, k=200)
    indices_train = list(set(indices).difference(set(indices_support)))
    e.log(f'split the indices into {len(indices_train)} training indices and {len(indices_support)} support indices')
    
    graphs_train = [index_data_map[i]['metadata']['graph'] for i in indices_train]
    targets_train = [index_data_map[i]['metadata']['target'] for i in indices_train]
    
    x_train = tensors_from_graphs(graphs_train)
    y_train = (np.array(targets_train), *mock_importances_from_graphs(graphs_train, e.IMPORTANCE_CHANNELS))
    
    # This call back can be used to record snapshots of the graph embedding latent space representations 
    # at different points during the training.
    embeddings_recorder_callback =  RecordIntermediateEmbeddingsCallback(
        epoch_step=e.INTERMEDIATE_EMBEDDINGS_STEP,
        active=e.RECORD_INTERMEDIATE_EMBEDDINGS,
        elements=graphs_train,
        logger=e.logger,
    )
    
    hist = model.fit(
        x_train, y_train,
        epochs=e.EPOCHS,
        batch_size=e.BATCH_SIZE,
        callbacks=[
            embeddings_recorder_callback,
        ]
    )
    
    # After the training is done, the recorder callback has now internally stored the snapshots of the 
    # graph embeddings during the training and now we need to store those into the experiment memory so 
    # that they can be analyzed later
    for epoch, embeddings in embeddings_recorder_callback.epoch_embeddings_map.items():
        e[f'intermediate_embeddings/{rep_index}/{sweep_key}/{epoch}'] = embeddings
    
    return model


@experiment.hook('train_model_pack_megan')
def train_model_megan_cluster(e: Experiment,
                              index_data_map: dict,
                              indices,
                              rep_index: int,
                              sweep_key: str = 'pack_megan',
                              **kwargs):
    
    if e.__TESTING__:
        e.EPOCHS = 3
        e.RUN_EAGERLY = True
            
    e.log(f'training PackMEGAN model...')
    model = EctMegan(
        units=e.UNITS,
        final_units=e.FINAL_UNITS,
        importance_channels=e.IMPORTANCE_CHANNELS,
        importance_factor=e.IMPORTANCE_FACTOR,
        importance_multiplier=e.IMPORTANCE_MULTIPLIER,
        final_activation=e.FINAL_ACTIVATION,
        regression_reference=e.REGRESSION_REFERENCE,
        regression_weights=e.REGRESSION_WEIGHTS,
        sparsity_factor=e.SPARSITY_FACTOR,
        # v2 specific
        fidelity_factor=e.FIDELITY_FACTOR,
        fidelity_funcs=e.FIDELITY_FUNCS,
        embedding_units=e.EMBEDDING_UNITS,
    )
    model.compile(
        optimizer=e.OPTIMIZER_CB(),
        loss=e.LOSS_CB(),
        run_eagerly=e.RUN_EAGERLY,
    )
    
    indices_support = random.sample(indices, k=300)
    indices_train = list(set(indices).difference(set(indices_support)))
    e.log(f'split the indices into {len(indices_train)} training indices and {len(indices_support)} support indices')
    
    graphs_train = [index_data_map[i]['metadata']['graph'] for i in indices_train]
    targets_train = [index_data_map[i]['metadata']['target'] for i in indices_train]
    
    x_train = tensors_from_graphs(graphs_train)
    y_train = (np.array(targets_train), *mock_importances_from_graphs(graphs_train, e.IMPORTANCE_CHANNELS))
    
    graphs_support = [index_data_map[i]['metadata']['graph'] for i in indices_support]
    
    epochs_warmup = int(e.EPOCHS * e.EPOCHS_WARMUP_RATIO)
    trainer = ClusterPackingTrainer(
        model=model,
        num_channels=e.IMPORTANCE_CHANNELS,
        cluster_batch_size=e.CLUSTER_BATCH_SIZE,
        min_samples=e.MIN_SAMPLES,
        epochs_warmup=epochs_warmup,
        factor=1.0,
        logger=e.logger
    )
    e.log(f'created packing trainer instance with {epochs_warmup} warmup epochs')
    
    # This call back can be used to record snapshots of the graph embedding latent space representations 
    # at different points during the training.
    embeddings_recorder_callback =  RecordIntermediateEmbeddingsCallback(
        epoch_step=e.INTERMEDIATE_EMBEDDINGS_STEP,
        active=e.RECORD_INTERMEDIATE_EMBEDDINGS,
        elements=graphs_train,
        logger=e.logger,
    )
    
    e.log('starting model training...')
    trainer.fit(
        x_train, y_train,
        epochs=e.EPOCHS,
        batch_size=e.BATCH_SIZE,
        callbacks=[
            embeddings_recorder_callback,
        ]
    )
    
    # After the training is done, the recorder callback has now internally stored the snapshots of the 
    # graph embeddings during the training and now we need to store those into the experiment memory so 
    # that they can be analyzed later
    for epoch, embeddings in embeddings_recorder_callback.epoch_embeddings_map.items():
        e[f'intermediate_embeddings/{rep_index}/{sweep_key}/{epoch}'] = embeddings
    
    return model


@experiment.hook('evaluate_model', replace=False, default=False)
def evaluate_model(e: Experiment,
                   model: t.Any,
                   index_data_map: dict,
                   indices: list,
                   rep_index: int,
                   sweep_key: str,
                   **kwargs,
                   ) -> t.Any:
    
    if e.__TESTING__:
        e.OUTLIER_FACTOR = 0.5
        
    e.log('saving the model...')
    model_path = os.path.join(e.path, f'{rep_index:02d}_{sweep_key}_model')
    model.save(model_path)
    
    e.log('evaluating model...')
    all_graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    graphs = [index_data_map[i]['metadata']['graph'] for i in indices]
    targets = [index_data_map[i]['metadata']['target'] for i in indices]

    # ~ PREDICTION EVALUATION
    e.log('making predictions and evaluating task performance...')
    predictions = model.predict_graphs(graphs)
    out_pred, ni_pred, ei_pred = list(zip(*predictions))
    out_pred = np.array(out_pred)
    out_true = np.array(targets)
    
    if e.DATASET_TYPE == 'regression':
        
        fig, rows = plt.subplots(
            ncols=e.NUM_TARGETS, 
            nrows=1, 
            figsize=(e.NUM_TARGETS * 8, 8),
            squeeze=False,
        )
        for target_index in range(e.NUM_TARGETS):
            ax = rows[0][target_index]
            values_true = out_true[:, target_index]
            values_pred = out_pred[:, target_index]
            
            r2_value = r2_score(out_true, out_pred)
            mse_value = mse_score(out_true, out_pred)
            e[f'r2/{rep_index}/{sweep_key}'] = r2_value
            e[f'mse/{rep_index}/{sweep_key}'] = mse_value
                        
            plot_regression_fit(values_true, values_pred, ax)
            ax.set_title(f'target {target_index} - r2: {r2_value:.2f} - mse: {mse_value:.2f}')
            
            e.log(f' * target {target_index} - r2: {r2_value:.2f} - mse: {mse_value:.2f}')
            
        fig.savefig(os.path.join(e.path, f'{rep_index:02d}_{sweep_key}_regression.pdf'))
        plt.close(fig)
            
    # ~ EXPLANATION EVALUATION
    e.log('evaluating local explanation metrics...')
    
    e.log('leave-one-out analysis...')
    results = leave_one_out_analysis(
        model=model,
        graphs=graphs,
        num_targets=e.NUM_TARGETS,
        num_channels=e.IMPORTANCE_CHANNELS,
    )
    fig = plot_leave_one_out_analysis(
        results=results,
        num_targets=e.NUM_TARGETS,
        num_channels=e.IMPORTANCE_CHANNELS,
        num_bins=e.NUM_BINS,
        x_lim=e.VALUE_RANGE,   
    )
    fig.savefig(os.path.join(e.path, f'{rep_index:02d}_{sweep_key}_leave_one_out.pdf'))
    plt.close(fig)
            
    # ~ EMBEDDING EVALUATION
    e.log('evaluating graph embeddings...')
    # embeddings: (B, K, D)
    embeddings = model.embedd_graphs(all_graphs)
    index_tuples = np.array([[i, c]for c in range(e.IMPORTANCE_CHANNELS) for i, _ in enumerate(embeddings)])
    b, k, d = embeddings.shape
    embeddings_all = embeddings.reshape((b * k, d))
    e.log(f' * embeddings shape: {embeddings.shape} - {embeddings_all.shape}')
    
    e.log('local outlier detection for the embeddings...')
    lof = LocalOutlierFactor(
        n_neighbors=e.OUTLIER_NEIGHBORS, 
        contamination=e.OUTLIER_FACTOR, 
        metric='cosine'
    )
    lof.fit(embeddings_all)
    num_outliers = int(len(embeddings_all) * e.OUTLIER_FACTOR)
    outlier_indices = np.argsort(lof.negative_outlier_factor_)[:num_outliers]
    is_omitted = np.zeros(shape=(b * k, ))
    is_omitted[outlier_indices] = 1
    is_omitted = is_omitted.reshape((b, k))

    embeddings_filtered = np.delete(embeddings_all, outlier_indices, axis=0)
    e.log(f' * filtered embeddings shape: {embeddings_filtered.shape}')
    
    if embeddings_all.shape[-1] == 2:
        e.log('the latent shape is 2D already - not performing any mapping...')
        class Mapper:
            pass
        
        mapper = Mapper()
        mapper.transform = lambda value: value
    else:
        e.log('fitting umap transformation...')
        mapper = UMAP(
            n_neighbors=e.UMAP_NUM_NEIGHBORS,
            min_dist=e.UMAP_MIN_DIST,
            metric=e.UMAP_METRIC,
            repulsion_strength=e.UMAP_REPULSION_STRENGTH,
        )
        mapper.fit(embeddings_filtered)
        
    
    e.log('visualizing the embeddings...')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
    ax.set_title(f'{sweep_key} Graph Embeddings\n'
                 f'{model.graph_embedding_shape[-1]}D to 2D ({mapper.__class__.__name__})')
    
    for channel_index, color in zip(range(model.importance_channels), e.CHANNEL_COLORS):
        channel_embeddings = embeddings[:, channel_index, :]
        channel_embeddings = np.delete(channel_embeddings, np.where(is_omitted[:, channel_index] == 1), axis=0)
        e.log(f' * ch. {channel_index} - num elements: {len(channel_embeddings)}')
        
        
        channel_mapped = mapper.transform(channel_embeddings)
        ax.scatter(
            channel_mapped[:, 0], channel_mapped[:, 1],
            c=color,
            s=1,
            label=f'ch. {channel_index}'
        )
        
    fig.savefig(os.path.join(e.path, f'{rep_index:02d}_{sweep_key}_graph_embeddings.pdf'))
    plt.close(fig)


@experiment.analysis
def analysis(e: Experiment):
    
    @e.hook('analyze_intermediate_embeddings', default=True)
    def analyze_intermediate_embeddings(e: Experiment, 
                                        rep: int,
                                        key: str,
                                        n_neighbors: int = 25,
                                        ):
        
        if 'intermediate_embeddings' not in e.data or key not in e[f'intermediate_embeddings/{rep}']:
            return
    
        e.log(f'starting to analyze the intermediate embeddings for key {key} at rep {rep}...')
        epochs: t.List[int] = []
        
        epoch_distances: t.List[t.List[float]] = []
        epoch_variances: t.List[t.List[float]] = []
        epoch_deltas: t.List[t.List[float]] = []
        
        prev_embeddings = None
        for epoch, embeddings in e[f'intermediate_embeddings/{rep}/{key}'].items():
            epoch = int(epoch)
            epochs.append(epoch)
            
            # embeddings: (B, D) or (B, K, D)
            # depending on whether the model is a multi channel model or not these embeddings could have a different 
            # shape, which is why we need to potentially clean this up here by merging the channel dimension into the
            # batch dimension - essentially treating the result from each channel as it's completely own embedding.
            embeddings = np.array(embeddings)
            if len(embeddings.shape) == 3:
                b, k, d = embeddings.shape
                embeddings = embeddings.reshape((b*k, d))
            
            # ~ average local distances
            # One interesting thing that we want to look at is the evolution of the average density in the latent space.
            # This is apparently not so trivial for high dimensional spaces but I think a good approximation is to calculate
            # k nearest neighbors for each embedding and then look at the average distances there.
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(embeddings)
            
            distances, _ = nn.kneighbors(embeddings, n_neighbors=n_neighbors)
            distances = np.mean(distances, axis=-1)
            epoch_distances.append(distances)
            
            # ~ average global variance
            # Another important perspective that we want to look at in the context of the evolution of the embeddings is the 
            # global view. Essentially how different are they from each other generally aka how big is the space over which 
            # the entire latent space is distributed.
            variances = np.std(embeddings, axis=0)
            epoch_variances.append(variances)
            
            # ~ embedding deltas
            # Also very interesting is to know how much the embeddings change over the course of the training. Do they change 
            # the most in the beginning or only later on?
            if prev_embeddings is None:
                epoch_deltas.append([0])
            
            else:
                # distances: (B, 1)    
                dist = lambda a, b: cosine(a, b)
                distances = np.array([dist(emb, emb_prev) for emb, emb_prev in zip(embeddings, prev_embeddings)])
                epoch_deltas.append(distances)
                
            prev_embeddings = embeddings
            
            
        fig, (ax_dist, ax_var, ax_delt) = plt.subplots(ncols=3, nrows=1, figsize=(3 * e.FIG_SIZE_BASE, e.FIG_SIZE_BASE))
        
        ax_dist.set_title(f'Average Local Embedding Distance - {n_neighbors} Neighbors')
        ax_dist.set_ylabel('Avg. Distance')
        ax_dist.set_xlabel('Epoch')
        plot_distributions(
            ax=ax_dist,
            xs=epochs,
            values_list=epoch_distances,
            color=e.COLOR_PRIMARY,
        )

        ax_var.set_title(f'Average Embedding Variance')
        ax_var.set_ylabel('Avg. Variance')
        ax_var.set_xlabel('Epoch')
        plot_distributions(
            ax=ax_var,
            xs=epochs,
            values_list=epoch_variances,
            color=e.COLOR_PRIMARY,
        )
        
        ax_delt.set_title(f'Average Distance to previous Embeddings')
        ax_delt.set_ylabel(f'Avg. Delta Distance')
        ax_delt.set_xlabel(f'Epoch')
        plot_distributions(
            ax=ax_delt,
            xs=epochs,
            values_list=epoch_deltas,
            color=e.COLOR_PRIMARY,
        )

        fig.savefig(os.path.join(e.path, f'{rep}_{key}_intermediate_embeddings.pdf'))
        plt.close(fig)

    
    if e.RECORD_INTERMEDIATE_EMBEDDINGS:
        
        for rep in range(e.REPETITIONS):
            for key in e.SWEEP_KEYS:
                
                e.apply_hook(
                    'analyze_intermediate_embeddings',
                    rep=rep,
                    key=key,
                )
    

experiment.run_if_main()
