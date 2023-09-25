"""
This experiment
"""
import os
import pathlib
import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks

from umap import UMAP
from pacmap import PaCMAP
from hdbscan import HDBSCAN
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.neighbors import LocalOutlierFactor
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.data import mock_importances_from_graphs
from graph_attention_student.training import NoLoss
from graph_attention_student.models.megan import Megan2
from graph_attention_student.models.utils import ChangeVariableCallback
from graph_attention_student.models import load_model
from graph_attention_student.utils import array_normalize
from graph_attention_student.fidelity import leave_one_out_analysis
from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.visualization import plot_regression_fit

from megan_global_explanations.models import EctMegan, load_model
from megan_global_explanations.deep_ect import DeepEctTrainer
from megan_global_explanations.pack import ClusterPackingTrainer
from megan_global_explanations.visualization import generate_contrastive_colors
from megan_global_explanations.visualization import create_concept_cluster_report
from megan_global_explanations.visualization import animate_deepect_history

# == EXPERIMENT PARAMETERS ==
PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True
__TESTING__ = False

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
DATASET_TYPE: str = 'regression' # or 'classification'
SUBSET: t.Optional[int] = None
NUM_TEST: int = 1000
NUM_TARGETS: int = 1
CHANNEL_DETAILS: dict = {
    0: {
        'name': 'negative',
        'color': 'lightskyblue',
    },
    1: {
        'name': 'positive',
        'color': 'lightcoral',
    }
}

# == MODEL PARAMETERS == 
UNITS = [32, 32, 32]
EMBEDDING_UNITS = [32, 32, 32]
IMPORTANCE_CHANNELS = 2
IMPORTANCE_FACTOR = 1.0
IMPORTANCE_MULTIPLIER = 0.5
REGRESSION_REFERENCE = 0.0
REGRESSION_WEIGHTS = [1.0, 1.0]
FINAL_UNITS = [32, 16, 1]
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 1.0
FIDELITY_FACTOR = 0.2
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]

PACKING_EPOCHS_WARMUP: int = 50
PACKING_MIN_SAMPLES: int = 20
PACKING_FACTOR: float = 0.1
PACKING_BATCH_SIZE: int = 2000

# == TRAINING PARAMETERS ==
LOSS_CB = lambda: [ks.losses.MeanSquaredError(), NoLoss(), NoLoss()]
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(
    learning_rate=0.001,
)
EPOCHS = 50
BATCH_SIZE = 16

# == CONCEPT EXTRACTION ==
FIDELITY_THRESHOLD: float = 0.5

REMOVE_OUTLIERS: bool = True
OUTLIER_FACTOR: float = 0.2

UMAP_NUM_NEIGHBORS: int = 300
UMAP_MIN_DIST: float = 0.0
UMAP_METRIC: str = 'euclidean'
UMAP_REPULSION_STRENGTH: float = 1.0

HDBSCAN_MIN_SAMPLES: int = 50
HDBSCAN_MIN_CLUSTER_SIZE: int = 100
HDBSCAN_METHOD = 'leaf'

# == VISUALIZATION PARAMETERS ==
SCATTER_SIZE: float = 1.0
FIG_SIZE = (8, 8)


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    if e.__TESTING__:
        e.SUBSET = e.NUM_TEST * 2
        e.EPOCHS = 1
        e.CONTRASTIVE_SAMPLING_FACTOR = 0.0

    e['device'] = tf.device('cpu:0')
    e['device'].__enter__()

    # ~ LOADING THE DATASET
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=e.VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read(subset=e.SUBSET)
    indices = list(index_data_map.keys())
    e['indices'] = indices
    e.log(f'loaded dataset with {len(indices)} elements')
    
    # ~ PROCESSING DATASET
    e.log('processing the dataset into tensors...')
    test_indices = random.sample(indices, k=e.NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices
    
    support_indices = random.sample(indices, k=500)
    train_indices = list(set(train_indices).difference(set(support_indices)))
    graphs_support = [index_data_map[i]['metadata']['graph'] for i in support_indices]
    
    graphs_train = [index_data_map[i]['metadata']['graph'] for i in train_indices]
    graphs_test = [index_data_map[i]['metadata']['graph'] for i in test_indices]
    
    x_train = tensors_from_graphs(graphs_train)
    y_train = (
        np.array([graph['graph_labels'] for graph in graphs_train]),
        *mock_importances_from_graphs(graphs_train, e.IMPORTANCE_CHANNELS),   
    )

    x_test = tensors_from_graphs(graphs_test)
    y_test = (
        np.array([graph['graph_labels'] for graph in graphs_test]),
        *mock_importances_from_graphs(graphs_test, e.IMPORTANCE_CHANNELS),
    )
    
    # ~ TRAINING MODEL
    e.log(f'training EctMEGAN model...')
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
        embedding_units=e.EMBEDDING_UNITS,
        # v2 specific
        fidelity_factor=e.FIDELITY_FACTOR,
        fidelity_funcs=e.FIDELITY_FUNCS,
    )
    model.compile(
        optimizer=e.OPTIMIZER_CB(),
        loss=e.LOSS_CB(),
        run_eagerly=False,
    )
    
    # trainer = DeepEctTrainer(
    #     model=model,
    #     elements=graphs_train,
    #     split_epoch_step=2,
    #     epochs_warmup=25,
    #     min_cluster_size=2,
    #     projection_factor=2.0,
    #     logger=e.logger,
    #     save_history=True,
    # )
    # trainer.initialize()
    trainer = ClusterPackingTrainer(
        model=model,
        num_channels=e.IMPORTANCE_CHANNELS,
        logger=e.logger,
        cluster_batch_size=e.PACKING_BATCH_SIZE,
        min_samples=e.PACKING_MIN_SAMPLES,
        epochs_warmup=e.PACKING_EPOCHS_WARMUP,
        factor=e.PACKING_FACTOR,
    )
    
    e.log('starting the model training...')
    trainer.fit(
        x_train, y_train,
        epochs=e.EPOCHS,
        # epochs=100,
        batch_size=e.BATCH_SIZE,
    )
    
    e.log('finished training. saving the model...')
    model_path = os.path.join(e.path, 'model')
    model.save(model_path)
    
    trainer_path = os.path.join(e.path, 'trainer.json')
    trainer.save(trainer_path)
    
    # e.log('saving the ect embedding history...')
    # anim_path = os.path.join(e.path, 'history.mp4')
    # animate_deepect_history(trainer.history, anim_path)
    
    # ~ MODEL EVALUATION
    e.log('evaluating model...')
    
    e.log('leave one out analysis...')
    results = leave_one_out_analysis(
        model=model,
        graphs=graphs_test,
        num_targets=e.NUM_TARGETS,
        num_channels=e.IMPORTANCE_CHANNELS,
    )
    fig = plot_leave_one_out_analysis(
        results=results,
        num_targets=e.NUM_TARGETS,
        num_channels=e.IMPORTANCE_CHANNELS,
    )
    fig_path = os.path.join(e.path, 'leave_one_out.pdf')
    fig.savefig(fig_path)
    
    e.log('prediction performance...')
    
    # For a classification task we calculate the the accuracy and draw the roc curve
    if e.DATASET_TYPE == 'classification':
        
        out_pred, ni_pred, ei_pred = model(x_test)
        out_test = y_test[0]
        
        out_pred_softmax = softmax(out_pred)
        values_pred = out_pred_softmax[:, 1]
        
        labels_pred = np.argmax(out_pred, axis=-1)
        labels_true = np.argmax(out_test, axis=-1)
        
        acc = accuracy_score(labels_true, labels_pred)
        e['prediction/accuracy'] = acc

        # ~ plotting ROC Curve
        fpr, tpr, _ = roc_curve(labels_true, values_pred)
        auroc = auc(fpr, tpr)
        e['prediction/auroc'] = auroc
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=e.FIG_SIZE)
        ax.plot(fpr, tpr, label=f'AUC: {auroc:.1f}')
        ax.plot([0, 1], [0, 1], alpha=0.5, ls='--')
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        fig.savefig(os.path.join(e.path, 'roc.pdf'))
        
        e.log(f'accuracy: {acc*100:.2f} - auroc: {auroc:.2f}')
     
    # For regression we calculate the r2 value and create a regression plot    
    elif e.DATASET_TYPE == 'regression':
        
        out_pred, ni_pred, ei_pred = model(x_test)
        out_test = y_test[0]
        
        mse_value = mse_score(out_test, out_pred)
        
        r2_value = r2_score(out_test, out_pred)
        e['prediction/r2_value'] = r2_value
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        plot_regression_fit(
            values_true=np.squeeze(out_test), 
            values_pred=np.squeeze(out_pred),
            ax=ax,
        )
        ax.set_title(f'R2: {r2_value:.2f} - MSE: {mse_value:.2f}')
        fig.savefig(os.path.join(e.path, 'regression.pdf'))
        
        e.log(f'r2: {r2_value:.2f} - mse: {mse_value:.2f}')
        


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')
    
    e.log('loading the model...')
    model_path = os.path.join(e.path, 'model')
    model = load_model(model_path)
    
    # e.log('loading the trainer...')
    # trainer_path = os.path.join(e.path, 'trainer.json')
    # trainer = DeepEctTrainer.load(trainer_path)
    # trainer.set_model(model)
    
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read(subset=e.SUBSET)
    
    # ~ CONCEPT EXTRACTION
    
    e.log('starting concept extraction...')
    extraction_indices = e['indices']
    e['extraction_indices'] = extraction_indices
    graphs = [index_data_map[i]['metadata']['graph'] for i in extraction_indices]
    
    # The first step to the concept extraction is to map ALL the elements of the dataset in to the 
    # latent space fo the model. One thing we have to be careful of here is that these embeddings are 
    # a 3-dim tensor because the embeddings are still separated by channel!
    # embeddings: (A, K, D)
    graph_embeddings = model.embedd_graphs(graphs)
    # we also need to compute the fidelity for each one of the input samples here because that will 
    # be required for the filter step that comes next. Important thing about the fidelity is that 
    # it is computed differently for regression and classifcation!
    # deviations: (A, K, C)
    deviations = model.leave_one_out_deviations(graphs)
    predictions = model.predict_graphs(graphs)
    fidelities = []
    for graph, dev, (out, ni, ei) in zip(graphs, deviations, predictions):
        
        # Some of the processing here depends on what kinf of problem we are dealing with here. Namely how 
        # we solve the prediction result and the fidelity.
        if e.DATASET_TYPE == 'regression':
            graph['graph_prediction'] = np.squeeze(out)
            fid = [-dev[0][0], +dev[1][0]]
            
        if e.DATASET_TYPE == 'classification':
            graph['graph_prediction'] = np.argmax(out)
            fid = [dev[i][i] for i in range(e.NUM_TARGETS)]
        
        fidelities.append(fid)
        
        graph['graph_fidelity'] = fid
        graph['graph_deviation'] = dev
        graph['node_importances'] = array_normalize(ni)
        graph['edge_importances'] = array_normalize(ei)
    
    # But we don't want to use all of these embeddigns because by definition the embeddings associated 
    # with a fidelity close to zero do not have any influence on the prediction at all which means that 
    # we dont have to care about those samples.
    # embeddings: (A * ?, D)
    embeddings = []
    # index_tuples: List of index tuples that identify the origin of each embedding
    index_tuples = []
    for index, emb, fid in zip(extraction_indices, graph_embeddings, fidelities):
        for channel_index in range(model.importance_channels):
            if fid[channel_index] > e.FIDELITY_THRESHOLD:
                embeddings.append(emb[channel_index, :])
                index_tuples.append([index, channel_index])
                
    e.log(f'embeddings shape: {np.array(embeddings).shape}')
                
    if e.REMOVE_OUTLIERS:
        e.log(f'removing the {e.OUTLIER_FACTOR*100:.1f}% most likely outliers...')
        lof = LocalOutlierFactor(
            contamination=e.OUTLIER_FACTOR,
            n_neighbors=25,
            metric=e.UMAP_METRIC,
        )
        outlier_scores = lof.fit_predict(embeddings)
        
        # The actual filtering we can now do by using that threshold value
        _embeddings = []
        _index_tuples = []
        for score, emb, tupl in zip(outlier_scores, embeddings, index_tuples):
            if score > 0:
                _embeddings.append(emb)
                _index_tuples.append(tupl)
                
        embeddings, index_tuples = _embeddings, _index_tuples
        
    e.log(f'after filtering: {len(embeddings)} embeddings remaining')
                
    
    # Now that we have extracted all the embeddings that we will ultimately need, we can fit the 
    # dimensionality retudciton mapper which will be used for all the visualizations.
    
    @e.hook('fit_mapper', default=True)
    def fit_mapper(e, embeddings):
        e.log('fitting umap transformation...')
        mapper = UMAP(
            n_neighbors=e.UMAP_NUM_NEIGHBORS,
            min_dist=e.UMAP_MIN_DIST,
            metric=e.UMAP_METRIC,
            repulsion_strength=e.UMAP_REPULSION_STRENGTH,
        )
        mapper.fit_transform(embeddings)
        return mapper
    
    mapper = e.apply_hook('fit_mapper', embeddings=embeddings)
    
    e.log('plotting the latent space...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=e.FIG_SIZE)
    for channel_index in range(model.importance_channels):
        channel_embeddings = []
        for emb, (i, k) in zip(embeddings, index_tuples):
            if k == channel_index:
                channel_embeddings.append(emb)

        mapped = mapper.transform(channel_embeddings)
        ax.scatter(
            mapped[:, 0], mapped[:, 1],
            color=e.CHANNEL_DETAILS[channel_index]['color'],
            label=e.CHANNEL_DETAILS[channel_index]['name'],
            s=e.SCATTER_SIZE,
        )
        
    ax.legend()
    fig_path = os.path.join(e.path, f'latent_space.pdf')
    fig.savefig(fig_path)
    
    # Now we can perform the automatic clustering to find the concept clusters 
    
    @e.hook('automatic_clustering', default=True)
    def hdbscan_clustering(e: Experiment,
                           embeddings: list):
        
        hdbscan = HDBSCAN(
            min_cluster_size=e.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=e.HDBSCAN_MIN_SAMPLES,
            cluster_selection_method=e.HDBSCAN_METHOD,
            cluster_selection_epsilon=0.0,
            metric='precomputed',
        )
        distances = pairwise_distances(embeddings, metric='cosine', n_jobs=-1)
        hdbscan.fit(distances.astype('float64'))
        
        return hdbscan.labels_
    
    e.log('automatic clustering...')
    # :hook automatic_clustering:
    #       This hook is just supposed to perform the clustering itself it receives the the (A, D) array 
    #       of all the embeddings and is supposed to return an array (A, ) which assigns an integer 
    #       cluster index to every one of these embeddings.

    cluster_labels = e.apply_hook(
        'automatic_clustering',
        embeddings=embeddings,
    )
    
    # e.log('gettting labels from cluster tree...')
    # cluster_labels = trainer.predict_embeddings(embeddings)
    
    # Now that we have a cluster assignment for every element, we can move on to the most important 
    # step which is the analysis of the cluster properties and the subsequent visualization of 
    # each cluster.
    cluster_indices = list(sorted(set(cluster_labels)))
    num_clusters = len(cluster_indices)
    e.log(f'found {num_clusters} clusters: {cluster_indices}')
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=e.FIG_SIZE)
    
    e.log('plotting the clustering...')
    colors = generate_contrastive_colors(num_clusters)
    cluster_data_list = []
    for c, (cluster_index, color) in enumerate(zip(cluster_indices, colors)):
        
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_index]
        
        cluster_index_tuples = [index_tuples[i] for i in cluster_indices]
        cluster_embeddings = np.array([embeddings[i] for i in cluster_indices])
        cluster_centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)
        
        cluster_mapped = mapper.transform(cluster_embeddings)
        centroid_mapped = mapper.transform(cluster_centroid)
        
        # There are some clustering methods which do not assign all the elements to a cluster but rather 
        # exclude some of them due to low confidence - those will be labeled with the special value -1
        if cluster_index < 0:
            ax.scatter(cluster_mapped[:, 0], cluster_mapped[:, 1], color='lightgray', s=e.SCATTER_SIZE)
            continue
    
        ax.scatter(
            cluster_mapped[:, 0], cluster_mapped[:, 1],
            color=color,
            label=f'cluster {cluster_index}',
            s=e.SCATTER_SIZE,
        )
        ax.scatter(
            centroid_mapped[0, 0], centroid_mapped[0, 1],
            color='black',
            marker='x',
        )
        ax.annotate(
            text=f'({cluster_index})',
            xy=[centroid_mapped[0, 0], centroid_mapped[0, 1]]
        )
        
        cluster_data = {
            'index': cluster_index,
            'embeddings': cluster_embeddings,
            'index_tuples': cluster_index_tuples,
            'graphs': [index_data_map[i]['metadata']['graph'] for i, k in cluster_index_tuples],
            'image_paths': [index_data_map[i]['image_path'] for i, k in cluster_index_tuples],
        }
        cluster_data_list.append(cluster_data)
        
        e.log(f' * ({c}/{num_clusters}) done')
        
    fig_path = os.path.join(e.path, 'latent_space__cluster.pdf')
    fig.savefig(fig_path)

    # creating the clustering report
    e.log('creating the clustering report PDF...')
    report_path = os.path.join(e.path, 'cluster_report.pdf')
    create_concept_cluster_report(
        cluster_data_list=cluster_data_list,
        path=report_path,
        logger=e.logger,
        dataset_type=e.DATASET_TYPE,
        examples_type='centroid',
        num_examples=20,
    )
    
    


experiment.run_if_main()