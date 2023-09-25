"""
This experiment performs a sweep over different models from the MEGAN model family on the rb motifs 
synthetic graph regression dataset. This dataset is especially nice to study the clustering behavior of 
the various models because as a synthetic dataset, the ground truth clusters for the concepts are known.

The final analysis compares the clustering obtained from the latent spaces of the model with the ground 
truth clustering to check how closely they overlap. The analysis also computes some unsupervised clustering 
statistics such as silhouette score and db index.
"""
import os
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import tensorflow.keras as ks
from umap import UMAP
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.models.megan import Megan
from graph_attention_student.models.megan import Megan2
from graph_attention_student.training import NoLoss
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.data import mock_importances_from_graphs
from graph_attention_student.visualization import plot_regression_fit
from graph_attention_student.visualization import plot_leave_one_out_analysis
from graph_attention_student.fidelity import leave_one_out_analysis
from graph_attention_student.util import latex_table, latex_table_element_mean
from graph_attention_student.util import render_latex, array_normalize

from megan_global_explanations.visualization import plot_distributions
from megan_global_explanations.visualization import create_concept_cluster_report

PATH = pathlib.Path(__file__).parent.absolute()

CHANNEL_COLORS = [
    '#54d3ffff',
    '#ff2b81ff',
]

# == SWEEP PARAMETERS ==

SWEEP_KEYS = [ 'megan_1', 'megan_2', 'pack_megan']
DEVICE = 'cpu:0'

# == MEGAN MODEL PARAMETERS ==
UNITS = [32, 32, 32]
IMPORTANCE_CHANNELS = 2
IMPORTANCE_FACTOR = 1.0
IMPORTANCE_MULTIPLIER = 0.5
REGRESSION_REFERENCE = 0.0
REGRESSION_WEIGHTS = [1.0, 1.0]
FINAL_UNITS = [32, 16, 1]
FINAL_ACTIVATION = 'linear'
SPARSITY_FACTOR = 1.0
FIDELITY_FACTOR = 0.1
FIDELITY_FUNCS = [
    lambda org, mod: tf.nn.relu(mod - org),
    lambda org, mod: tf.nn.relu(org - mod),
]
CONTRASTIVE_SAMPLING_FACTOR = 1e-60
CONTRASTIVE_SAMPLING_TAU = 1.0
POSITIVE_SAMPLING_RATE = 1
POSITIVE_SAMPLING_NOISE_ATTRIBUTES = 0.3
POSITIVE_SAMPLING_NOISE_IMPORTANCES = 0.3

REPETITIONS: int = 10
LOSS_CB = lambda: [ks.losses.MeanSquaredError(), NoLoss(), NoLoss()]
OPTIMIZER_CB = lambda: ks.optimizers.Adam(learning_rate=0.001)
EPOCHS: int = 100
BATCH_SIZE: int = 32
RUN_EAGERLY: bool = False

# == UMAP MAPPER PARAMETERS ==
OUTLIER_NEIGHBORS: int = 100
OUTLIER_FACTOR: float = 0.01

UMAP_METRIC: str = 'euclidean'
UMAP_MIN_DIST: float = 0.0
UMAP_NUM_NEIGHBORS: int = 500
UMAP_REPULSION_STRENGTH: float = 1.0

# == EVALUATION PARAMETERS ==
VALUE_RANGE: tuple = (-4, +4)
NUM_BINS: int = 50

FIDELITY_THRESHOLD: float = 0.5
USE_LABEL_PROPAGATION: bool = True

KMEANS_NUM_CLUSTERS = 6

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

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'vgd_model_sweep__megan_versions.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

def get_true_cluster(motifs: dict):
    negative_cluster = -1
    if motifs['blue_triangle'] and motifs['blue_star']:
        negative_cluster = 0
    elif motifs['blue_triangle']:
        negative_cluster = 1
    elif motifs['blue_star']:
        negative_cluster = 2
        
    positive_cluster = -1
    if motifs['red_triangle'] and motifs['red_star']:
        positive_cluster = 3
    elif motifs['red_triangle']:
        positive_cluster = 4
    elif motifs['red_star']:
        positive_cluster = 5
        
    return (negative_cluster, positive_cluster)


@experiment.hook('evaluate_model', replace=False, default=False)
def evaluate_model(e: Experiment,
                   model: t.Any,
                   index_data_map: dict,
                   indices: list,
                   rep_index: int,
                   sweep_key: str,
                   **kwargs,
                   ) -> t.Any:
    
    # For this particular dataset, rbmotifs, we actually know the ground truth clusters of the concepts 
    # because those depend on the seeds which were used to determine the graph labels. And now depending 
    # on the specific combination of seeds used for each graph we can assign a ground truth cluster.
    e.log('ground truth clustering...')
    for index, data in index_data_map.items():
        cluster_indices: t.Tuple[int, int] = get_true_cluster(data['metadata']['motifs']) 
        e[f'cluster/true/{rep_index}/{sweep_key}/{index}'] = cluster_indices
        e[f'cluster/pred/{rep_index}/{sweep_key}/{index}'] = [-1, -1]

    # Now in the next step we have to evaluate the results of the automatic clustering based on the latent 
    # space of the model that was just trained.
    # The first step for this is to create all the node embeddings and then also calculate all the fidelities 
    # for the entire dataset. Then we actually want to filter by fidelty. Low fidelty embeddings will be 
    # ignored since those are either inert motifs or empty masks which are irrelevant in terms of " semantic concepts"
    e.log('calculating dataset graph embeddings...')
    all_graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    all_embeddings = model.embedd_graphs(all_graphs)
    e.log('calculating dataset fidelities...')
    all_leave_one_out = model.leave_one_out_deviations(all_graphs)
    if 'regression':
        all_fidelities = [[-v[0][0], +v[1][0]] for v in all_leave_one_out]
        
    # In this step we will now actually filter all the embeddings which do not satisfy the fidelity threshold.
    index_embedding_tuples = []
    for index, emb, fid in zip(index_data_map.keys(), all_embeddings, all_fidelities):
        
        # We'll also want to save the graph embeddings here because we will need them during the analysis later to 
        # create the umap visualizations. As well as the fidelity values
        e[f'graph_embedding/{rep_index}/{sweep_key}/{index}'] = emb
        e[f'fidelity/{rep_index}/{sweep_key}/{index}'] = fid
        
        for channel_index in range(model.importance_channels):
            if fid[channel_index] >= e.FIDELITY_THRESHOLD:
                index_embedding_tuples.append((index, channel_index, emb[channel_index]))

    e[f'index_tuples/{rep_index}/{sweep_key}'] = [[i, k] for i, k, _ in index_embedding_tuples]
    embeddings = np.array([emb for index, channel_index, emb in index_embedding_tuples])
    e.log(f'embedding shape after filter: {embeddings.shape}')
    
    if 'ect_trainer' in e.data and sweep_key in e['ect_trainer']:
        e.log('clustering with the cluster tree...')
        trainer = e[f'ect_trainer/{sweep_key}/{rep_index}']
        labels = trainer.predict_embeddings(embeddings)
    
    else:
        # We can use KMEANS clustering here because for the synthetic dataset we actually know how many clusters have to exist!
        clusterer = KMeans(
            n_clusters=KMEANS_NUM_CLUSTERS,
        )
        # clusterer = GaussianMixture(
        #     n_components=KMEANS_NUM_CLUSTERS,
        #     covariance_type='full',
        # )
        e.log(f'{clusterer.__class__.__name__} clustering on the latent space...') 
        clusterer.fit(embeddings)
        labels = clusterer.labels_
        # labels = clusterer.predict(embeddings)
    
    e.log(f'clustering labels shape: {labels.shape} - contains None? {None in labels}')
    e.log(f'found {max(labels) + 1} clusters')
        
    e.log('saving clustering results...')
    for cluster_index, (index, channel_index, emb) in zip(labels, index_embedding_tuples):
        cluster_index = int(cluster_index)
        e[f'cluster/pred/{rep_index}/{sweep_key}/{index}'][channel_index] = cluster_index
        
    e.log('plotting the clusters...')
    
    deviations = model.leave_one_out_deviations(all_graphs)
    predictions = model.predict_graphs(all_graphs)
    fidelities = []
    for graph, dev, (out, ni, ei) in zip(all_graphs, deviations, predictions):
        
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
    
    cluster_labels = list(set(labels))
    num_clusters = len(cluster_labels)
    cluster_data_list = []
    for c, label in enumerate(cluster_labels):
        cluster_index = label
        cluster_embeddings = [emb for emb, l in zip(embeddings, labels) if l == label]
        cluster_index_tuples = [[i, k] for (i, k, _), l in zip(index_embedding_tuples, labels) if l == label]
        cluster_data = {
            'index': cluster_index,
            'embeddings': cluster_embeddings,
            'index_tuples': cluster_index_tuples,
            'graphs': [index_data_map[i]['metadata']['graph'] for i, k in cluster_index_tuples],
            'image_paths': [index_data_map[i]['image_path'] for i, k in cluster_index_tuples],
        }
        cluster_data_list.append(cluster_data)
        e.log(f' * ({c}/{num_clusters}) done')

    # creating the clustering report
    e.log('creating the clustering report PDF...')
    report_path = os.path.join(e.path, f'{rep_index}_{sweep_key}_cluster_report.pdf')
    create_concept_cluster_report(
        cluster_data_list=cluster_data_list,
        path=report_path,
        logger=e.logger,
        dataset_type=e.DATASET_TYPE,
        examples_type='centroid',
        num_examples=20,
    )
        
        

# == ANALYSIS ==
# Now in the case of the RbMotifs experiment we actually need to perform some analyses. At the core 
# rbmotifs is a synthetic dataset where we know the ground truth of which graph contains which 
# explanatory motif which also means we know a ground truth cluster assignment.
@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')
    e.log(list(e.data.keys()))
    
    e.log('calculating cluster metrics...')
    for key in e['sweep_keys']:
        
        for rep in e['repetitions']:
            
            labels_true = []
            labels_pred = []
            embeddings = []
            for index in e['indices']:
                for channel_index in range(2):
                    # We'll only look at actual semantic clusters which means that only where the ground truth 
                    # cluster label is NOT the special -1 value which indicates that none of the seed graphs 
                    # is contained.
                    if e[f'cluster/true/{rep}/{key}/{index}'][channel_index] >= 0:
                        labels_true.append(e[f'cluster/true/{rep}/{key}/{index}'][channel_index])
                        labels_pred.append(e[f'cluster/pred/{rep}/{key}/{index}'][channel_index])
                        
                        embeddings.append(e[f'graph_embedding/{rep}/{key}/{index}'][channel_index])
                        
            labels_true = np.array(labels_true)
            labels_pred = np.array(labels_pred)
            embeddings = np.array(embeddings)
            
            e[f'rand_score/{rep}/{key}'] = adjusted_rand_score(labels_true, labels_pred)
            e[f'completeness/{rep}/{key}'] = completeness_score(labels_true, labels_pred)
            e[f'v_measure/{rep}/{key}'] = v_measure_score(labels_true, labels_pred)
            e[f'nmi/{rep}/{key}'] = normalized_mutual_info_score(labels_true, labels_pred)

            e[f'num_clusters/{rep}/{key}'] = max(labels_pred) + 1
            e[f'silhouette/{rep}/{key}'] = silhouette_score(embeddings, labels_pred, metric='cosine')
            e[f'davis_bouldin/{rep}/{key}'] = davies_bouldin_score(embeddings, labels_pred)
    
    # After we have processed all the data we can now save the updated experiment state before we go on 
    # to actually create all the visualization artifacts. 
    #e.log('saving experiment data...')
    # experiment.save_data()
    
    e.log('printing metrics...')
    metric_dict = {
        'r2': {
            'title': r'$r^2 \uparrow$' 
        },
        'nmi': {
            'title': r'NMI $\uparrow$'
        },
        'rand_score': {
            'title': r'Adjusted Rand Score $\uparrow$'
        }, 
        'completeness': {
            'title': r'Completeness Score $\uparrow$'
        }, 
        'v_measure': {
            'title': r'V-Measure $\uparrow$'    
        }, 
        'silhouette': {
            'title': r'Silhouette Score $\uparrow$'
        },
        'davis_bouldin': {
            'title': r'DBI $\downarrow$'
        },
        'num_clusters': {
            'title': r'No. Clusters'
        }
    }
    column_names = ['Sweep Key'] + [d['title'] for d in metric_dict.values()]
    rows = []
    for key in e['sweep_keys']:
        e.log(f'\nkey "{key}"')
        row = [key.replace('_', ' ')]
        
        for metric_key, metric_data in metric_dict.items():
            
            values = [e[f'{metric_key}/{rep}/{key}'] for rep in e['repetitions']]
            avg = np.mean(values)
            std = np.std(values)
            row.append(values)
            
            e.log(f'  * {metric_key}: Ø{avg:.2f} (±{std:.2f})')

        rows.append(row)
        
    content, table = latex_table(
        column_names=column_names,
        rows=rows,
        list_element_cb=latex_table_element_mean,
        caption=f'Results of {e.REPETITIONS}'
    )
    pdf_path = os.path.join(e.path, f'result_table.pdf')
    render_latex({'content': table}, output_path=pdf_path)


experiment.run_if_main()