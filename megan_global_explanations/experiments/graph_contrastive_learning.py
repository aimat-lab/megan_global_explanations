"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import yaml
import random
import pathlib
import typing as t

import umap
import umap.plot
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import DenseEmbedding
from sklearn.cluster import OPTICS
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from vgd_counterfactuals.base import CounterfactualGenerator
from vgd_counterfactuals.generate.colors import get_neighborhood
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.training import NoLoss
from graph_attention_student.models.utils import tf_ragged_random_binary_mask
from umap import UMAP

PATH = pathlib.Path(__file__).parent.absolute()

# == DATSET GENERATION PARAMETERS ==

SEED_GRAPHS = {
    'red-star': {
        'value': 'Y(R)(R)(R)(R)'  
    },
    'red-ring': {
        'value': 'R-1RRRRM-1'
    },
    'blue-star': {
        'value': 'Y(B)(B)(B)(B)'
    },
    'blue-ring': {
        'value': 'B-1BBBBC-1',
    },
    'green-grid': {
        'value': 'Y(G-1)(G-2-1)(G-3-2)(G-4-1-3)'
    },
    'green-ring': {
        'value': 'G-1GGGGY-1'
    },
    'red-green-ring': {
        'value': 'G-1RGRGM-1'
    }
}
COLORS = [
    [0.2, 0.2, 0.2], # dark gray
    [0.8, 0.8, 0.8], # dark gray
]

# == TRAINING PARAMETERS == 

BATCH_SIZE = 16
EPOCHS = 25

UMAP_NUM_NEIGHBORS: int = 100
UMAP_MIN_DIST: float = 0.0

OPTICS_MIN_SAMPLES: int = 10
OPTICS_XI: float = 0.0
CLUSTER_COLOR_MAP = {
    -1: 'lightgray',
    **dict(enumerate(mcolors.CSS4_COLORS.values()))
}

NUM_EXAMPLES: int = 5


# == MODEL PARAMETERS == 

UNITS = [32, 32, 32]
CONTRASTIVE_SAMPLING_FACTOR: float = 1.0
CONTRASTIVE_SAMPLING_TAU: float = 1.0
POSITIVE_SAMPLING_RATE: int = 3
POSITIVE_SAMPLING_NOISE: float = 0.4


class Model(ks.models.Model):
    
    def __init__(self,
                 units: t.List[int] = [32, 32, 32],
                 contrastive_sampling_factor: float = 1.0,
                 contrastive_sampling_tau: float = 1.0,
                 positive_sampling_noise: float = 0.1,
                 positive_sampling_rate: int = 1,
                 **kwargs):
        super(Model, self).__init__(self, **kwargs)
        self.units = units
        # contrastive learning
        self.contrastive_sampling_factor = contrastive_sampling_factor
        self.contrastive_sampling_tau = contrastive_sampling_tau
        self.positive_sampling_noise = positive_sampling_noise
        self.positive_sampling_rate = positive_sampling_rate
        
        self.conv_layers = []
        for k in units:
            lay = AttentionHeadGATV2(
                units=k,
                use_edge_features=True,
            )
            self.conv_layers.append(lay)
            
        self.lay_pool = PoolingNodes(pooling_method='sum')
        
        self.dense_layers = []
        self.dense_layers.append(DenseEmbedding(units=self.units[-1], activation='relu'))
        self.dense_layers.append(DenseEmbedding(units=self.units[-1], activation='linear'))
            
        
    @property
    def embedding_shape(self):
        return (self.units[-1], )
        
    def call(self, inputs, **kwargs):
        node_input, edge_input, edge_indices = inputs
        
        node_embeddings = node_input
        for lay in self.conv_layers:
            node_embeddings = lay([node_embeddings, edge_input, edge_indices])
            
        # graph_embeddings: ([B], D)
        graph_embeddings = self.lay_pool(node_embeddings)
        for lay in self.dense_layers:
            graph_embeddings = lay(graph_embeddings)
            pass
        
        return None, graph_embeddings
    
    def train_step(self, data, **kwargs):
        
        x, y = data
        node_input, edge_input, edge_indices = x
        
        with tf.GradientTape() as tape:
            loss = 0.0
            loss_contrast = 0.0
            
            y_pred, graph_embeddings = self([node_input, edge_input, edge_indices])
            
            if self.contrastive_sampling_factor != 0:
                
                batch_size = tf.shape(graph_embeddings)[0]
                # ~ L2 normalization
                # Here we apply an L2 normalization on the embeddings. This is VERY important because this allows us to 
                # later compute the cosine similarity between embedding vectors very easily as the dot product between 
                # them, which only works for L2 normalized vectors!
                graph_embeddings = tf.math.l2_normalize(graph_embeddings, axis=-1)
                
                # ~ negative sampling
                # For the negative sampling we simply need to create the multiplication of every graph embedding with 
                # every other graph embedding. The following code leverages "broadcasting" to achieve just that.
                graph_embeddings_expanded = tf.expand_dims(graph_embeddings, axis=-2)
                # graph_embeddings_product: ([B], [B])
                graph_embeddings_sim = tf.reduce_sum(graph_embeddings * graph_embeddings_expanded, axis=-1)  

                # Now the problem is that we do not want to consider the terms where the elements are being multiplied 
                # with themselves. So we mask them out by creating a diagonal mask of zeros here.
                eye = tf.cast(1.0 - tf.eye(num_rows=batch_size, num_columns=batch_size), tf.float32)
                #graph_embeddings_product = graph_embeddings_product * eye
                graph_embeddings_sim = graph_embeddings_sim * eye
                
                # The rest is the formula given in the paper which I assume is some variation of a cross entropy
                
                exponentials = tf.reduce_sum(tf.exp(graph_embeddings_sim / self.contrastive_sampling_tau), axis=-1)
                loss_contribs = tf.math.log(exponentials)
                #loss_neg = tf.reduce_mean(loss_contribs)
                
                #exponentials = tf.exp(tf.reduce_sum(graph_embeddings_sim, axis=-1) / self.contrastive_sampling_tau)
                # print(exponentials.shape)
                #loss_neg = tf.math.log(tf.reduce_sum(exponentials))
                
                loss_neg = tf.reduce_mean(tf.reduce_sum(graph_embeddings_sim / self.contrastive_sampling_tau, axis=-1))
                
                # In the first few iterations before the l2 normalization kicks in, this loss is basically guaranteed to 
                # be infinity. This is why we have to block the loss from influencing the weight updates in those cases.
                loss_contrast += loss_neg
                
                # ~ positive sampling
                # The idea is that we also need positive samples as anchor points. These positive samples are supposed to 
                # be examples of graph embeddings that are CLOSE to the original ones in the batch. We can obtain these 
                # by data augmentation techniques. For example we can make the assumption that despite small perturbations 
                # of the input graphs, the graph embedding should still be functionally the same.
                loss_pos = 0
                for p in range(self.positive_sampling_rate):
                    # first of all we create the perturbations for the input graphs and then we make another forward pass 
                    # with those inputs.
                    
                    # We can use this somewhat complicated code here to produce a random binary mask with the same shape as 
                    # the predicted node importances tensor. We effectively use this binary mask to drop out only a very few
                    # importances annotations.

                    # ni_pred_normalized = ni_pred / tf.reduce_max(ni_pred)
                    # ni_pred_binary = tf.cast(ni_pred_normalized > 0.5,tf.float32)
                    random_node_mask = tf_ragged_random_binary_mask(
                        template=node_input,
                        chances=[self.positive_sampling_noise, 1 - self.positive_sampling_noise],
                        n=1,
                    )
                    
                    _, graph_embeddings_mod =  self(
                        (node_input * random_node_mask, edge_input, edge_indices),
                        training=True,
                    )
                    graph_embeddings_mod = tf.math.l2_normalize(graph_embeddings_mod, axis=-1)
                    
                    # graph_embeddings_mod_sim = tf_pairwise_cosine_sim(graph_embeddings, graph_embeddings_mod)
                    # graph_embeddings_mod_sim = graph_embeddings_mod_sim * eye
                    # # The rest is the formula given in the paper which I assume is some variation of a cross entropy
                    # exponentials = tf.reduce_sum(tf.exp(graph_embeddings_mod_sim / self.contrastive_sampling_tau), axis=-1)
                    # loss_contribs = tf.math.log(exponentials)
                    # loss_contrast += (1 / self.positive_sampling_rate) * tf.reduce_mean(loss_contribs)
                    
                    loss_contribs_pos = tf.reduce_sum(graph_embeddings * graph_embeddings_mod, axis=-1)
                    loss_pos += tf.reduce_mean(loss_contribs_pos / self.contrastive_sampling_tau)
                    
                # As per the formula in the paper, we actually want to *maximize* this term which is why we add the negative 
                # of it to the overall loss value.
                if self.positive_sampling_rate != 0:
                    loss_contrast -= loss_pos
                    
                loss_contrast *= self.contrastive_sampling_factor
                loss += loss_contrast
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            # **{m.name: m.result() for m in self.metrics},
            'loss': loss,
        }
    
    
        
    

__DEBUG__ = True
__TESTING__ = False

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    if __TESTING__:
        e.EPOCHS = 10
    
    e.log('creating the processing instance...')
    processing = ColorProcessing()
    
    e.log('generating the dataset by applying random edits on the seeds...')
    dataset_path = os.path.join(e.path, 'dataset')
    os.mkdir(dataset_path)
    index: int = 0
    index_data_map = {}
    for seed_index, (seed_name, seed_data) in enumerate(SEED_GRAPHS.items()):
        
        # First of all we would like to visualize the seed graphs so that the user can visually verify that 
        # they are the correct ones in the end for that purpose we create a seperate folder into which we 
        # save the corresponding visual graph element representation
        seed_path = os.path.join(e.path, seed_name)
        os.mkdir(seed_path)
        processing.create(
            value=seed_data['value'],
            index=0,
            output_path=seed_path,
        )
        
        # Then we need to actually create the dataset here by applying all possible single graph edits on 
        # the seed graphs to basically create graph clusters.
        neighbors: t.List[dict] = get_neighborhood(
            value=seed_data['value'],
            processing=processing,
            # colors=COLORS,
        )
        
        # The previous function only returns a list of dicts, where each dict contains the domain specific 
        # representation of a graph and now we need to convert those into actual graphs to attach them to the 
        # dataset.
        for data in neighbors:
            processing.create(
                value=data['value'],
                index=index,
                output_path=dataset_path,
                additional_metadata={
                    'seed_index': seed_index,
                    'seed_name': seed_name,
                }
            )
            index += 1
            
        e.log(f' * seed "{seed_name}" done')
            
    # Now we have created the visual graph dataset files in the folder and can consequently load them now
    e.log('loading dataset...')
    reader = VisualGraphDatasetReader(
        path=dataset_path,
        logger=e.logger,
        log_step=100,
    )
    index_data_map = reader.read()
    dataset_length = len(index_data_map)
    e.log(f'loaded dataset with {dataset_length} elements')
        
    # ~ DATASET TO TENSORS
    indices = np.array([index for index in index_data_map.keys()])
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    x = tensors_from_graphs(graphs)
    y = np.array([0.0 for graph in graphs])
        
    # ~ TRAINING THE MODEL
    e.log('constructing model...')
    model = Model(
        units=UNITS,
        contrastive_sampling_factor=CONTRASTIVE_SAMPLING_FACTOR,
        contrastive_sampling_tau=CONTRASTIVE_SAMPLING_TAU,
        positive_sampling_noise=POSITIVE_SAMPLING_NOISE,
        positive_sampling_rate=POSITIVE_SAMPLING_RATE,
    )
    model.compile(
        # loss=ks.losses.CategoricalCrossentropy(),
        loss=NoLoss(),
    )
    e.log('starting model training process...')
    model.fit(
        x=x, 
        y=y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    
    # ~ EVALUATION OF LATENT EMBEDDINGS
    
    e.log('making predictions and embeddings for all elements...')
    y_pred, encoded = model(x)
    
    e.log('fitting UMAP mapper...')
    mapper = UMAP(
        random_state=1,
        n_components=2,
        metric='cosine',
        n_neighbors=UMAP_NUM_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
    )
    mapped = mapper.fit_transform(encoded)
    
    e.log('plotting ground truth cluster membership...')
    seed_names = np.array([data['metadata']['seed_name'] for data in index_data_map.values()])
    umap.plot.points(mapper, labels=seed_names)
    fig_path = os.path.join(e.path, 'graph_embeddings__gt.pdf')
    plt.savefig(fig_path)
    
    # ~ AUTOMATIC CLUSTERING
    
    @e.hook('visualize_graphs', default=True)
    def visualize_graphs(e: Experiment, 
                         index_data_map: dict, 
                         indices: t.List[int],
                         active_channel: int = 0,
                         ):
        num_graphs = len(indices)
        
        fig, rows = plt.subplots(
            ncols=num_graphs, 
            nrows=1,
            figsize=(8 * num_graphs, 8),
            squeeze=False
        )

        for i in range(num_graphs):
            ax = rows[0][i]
            
            index = indices[i]
            data = index_data_map[index]
            graph = data['metadata']['graph']
            node_positions = graph['node_positions']
            
            draw_image(ax, data['image_path'])
            ax.set_title(f'index: {index}')
                
        return fig
    
    @e.hook('clustering_results', default=True)
    def clustering_results(e: Experiment,
                           clustering_name: str,
                           clustering_labels: t.List[int],
                           indices: t.List[int],
                           encoded: t.List[list],
                           index_data_map: dict,
                           mapper: UMAP,
                           ) -> None:
        e.log(f'processing clustering results "{clustering_name}"...')
        clusters = list(sorted(set(clustering_labels)))
        clustering_labels = np.array(clustering_labels)
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        pdf = PdfPages(os.path.join(e.path, f'{clustering_name}__examples.pdf'))
        pdf.__enter__()
        for label in clusters:
            encoded_ = encoded[clustering_labels == label]
            indices_ = indices[clustering_labels == label]
            
            mapped_ = mapper.transform(encoded_)
            ax.scatter(
                mapped_[:, 0], mapped_[:, 1],
                s=1,
                c=e.CLUSTER_COLOR_MAP[label],
                label=f'cluster ({label})'
            )
            
            # There may be some clustering methods (DBSCAN) which do not assign clusters to all elements 
            # and instead some elements might not be clustered at all. This will be represented by the 
            # special -1 cluster label. We do not want to process any statistics or examples for these 
            # non clustered examples because that would be meaningless.
            if label < 0:
                continue
            
            # We determine the centroid of the cluster (simple average of all the cluster embeddings) in 
            # the original embedding space here and can then transform this into the mapped space to 
            # visualize it there.
            centroid = np.mean(encoded_, axis=0, keepdims=True)
            e[f'{clustering_name}/centroid/{label}'] = centroid
            centroid_mapped = mapper.transform(centroid)
            ax.scatter(
                centroid_mapped[0, 0], centroid_mapped[0, 1],
                s=2,
                c='black',
                marker='x',
            )
            ax.annotate(
                xy=centroid_mapped[0],
                text=f'({label})',
            )
            
            # ~ examples
            indices_examples = []
            indices_examples = random.sample(indices_.tolist(), k=e.NUM_EXAMPLES)

            fig_examples = e.apply_hook(
                'visualize_graphs',
                index_data_map=index_data_map,
                indices=indices_examples,
                # active_channel=active_channel,
            )
            fig_examples.suptitle(f' cluster ({label})')
            pdf.savefig(fig_examples)
            plt.close(fig_examples)
            
            e.log(f' * cluster ({label:02d})')
            #       f' - num_elements: {len(channels):04d}'
            #       f' - channel: Ø{np.mean(channels):.1f}(±{np.std(channels):.1f})')
            
        ax.legend()
        fig_path = os.path.join(e.path, f'{clustering_name}__space.pdf')
        fig.savefig(fig_path)
        plt.close(fig)
        pdf.__exit__(None, None, None)
    
    e.log('performing automatic clustering...')
    clusterer = OPTICS(
        metric='cosine',
        min_samples=OPTICS_MIN_SAMPLES,
        xi=0.0,
    )
    clusterer.fit(encoded)
    labels = clusterer.labels_
    
    num_clusters = max(labels) + 1
    e.log(f'plotting the clustering results with {num_clusters} clusters...')
    e.apply_hook(
        'clustering_results',
        clustering_name='optics',
        clustering_labels=labels,
        encoded=encoded,
        indices=indices,
        index_data_map=index_data_map,
        mapper=mapper,
    )
    
    # ~ SAVING PARAMETERS
    # Here we save a dictionary with the parameter configuration that was used in this experiment so that these can be 
    # compared over multiple executions of the experiment
    e.log('saving the parameter configuration...')
    parameters = {
        'umao': {
            'min_dist': UMAP_MIN_DIST,
            'n_neighbors': UMAP_NUM_NEIGHBORS,
        },
        'model': {
            'units': UNITS,
            'contrastive_sampling_factor': CONTRASTIVE_SAMPLING_FACTOR,
            'contrastive_sampling_tau': CONTRASTIVE_SAMPLING_TAU,
            'positive_sampling_rate': POSITIVE_SAMPLING_RATE,
            'positive_sampling_noise': POSITIVE_SAMPLING_NOISE,
        },
        'optics': {
            'min_samples': OPTICS_MIN_SAMPLES,
        }
    }
    parameters_path = os.path.join(e.path, 'parameters.yml')
    with open(parameters_path, mode='w') as file:
        yaml.dump(parameters, file, sort_keys=False, Dumper=yaml.Dumper)


experiment.run_if_main()