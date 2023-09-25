import os
import random
import logging
import typing as t

import hdbscan
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from scipy.special import softmax
from scipy.spatial.distance import euclidean
from hdbscan import HDBSCAN
from kgcnn.layers.modules import DenseEmbedding
from graph_attention_student.training import EpochCounterCallback
from graph_attention_student.models.utils import tf_pairwise_manhattan_distance, tf_pairwise_euclidean_distance
from graph_attention_student.models.utils import tf_manhattan_distance
from graph_attention_student.models.utils import tf_pairwise_cosine_sim
from graph_attention_student.models.utils import tf_pairwise_cauchy_sim
from graph_attention_student.training import NoLoss

from megan_global_explanations.utils import NULL_LOGGER


def gather_arrays(inp: t.Union[tf.Tensor, np.ndarray, t.List[tf.Tensor]], 
                 indices: t.List[int]
                 ):
    indices = list(indices)
    
    if isinstance(inp, (list, tuple)):
        return [gather_arrays(v, indices) for v in inp]
    elif isinstance(inp, np.ndarray):
        return inp[indices]
    else:
        # print(type(inp))
        return tf.gather(inp, indices)


class ClusterPackingMixin:

    def __init__(self):
        pass

    def set_clusters(self, num_clusters: int):
        self.cluster_layers = [
            DenseEmbedding(
                units=16,
                activation='swish',
                use_bias=True,
            ),
            DenseEmbedding(
                units=16,
                activation='swish',
                use_bias=True,
            ),
            DenseEmbedding(
                units=num_clusters,
                activation='softmax',
                use_bias=True,
            )
        ]

    def get_packing_loss(self):
        return tf.constant(0.0)


class ClusterPackingTrainer:
    
    def __init__(self,
                 model: ClusterPackingMixin,
                 num_channels: int = 1,
                 cluster_batch_size: int = 2048,
                 min_samples: float = 10,
                 epochs_warmup: int = 50,
                 factor: float = 0.1,
                 logger: logging.Logger = NULL_LOGGER,
                 record_history: bool = True,
                 **kwargs):
        self.model = model
        self.num_channels = num_channels
        self.set_model(model)
        self.cluster_batch_size = cluster_batch_size
        self.min_samples = min_samples
        self.epochs_warmup = epochs_warmup
        self.factor = factor
        self.logger = logger
        self.record_history = record_history
        
        self.clusterer = None
        self.compile_kwargs = {}
        
        # These are parameters of the "fit" method and will only be set to the true values when that method is actually called.
        self.batch_size: int = 0
        self.epochs: int = 0
        
        self.history = {}
        self.callback = EpochCounterCallback()
        setattr(self.callback, 'on_epoch_end', self.on_epoch_end)
        self.var_factor = tf.Variable(1e-50, trainable=False)
        
        self.is_warm: bool = False
        
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
    
    @property
    def epoch(self):
        return self.callback.epoch
    
    def log(self, message: str):
        self.logger.info(f' [ClusterPackingTrainer] {message}')
        
    def set_model(self, model):
        self.model = model
        setattr(self.model, 'get_packing_loss', self.get_packing_loss)
        return self.model
        
    def get_packing_loss(self, embeddings, clusters):
        
        loss = 0.0
        centers, masks = clusters
        
        # distances: ([B], )
        distances = tf_manhattan_distance(centers, embeddings)
        loss_contribs = tf.square(distances) * masks
        loss = tf.reduce_mean(loss_contribs)
        
        return self.var_factor * loss
        
    def __get_packing_loss(self, embeddings, clusters): 
        
        loss = 0.0
        
        labels, masks = clusters
        out = embeddings
        for lay in self.model.cluster_layers:
            out = lay(out)
            
        loss_contribs = tf.losses.categorical_crossentropy(labels, out)
        loss = tf.reduce_mean(loss_contribs * masks)
        
        return self.var_factor * loss
        
    def _get_packing_loss(self, embeddings, clusters):
        loss = tf.constant(1.0)
        loss_contribs = 0.0
        
        # distances = tf_pairwise_euclidean_distance(clusters, embeddings)
        # loss_contribs = -tf.math.top_k(-distances, k=1).values
        
        #clusters = tf.math.l2_normalize(clusters, axis=-1)
        #embeddings = tf.math.l2_normalize(embeddings, axis=-1)
        #similarities = tf_pairwise_cosine_sim(clusters, embeddings)
        distances = tf_pairwise_manhattan_distance(clusters, embeddings)
        cauchy_similarities = tf_pairwise_cauchy_sim(clusters, embeddings)
        cosine_similarities = tf_pairwise_cosine_sim(clusters, embeddings, normalize=True)
        # top_similarities: ([B], 2)
        bot_distances = -tf.math.top_k(-distances, k=2).values
        top_cauchy_similarities = tf.math.top_k(cauchy_similarities, k=2).values 
        top_cosine_similarities = tf.math.top_k(cosine_similarities, k=2).values
        # loss_contribs += -tf.reduce_mean(top_cosine_similarities[:, 0])
        # loss_contribs += -tf.reduce_mean(top_cauchy_similarities[:, 0])

        #bot_distances = -tf.math.top_k(-bot_distances[:, 0], k=16).values
        #loss_contribs += tf.reduce_mean(tf.square(bot_distances))
        
        loss_contribs += tf.reduce_mean(tf.square(bot_distances[:, 0]))
        #loss_contribs += tf.reduce_mean(top_cosine_similarities[:, 1])
        
        #print(loss_contribs.shape)
        loss = self.var_factor * tf.reduce_mean(loss_contribs)
        
        return loss
        
    def on_epoch_end(self, *args, **kwargs):
        # At the beggining of the epoch we want to perform the clustering which will then be 
        # used as the reference for the compression for that entire epoch.
        if self.callback.epoch >= self.epochs_warmup and not self.is_warm:
            self.var_factor.assign(self.factor)
            
            self.model.stop_training = True
            self.is_warm = True
        
        # If the warmup epoch is not yet reached we make sure that the packing loss is not active by 
        # setting the loss factor to a very low value.
        if self.epoch < self.epochs_warmup:
            # I think we cant assign flat out zero here because then the static graph will just not build
            # that branch at all so instead we set it to a very low value which is the same as zero in practice
            self.var_factor.assign(1e-200)
            
        # Then if the total number of epochs has been reached we obv. need to stop the training process of the 
        # model for good.
        if self.epoch > self.epochs:
            self.log(f'max. number of epochs ({self.epochs}) reached. stopping model training...')
            self.model.stop_training = True
    
    def create_clusters(self, inputs) -> np.ndarray:
        embeddings = self.model.embedd(inputs).numpy()
        
        self.clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=self.min_samples, 
            cluster_selection_method='eom',
            metric='manhattan',
            prediction_data=True,
        )
        self.clusterer.fit(embeddings)
        labels = self.clusterer.labels_
        
        # Now we need to get the center point for each of those clusters
        clusters = list(set(labels))
        centers = []
        for label in clusters:
            
            if label < 0:
                continue
            
            cluster_center = self.clusterer.weighted_cluster_centroid(label)
            centers.append(cluster_center)
            
        # Slightly shift the centers away from each other
        centers = np.array(centers)
        centers_shifted = []
        for i, center in enumerate(centers):
            other_centers = [vec for j, vec in enumerate(centers) if j != i]
            distances = [1 / (1e-6 + euclidean(center, vec)) for vec in other_centers]
            distances = softmax(distances)
            
            vectors = [ (center - vec) * (dist / np.linalg.norm(center - vec)) for vec, dist in zip(other_centers, distances)]
            direction = np.mean(vectors, axis=0)
            # print(direction)
            centers_shifted.append(center + direction * np.linalg.norm(center))
            
        centers_shifted = np.array(centers_shifted)
        return centers
    
    def generate(self, x, y, batch_size: int = 32):
        
        num_samples = len(y) if not isinstance(y, (list, tuple)) else len(y[0])
        indices = list(range(num_samples))
        
        cluster_centers = None
        one_hot = None
        mask = None
        centers = None
        
        clustering_indices = random.sample(indices, k=self.cluster_batch_size)
        clustering_inputs = gather_arrays(x, clustering_indices)
        cluster_centers = self.create_clusters(clustering_inputs)
        self.logger.info(f' * created clustering with shape: {cluster_centers.shape}')
        
        num_clusters = len(cluster_centers)
        self.logger.info(f' * modified and recompiled model')
        
        num_clusters = len(cluster_centers)
        embeddings = self.model.embedd(x).numpy()
        self.logger.info(f' * calculated dataset embeddings with shape: {embeddings.shape}')

        labels, _ = hdbscan.approximate_predict(self.clusterer, embeddings)
        identity_matrix = np.eye(num_clusters)
        
        one_hot = identity_matrix[labels]
        centers = cluster_centers[labels]
        # centers = np.concatenate((centers, centers), axis=-1)
        mask = np.where(labels < 0, 0.0, 1.0)
        self.logger.info(f' * calculated cluster labels for embeddings: {one_hot.shape}')
        
        self.logger.info('starting custom batch generator')
        epoch = 0
        
        while True:
            
            # shuffle the dataset
            indices_epoch = indices.copy()
            random.shuffle(indices_epoch)
            x_epoch = gather_arrays(x, indices_epoch)
            y_epoch = gather_arrays(y, indices_epoch)
            

            # if self.epoch == self.epochs_warmup and not self.is_warm:
            #     self.is_warm = True
            #     self.model.stop_training = True
            #     return
            
            # if self.epoch > epochs:
                
                
            indices_epoch_cluster = [i + k * num_samples for k in range(self.num_channels) for i in indices_epoch]
            # print(indices_epoch_cluster)
            one_hot_epoch = one_hot[indices_epoch]
            centers_epoch = centers[indices_epoch_cluster]
            mask_epoch = mask[indices_epoch_cluster]

            i_batch = 0
            while i_batch < num_samples:
                num_batch = min(batch_size, num_samples - i_batch)
                if num_batch < batch_size:
                    i_batch += num_batch
                    continue
                
                indices_batch = list(range(i_batch, i_batch+num_batch))
                x_batch = gather_arrays(x_epoch, indices_batch)
                y_batch = gather_arrays(y_epoch, indices_batch)
                
                one_hot_batch = one_hot_epoch[indices_batch]
                indices_batch_cluster = [i + k * num_samples for k in range(self.num_channels) for i in indices_batch]
                
                centers_batch = centers_epoch[indices_batch_cluster]
                mask_batch = mask_epoch[indices_batch_cluster]
                # mask_batch = mask_epoch[indices_batch]
                # centers_batch = centers_epoch[indices_batch]
                
                # yield (x_batch, y_batch, cluster_centers)
                yield (x_batch, y_batch, (centers_batch, mask_batch))
                
                i_batch += num_batch
                
            epoch += 1


    def compile(self, *args, **kwargs):
        self.compile_kwargs = kwargs
        self.model.compile(
            **self.compile_kwargs,
        )
    
    def fit(self, x, y, **kwargs):
        num_samples = len(y) if not isinstance(y, (list, tuple)) else len(y[0])
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']

        
        if 'callbacks' in kwargs:
            kwargs['callbacks'].append(self.callback)
        else:
            kwargs['callbacks'] = [self.callback]
            
        while self.epoch < self.epochs:
            self.logger.info(f'restarting model training at epoch {self.epoch}/{self.epochs}...')
            generator = self.generate(x, y, batch_size=self.batch_size)
            self.model.fit(
                generator,
                **kwargs,
                steps_per_epoch=num_samples // self.batch_size,
            )