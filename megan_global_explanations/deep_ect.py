import os
import time
import json
import random
import logging
import typing as t
from copy import deepcopy

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.semi_supervised import LabelPropagation
from scipy.spatial.distance import euclidean

from graph_attention_student.training import EpochCounterCallback
from graph_attention_student.models.utils import tf_euclidean_distance, tf_pairwise_euclidean_distance
from graph_attention_student.data import tensors_from_graphs

from megan_global_explanations.utils import NULL_LOGGER



class CustomJsonEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if isinstance(obj, (tf.Variable, )):
            return obj.numpy().tolist()


class DeepEctMixin():
    
    def __init__(self):
        pass
    
    def embedd(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()
    
    def convert_elements(self, elements: list) -> tf.Tensor:
        raise NotImplementedError()
    
    def embedd_elements(self, elements: list) -> np.ndarray:
        inputs = self.convert_elements(elements)
        return self.embedd(inputs).numpy()
    
    def ect_loss(self):
        return tf.constant(0.0)


class DeepEctTrainer:
    
    def __init__(self,
                 model: DeepEctMixin = None,
                 elements: t.List[t.Any] = [],
                 batch_size: int = 256,
                 split_epoch_step: int = 10,
                 epochs_warmup: int = 0,
                 min_cluster_size: int = 5,
                 projection_factor: float = 1.0,
                 logger: logging.Logger = NULL_LOGGER,
                 save_history: bool = False,
                 **kwargs):
        
        self.model = model
        self.elements = elements
        
        self.batch_size = batch_size
        self.split_epoch_step = split_epoch_step
        self.epochs_warmup = epochs_warmup
        self.min_cluster_size = min_cluster_size
        self.projection_factor = projection_factor
        self.logger = logger
        self.save_history = save_history
        
        self.indices = list(range(len(self.elements) * 1))
        self.inputs: t.Union[tf.Tensor, t.List[tf.Tensor]] = None
        
        self.batch_indices = []
        self.batch_inputs = []
        self.batch_embeddings = []
        
        self.set_model(model)
        
        self.callback =EpochCounterCallback()
        setattr(self.callback, 'on_train_batch_end', self.on_train_batch_end)
        setattr(self.callback, 'on_epoch_end', self.on_epoch_end)
        
        self.tree = {}
        self.leafs = []
        self.splits = []
        
        self.leaf_blacklist: t.Dict[int, int] = {}
        
        self.history = {}
        
        self.next_index: int = 0
        self.num_nodes: tf.Variable = tf.Variable(0, trainable=False)
        
    # -- PERSISTENT STORAGE --
    
    def save(self, path: str) -> None:
        
        with open(path, mode='w') as file:
            data = self.to_dict()
            content = json.dumps(data, cls=CustomJsonEncoder)
            file.write(content)
    
    @classmethod
    def load(cls, path: str) -> t.Any:
        
        with open(path, mode='r') as file:
            content = file.read()
            data = json.loads(content)
            return cls.from_dict(data)
        
    def to_dict(self) -> dict:
        
        data = {
            # The arguments for the constructor
            'elements': self.elements,
            'split_epoch_step': self.split_epoch_step,
            'epochs_warmup': self.epochs_warmup,
            'min_cluster_size': self.min_cluster_size,
            'projection_factor': self.projection_factor,
            # These values represent the state of the 
            'tree': deepcopy(self.tree)
        }
        return data
        
    @classmethod
    def from_dict(cls, data: dict) -> t.Any:
        
        obj = cls(**data)
        
        # One pre-processing step we need to do is to fix the datatypes of the tree, specifically the "variable" 
        # field of each node is meant to be a tf.Variable object but for the dict export is is being converted 
        # into just a list of numbers.
        for node_index, node in data['tree'].items():
            # When loading from JSON all the keys which are meant to be int are actually converted to str! which we fix here
            node_index = int(node_index)
            
            if not isinstance(node['variable'], tf.Variable): 
                node['variable'] = tf.Variable(np.array(node['variable']), trainable=True)
                
            # At the end we can add the node to the tree of the actual instantiated object
            obj.tree[node_index] = node
        
        # After the object is constructed we can call collect_leaves to make 
        # sure that all the other secondary instance fields such as the leaf node list are updated correspondingly based 
        # on that new tree.
        obj.collect_leaves()
        
        return obj
        
    # -- ATTRIBUTE ACCESS --
        
    def set_model(self, model: DeepEctMixin):
        
        if isinstance(model, DeepEctMixin):
            self.model = model
            self.inputs = self.model.convert_elements(self.elements)
        
    @property
    def num_leaves(self):
        return len(self.leafs)
        
    def iterate_leafs(self):
        for node_index in self.leafs:
            yield node_index, self.tree[node_index]
        
    # -- TREE MANAGEMENT --
        
    def initialize(self):
        self.sample_batch()
        
        # Clearing the tree at the beginning here so that the initialize method can be used to re-initialize an object 
        # back to the base state as well.
        self.tree = {}
        
        # We also want to make sure that we provide an initial set of latent embeddings
        self.embeddings = self.embedd_elements(self.elements)
        
        # ~ inserting the first leaf node
        # At the very beginning I need to add a single leaf node that will include where all the given elements as 
        # members of that one cluster.
        node_index = self.insert_node(self.batch_indices)
        
        # This method will actually collects all the tensorflow variables from the tree into the variables list!
        self.collect_leaves(node_index)
        
        self.inputs = self.model.convert_elements(self.elements)
        
        self.logger.info(f'initialized the tree with {len(self.tree)} nodes and {len(self.leafs)}')
        
    def insert_node(self, indices: t.List[int]):
        # members = [self.embeddings[i] for i in indices] 
        # centroid = self.centroid_from_members(members)
        members = [self.batch_embeddings[i] for i in indices]
        centroid = np.mean(members, axis=0)
        
        node_index = self.next_index
        self.tree[node_index] = {
            'position':     centroid,
            'variable':     tf.Variable(centroid, trainable=True),
            'indices':      indices,
            'weight':       len(indices),
            'children':     None,
            'parent':       None,
            'sibling':      None,
        }
        # After inserting a node into the tree we need to update the counter for the next index so that the next node 
        # that will be inserted has a unique index.
        self.next_index += 1
        
        self.num_nodes.assign_add(1)
        
        return node_index
        
    def collect_leaves(self, node_index: int = 0, clear: bool = True):
        # First we clear all the current tensorflow variables from the collection so that we can then afterwards iterate 
        # through the entire tree to collect all the new ones from the leaf nodes.
        if clear:
            self.leafs = []
            self.splits = []
        
        node_data = self.tree[node_index]
        node = self.tree[node_index]
        
        # ~ leaf nodes
        # If the special children key is None then that indicates that this node is a leaf node of the tree which means 
        # that it will actually be associated with a variable that we can put into the new list
        if node_data['children'] == None:
            
            self.leafs.append(node_index)
            
            node['position'] = node['variable'].numpy()
        
        # ~ split nodes
        # If its not None that means its a split node in which case we recursively explore the tree further along the 
        # branches of the split.
        else:
            
            self.splits.append(node_index)
            
            # Here we recursively call the function itself to make sure that the children nodes of this split node 
            # are definitley already collected so that we can extract the necessary information from them afterwards.
            children = node_data['children']
            for child_index in children:
                self.collect_leaves(child_index, clear=False)
            
            # After the children have been collected we want to calculate the virtual location of the split node from 
            # those two children, which is simply in the middle of them (weighted average)
            child_0, child_1 = [self.tree[i] for i in children]
            
            indices = child_0['indices'] + child_1['indices']
            node['indices'] = indices
            
            # The position of the weight
            self.tree[node_index]['position'] = (
                child_0['position'] * child_0['weight'] +
                child_1['position'] * child_1['weight']
            ) / (child_0['weight'] + child_1['weight'])
            
        # This is the formula from the paper: We calculate the new weight of each leaf node as the average of the 
        # of the previous batch's weight and the new number of elements assigned to this leaf.
        node['weight'] = 0.5 * node['weight'] + 0.5 * len(node['indices'])
                
    def split(self):
        self.embeddings = self.embedd_elements(self.elements)
        
        # We iterate through all leaf nodes and calculate the split metric for all of them and the one with the 
        # hightest metric we split.
        split_values: t.List[float] = [self.get_split_value(node_index) for node_index in self.leafs]
        node_index = self.leafs[np.argmax(split_values)]
        
        success = self.split_node(node_index)
        if not success:
            self.leaf_blacklist[node_index] = self.callback.epoch
            self.logger.info(f'rejected split of leaf node {node_index}')
        else:
            self.logger.info(f'split leaf node {node_index}')
        
        self.collect_leaves()
        
        return node_index
        
    def split_node(self, node_index: int, force: bool = False) -> bool:
        node = self.tree[node_index]
        indices = node['indices']
        embeddings = [self.batch_embeddings[i] for i in indices]
        
        # try:
        #     gm_single = GaussianMixture(n_components=1, max_iter=300, covariance_type='diag')
        #     gm_single.fit(embeddings)
            
        #     gm = GaussianMixture(n_components=2, max_iter=300, covariance_type='diag')
        #     gm.fit(embeddings)
        # except ValueError:
        #     return False
        # labels = gm.predict(embeddings) 
        
        kmeans = KMeans(
            n_clusters=2
        )
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        print('\n')    
        
        # ~ cluster member check
        # As the first check to see if we accept a certain cluster splitting or not is by looking at the 
        # number of cluster members. If the clusters are severely unbalanced aka if one of the clusters 
        # would get more than X% of the members then we reject the splitting
        cluster_counts = [np.sum((labels == k).astype(int)) for k in [0, 1]]
        print('COUNTS', cluster_counts)
        for count in cluster_counts:
            if (count < 0.1 * len(indices) or count < self.min_cluster_size) and not force:
                return False
        
        # ~ variance reduction check
        # Another criterion is how much a potential splitting reduces the variance compared to leaving the 
        # cluster as it is. We say that we only accept a splitting here if the average variance is reduced 
        # by at a factor of Y
        
        # print(gm_single.score(embeddings))
        # node_score = gm_single.score(embeddings)

        # for k in [0, 1]:
        #     embeddings_k = np.array([emb for emb, l in zip(embeddings, labels) if l == k])
        #     score_k = gm.score(embeddings_k)
        #     print(gm.score(embeddings_k))
        #     if score_k < node_score and not force:
        #         return False
        
        # ~ performing the splitting

        node['children'] = []
        
        for k in [0, 1]:
            
            child_indices = [i for i, l in zip(indices, labels) if l == k]
            child_index = self.insert_node(child_indices)
            self.tree[child_index]['parent'] = node_index
            node['children'].append(child_index)
            
        # We also need to set the sibling references!
        self.tree[node['children'][0]]['sibling'] = node['children'][1]
        self.tree[node['children'][1]]['sibling'] = node['children'][0]
        
        return True
                
    def get_split_value(self, node_index: int):
        
        if node_index in self.leaf_blacklist:
            return -1
        
        # This is the list of embeddings for the cluster with the given node_index
        embeddings = [self.batch_embeddings[i] for i in self.tree[node_index]['indices']]
        
        # Now we need to calculate the sum of square distances towards the center of that cluster
        center = self.tree[node_index]['variable'].numpy()
        
        distances = np.array([euclidean(center, emb) for emb in embeddings])
        distances = distances
        
        return np.sum(distances)
                
    def sample_batch(self):
        
        self.batch_elements = random.sample(self.elements, k=self.batch_size)
        # sample_indices = random.sample(self.indices, k=self.batch_size)
        # if isinstance(self.inputs, (list, tuple)):
        #     self.batch_inputs = [tf.gather(inp, sample_indices) for inp in self.inputs]
        # else:
        #     self.batch_inputs = tf.gather(self.inputs, sample_indices)    
        self.batch_inputs = self.model.convert_elements(self.batch_elements)
        self.batch_embeddings = self.model.embedd(self.batch_inputs)
        self.batch_indices = list(range(len(self.batch_embeddings)))
        
    def assign_nodes(self):
        
        # First of all we need to clear the lists that still contain the previous batch's assignments of the 
        # leaf nodes
        for node_index in self.leafs:
            self.tree[node_index]['indices'] = []
        
        # Then the assignment scheme is very simple: We assign every embedding of this batch to the closest 
        # leaf node. For that we compute all the pairwise distances of all the embeddings with the leaf centers
        # and then select the minimum one for each.
        leaf_centers = [self.tree[i]['variable'].numpy() for i in self.leafs]
        distances = pairwise_distances(self.batch_embeddings, leaf_centers, metric='euclidean')
        closest = np.argmin(distances, axis=1)
        
        for index, closest in enumerate(closest):
            node_index = self.leafs[closest]
            self.tree[node_index]['indices'].append(index)
                
    # -- CALLBACK METHODS --
                
    def on_train_batch_end(self, step, *args, **kwargs):
        
        if self.callback.epoch >= self.epochs_warmup:
            
            self.sample_batch() # could be improved
        
            self.assign_nodes()
            
            self.collect_leaves()
            
            self.train_step_projection() # ! bottleneck

            self.train_step_node_centers() # cheap
        
    def on_epoch_end(self, *args, **kwargs):
        
        if self.callback.epoch >= self.epochs_warmup and self.callback.epoch % self.split_epoch_step == 0:
            node_index = self.split()
            
            for node_index in list(self.leaf_blacklist.keys()):
                epoch = self.leaf_blacklist[node_index]
                if self.callback.epoch >= epoch + 3 * self.split_epoch_step:
                    del self.leaf_blacklist[node_index]
        
        if self.save_history:
            
            self.logger.info(f'recording ect history at epoch {self.callback.epoch}') 
            
            self.batch_embeddings = self.embedd_elements(self.batch_elements)
            self.history[self.callback.epoch] = [
                {
                    'index':        node_index,
                    'embeddings':   [self.batch_embeddings[i] for i in node['indices']],
                    'center':       node['variable'].numpy(),
                    'weight':       node['weight'],
                }
                for node_index in self.leafs
                if (node := self.tree[node_index])
            ]
    
    def train_step_node_centers(self):
        
        with tf.GradientTape() as tape:
            loss = 0.0
            
            # pass all the elements through the model
            # x = tf.constant(self.elements)
            # embeddings = self.model.embedd(self.inputs)
            embeddings = self.batch_embeddings
            
            for node_index in self.leafs:
                
                node = self.tree[node_index]
                leaf_center = node['variable']
                
                if len(node['indices']) == 0:
                    continue
                
                # print(node['indices'])
                leaf_embeddings = tf.gather(embeddings, node['indices'])
                leaf_embeddings_mean = tf.reduce_mean(leaf_embeddings, axis=0)
            
                loss += tf_euclidean_distance(leaf_center, leaf_embeddings_mean) / self.num_leaves
    
        variables = [self.tree[i]['variable'] for i in self.leafs]
        gradients = tape.gradient(loss, variables)

        for var, grad in zip(variables, gradients):
            if grad is not None:
                var.assign_add(-0.01 * grad)
        
        return loss
    
    def train_step_projection(self):
        
        loss = 0.0
        if len(self.tree) > 1:
            
            with tf.GradientTape() as tape:
                loss = tf.constant(0.0)
                
                # pass all the elements through the model
                self.batch_embeddings = self.model.embedd(self.batch_inputs)
                embeddings = self.batch_embeddings
                
                for node_index in range(1, self.next_index):
                    
                    node = self.tree[node_index]
                    sibling = self.tree[node['sibling']]
                    
                    if len(node['indices']) == 0:
                        continue
                    
                    # rho: (D, )
                    # node_center: (D, )
                    #rho = node['position'] - sibling['position']
                    rho = sibling['position'] - node['position']
                    rho = rho / np.linalg.norm(rho)
                    rho = tf.constant(rho)
                    rho = tf.expand_dims(rho, axis=-1)
                    
                    node_center = node['position']
                    
                    # node_embeddings: ([B], D)
                    node_embeddings = tf.gather(embeddings, node['indices'])
                    
                    loss_contribs = tf.abs(tf.matmul(node_center - node_embeddings, rho))
                    loss += tf.reduce_mean(loss_contribs) / len(self.tree)
        
                loss *= self.projection_factor
        
                trainable_vars = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                
                # This list comprehension here is a bit messy and one might ask why it is even necessary. What it essentially 
                # does is it filters out all the gradients that are None. None gradients can happen here because in this 
                # train step we are effectively only training the encoder part of the network and not at all the output head
                # We need to filter these None gradients here because they would evoke tf warning messages that would absolutely 
                # flood the console...
                self.model.optimizer.apply_gradients([
                    (grad, var) 
                    for (grad, var) in zip(gradients, trainable_vars) 
                    if grad is not None
                ])
        
        return loss
    
    def fit(self, x, y, **kwargs):
        
        # We need to make sure to inject our custom callback into the training process here
        if 'callbacks' in kwargs:
            kwargs['callbacks'].append(self.callback)
        else:
            kwargs['callbacks'] = [self.callback]
            
        self.model.fit(x, y, **kwargs) 
    
    def predict_embeddings(self, embeddings: list):
        
        leaf_centers = [self.tree[i]['variable'].numpy() for i in self.leafs]
        distances = pairwise_distances(embeddings, leaf_centers, metric='euclidean')
        closest = np.argmin(distances, axis=1)
        
        labels = []
        for index, closest in enumerate(closest):
            node_index = self.leafs[closest]
            labels.append(node_index)
        
        return labels
    
    def predict_elements(self, elements: list):
    
        embeddings = self.embedd_elements(elements)
        labels = self.predict_embeddings(embeddings)
        
        return labels
    
    # -- Utility
    
    def get_cluster_labels(self):
        labels = []
        for index in self.indices:
            for node_index in self.leafs:
                if index in self.tree[node_index]['indices']:
                    labels.append(node_index)
                    break
                
        return labels
        
    def centroid_from_members(self, members: t.List[t.Any]):
        # embeddings: (B, D)
        embeddings = self.embedd_elements(members)
        centroid = np.mean(embeddings, axis=0)
        return centroid
    
    def embedd_elements(self, elements: t.List[t.Any]):
        # return self.model.embedd_vectors(elements)
        return self.model.embedd_elements(elements)