import os
import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.base import GraphBaseLayer
from graph_attention_student.layers import MultiHeadGATV2Layer, DenseEmbedding
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.models.megan import Megan, shifted_sigmoid
from graph_attention_student.training import NoLoss

from megan_global_explanations.deep_ect import DeepEctMixin
from megan_global_explanations.pack import ClusterPackingMixin


class EctMegan(Megan, DeepEctMixin, ClusterPackingMixin):
    
    def __init__(self,
                 # message passing related
                 units: t.List[int],
                 activation: str = "swish",
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 use_edge_features: bool = True,
                 # node/edge importance related
                 importance_units: t.List[int] = [],
                 importance_channels: int = 2,
                 importance_activation: str = "sigmoid",  # do not change
                 importance_factor: float = 0.0,
                 importance_multiplier: float = 10.0,
                 sparsity_factor: float = 0.0,
                 concat_heads: bool = False,
                 # mlp tail end related 
                 final_units: t.List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 final_pooling: str = 'sum',
                 final_bias: t.Optional[list] = None,
                 regression_weights: t.Optional[t.Tuple[float, float]] = None,
                 regression_reference: t.Optional[float] = None,
                 # fidelity training related
                 fidelity_factor: float = 0.0,
                 fidelity_funcs: t.List[t.Callable] = [],
                 # constrastive representation learning related
                 embedding_units: t.Optional[t.List[int]] = None,
                 contrastive_sampling_factor: float = 0.0,
                 contrastive_sampling_tau: float = 0.9,
                 positive_sampling_rate: int = 1,
                 positive_sampling_noise_attributes: float = 0.2,
                 positive_sampling_noise_importances: float = 0.2,
                 **kwargs,
                 ):
                
        super(EctMegan, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            use_edge_features=use_edge_features,
            importance_units=importance_units,
            importance_channels=importance_channels,
            importance_activation=importance_activation,
            importance_factor=importance_factor,
            importance_multiplier=importance_multiplier,
            sparsity_factor=sparsity_factor,
            concat_heads=concat_heads,
            final_units=final_units,
            final_dropout_rate=final_dropout_rate,
            final_activation=final_activation,
            final_pooling=final_pooling,
            regression_weights=regression_weights,
            regression_reference=regression_reference,
            **kwargs
        )
        # mlp backend
        self.final_bias = final_bias
        # fidelity training
        self.fidelity_factor = fidelity_factor
        self.fidelity_funcs = fidelity_funcs
        # contrastive representation learning
        self.embedding_units = embedding_units
        self.contrastive_sampling_factor = contrastive_sampling_factor
        self.contrastive_sampling_tau = contrastive_sampling_tau
        self.positive_sampling_rate = positive_sampling_rate
        self.positive_sampling_noise_attributes = positive_sampling_noise_attributes
        self.positive_sampling_noise_importances = positive_sampling_noise_importances
        
        self.var_contrastive_sampling_factor = tf.Variable(
            tf.constant(self.contrastive_sampling_factor), 
            dtype=tf.float32, 
            trainable=False
        )
        
        # ~ modifying the attention layers
        self.attention_activations = [activation for _ in self.units]
        self.attention_layers: t.List[GraphBaseLayer] = []
        for u, act in zip(self.units, self.attention_activations):
            lay = MultiHeadGATV2Layer(
                units=u,
                activation=act,
                num_heads=self.importance_channels,
                use_bias=self.use_bias,
                concat_heads=self.concat_heads,
                has_self_loops=True,
                use_edge_features=True,
                # modified - fixes the issue that the GATv2 attention message passing does not take into 
                # consideration each node's own features!
                concat_self=True,
            )
            self.attention_layers.append(lay)
        
        # ~ adding projection network for the graph embeddings
        
        if embedding_units is None:
            self.embedding_units = [self.units[-1], self.units[-1]]
        
        self.channel_dense_layers = []
        for _ in range(self.importance_channels):
            
            layers = []

            embedding_acts = ['swish' for _ in self.embedding_units]
            embedding_acts[-1] = 'tanh'
            
            embedding_biases = [True for _ in self.embedding_units]
            embedding_biases[-1] = False
            
            for u, act, bias in zip(self.embedding_units, embedding_acts, embedding_biases):
                layers.append(DenseEmbedding(
                    units=u,
                    activation=act,
                    use_bias=bias,
                ))
            
            self.channel_dense_layers.append(layers)
            
        # ~ contrsative learning related
        self.x_support: t.Optional[tuple] = None
        self.graph_embeddings_support = None
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.step_delay: int = 2
        self.num_anchors: int = 5
        
    def get_config(self):
        config = super(EctMegan, self).get_config()
        config.update(**{
            'fidelity_factor': self.fidelity_factor,
            'fidelity_funcs': self.fidelity_funcs,
            'embedding_units': self.embedding_units,
        })
        
        return config
    
    @property
    def graph_embedding_shape(self) -> t.Tuple[int, int]:
        """
        Returns a tuple which defines contains the information about the shape of the graph embeddings (K, D)
        where K is the number of explanation channels employed in the model and D is the number of elements in 
        each of the embedding vectors for each of the explanation channels.
        
        Note that every explanation channel produces it's own graph embeddings!
        
        :returns: int
        """
        return self.importance_channels, self.embedding_units[-1]
        
    def embedd(self,
               inputs: list,
               training: bool = True,
               squeeze: bool = True,
               node_importances_mask: t.Optional[tf.Tensor] = None,
               ) -> tf.Tensor:
        
        # node_input: ([B], [V], N)
        # edge_input: ([B], [E], M)
        # edge_index_input: ([B], [E], 2)
        node_input, edge_input, edge_index_input = inputs
        
        # ~ MESSAGE PASSING / ATTENTION LAYERS
        # The first part of the network consists of a message passing part or more specifically a number of 
        # graph attention layers. These attention layers receive the graph and the node embeddings as input 
        # and return a transformed vector of node embeddings which is in turn used as the input of the next 
        # attention layer.
        # The important part in this step is that these special attention layers also return the attention 
        # logits as a byproduct. Those are multiple values per *edge* which we nedd to keep track of for 
        # every layer so that they can be processed afterwards.
                
        node_embedding = node_input
        alphas: t.List[tf.RaggedTensor] = []
        for lay in self.attention_layers:
            # node_embedding: ([B], [V], N_l)
            # alpha: ([B], [E], K, 1)
            # The alpha values are the attention *logits* for each edge of each of the graphs along all the 
            # attention heads, which is equal to the number of K explanation channels here as defined in the 
            # constructor!
            node_embedding, alpha = lay([node_embedding, edge_input, edge_index_input])
            alphas.append(alpha)
            
            if training:
                node_embedding = self.lay_dropout(node_embedding)
        
        # ~ EDGE IMPORTANCES
        # In this section we proceed to create the edge importance explanations from the attention logits we 
        # have just collected. We achieve the correct shape by aggregating over all the tensors collected 
        # from the different layers.
        
        # alphas: ([B], [E], K, L)
        alphas = tf.concat(alphas, axis=-1)
        edge_importances = tf.reduce_sum(alphas, axis=-1)
        # edge_importances: ([B], [E], K)
        edge_importances = self.lay_act_importance(edge_importances)
        
        # Now we need to perform a local pooling that will broadcast these edge values into a node shape
        # such that we can use it as part of the node explanations
        pooled_edges_in = self.lay_pool_edges_in([node_input, edge_importances, edge_index_input])
        pooled_edges_out = self.lay_pool_edges_out([node_input, edge_importances, edge_index_input])
        # pooled_edges: ([B], [V], K)
        pooled_edges = self.lay_average([pooled_edges_in, pooled_edges_out])
        
        # ~ NODE IMPORTANCES
        # In this section we assmeble the node importances. We will need this fully assembled tensor of 
        # node importances to use those as the weights for the final global weighted pooling operation that 
        # turns the node embeddings into the graph embeddings.
        # The node importances consist of two parts which are being multiplied with each other. 
        # (1) the first part we have already created - that is the pooled edge importances
        # (2) the second part is created by using the node embeddings of the message passing part 
        # as the basis of a special dense network which will create a node tensor of correct shape
        
        node_importances_tilde = node_embedding
        for lay in self.node_importance_layers:
            node_importances_tilde = lay(node_importances_tilde)
            
        node_importances_tilde = self.lay_act_importance(node_importances_tilde)
            
        # node_importances_tilde: ([B], [V], K)
        # node_importances: ([B], [V], K)
        node_importances = pooled_edges * node_importances_tilde
    
        # ~ EXPLANATION AUGEMENTATIONS
        # In this section we will be applying various augmentations to the explanations we have just 
        # created. This includes for example regularization
        
        # Sparsity regularization is essentially just L1 regularization, which will provide a constant 
        # small gradient driving all of the explanation weights to become zero. This will effectively 
        # only make the unimportant weights zero, as the important ones have stronger gradients acting 
        # on them as well which will promote != 0 values.
        if self.sparsity_factor > 0:
            # Now here one could question why we are applying to separately on the edge importances and 
            # tne partial node importances instead of just on the final assembled node importances since 
            # that is just a connection of those two anyways.
            # The answer to this is that experiments showed that this work better, don't know why.
            # self.lay_sparsity(tf.reduce_sum(node_importances_tilde, axis=-1))
            # self.lay_sparsity(tf.reduce_sum(edge_importances, axis=-1))
            self.lay_sparsity(node_importances_tilde)
            self.lay_sparsity(edge_importances)
        
        # Optionally we will apply an additional external mask to the already existing values that can be 
        # used to suppress certain parts of these explanations.
        # This is a core part of the multi-channel fidelity computation. The channel-specific fidelity is 
        # essentially just the deviation of the networks output prediction in case one specific channel 
        # is supporessed from entering the final prediction MLP.
        if node_importances_mask is not None:
            node_importances_mask = tf.cast(node_importances_mask, tf.float32)
            node_importances *= node_importances_mask
               
        # In this first section we perform the pooling operation. For each channel we do the weighted 
        # pooling and then the overall graph embedding vector is assembled as a concatenation of the 
        # individual embeddings.
        graph_embeddings: t.List[tf.RaggedTensor] = []
        for k in range(self.importance_channels):
            # We select the appropriate slice of the node importances for each of the channels 
            # and use that as multiplication weights for the node embeddings
            # node_importances_slice: ([B], [V], 1)
            # graph_embedding: ([B], [V], N_L)
            node_importances_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)
            graph_embedding = self.lay_pool_out(node_embedding * node_importances_slice)
            # for lay in self.embedding_layers:
            #     graph_embedding = lay(graph_embedding)
                
            for lay in self.channel_dense_layers[k]:
                graph_embedding = lay(graph_embedding)
                
            # graph_embedding = self.lay_embedding_dense(graph_embedding)
            
            # graph_embedding = ([B], D)
            graph_embeddings.append(graph_embedding)
        
        if squeeze:
            # graph_embeddings_squeezed: ([B * K], D)
            graph_embeddings_squeezed = tf.concat(graph_embeddings, axis=0)
            return graph_embeddings_squeezed
        else:
            return graph_embeddings, node_importances, edge_importances
        
    def call(self,
             inputs: tuple,
             training: bool = True,
             return_importances: bool = True,
             return_embeddings: bool = False,
             node_importances_mask: t.Optional[tf.RaggedTensor] = None,
             edge_importances_mask: t.Optional[tf.RaggedTensor] = None,
             **kwargs) -> tuple:
        """
        Implements the forwards pass of the model.
        
        Roughly speaking the forward pass consists of three main parts. 
        (1) The first part is a graph message passing part consisting of attention layers. 
        It produces the final node embeddings and a bunch of edge attention logits. 
        (2) The second part of the model assembles the edge attention logits and the node embeddings 
        into the edge and node explanation masks, which are also called "importances" tensors.
        (3) The third part of the model performs a global pooling operation which turns the node 
        embeddings into graph embeddings and then uses those as the basis for a dense prediction network
        
        The return signature of this method depends on the flags set in the arguments. in the default state, this method 
        will return a tuple of 3 tensors: The actual prediction output tensor, the node importances mask tensor and the 
        edge importances mask tensor.
        """
        
        # node_input: ([B], [V], N)
        # edge_input: ([B], [E], M)
        # edge_index_input: ([B], [E], 2)
        node_input, edge_input, edge_index_input = inputs
            
        # graph_embeddings: list_K ([B], D)
        graph_embeddings, node_importances, edge_importances = self.embedd(
            inputs=inputs,
            training=training,
            node_importances_mask=node_importances_mask,
            squeeze=False,
        )
         # graph_embeddings_separate: ([B], K, D)
        graph_embeddings_separate = tf.concat([tf.expand_dims(emb, axis=-2) for emb in graph_embeddings], axis=-2)
        # graph_embeddings: ([B], N_L * K)
        graph_embeddings = tf.concat(graph_embeddings, axis=-1)
        
        # Appyling all the layers of the final prediction MLP
        output = graph_embeddings
        for lay in self.final_layers:
            output = lay(output)
            
        # self.add_loss(0.1 * tf.reduce_mean(tf.abs(output)))
        output = self.lay_final_activation(output)
            
        if return_embeddings:
            return output, node_importances, edge_importances, graph_embeddings_separate
        if return_importances:
            return output, node_importances, edge_importances
        else:
            return output
        
    def train_step_fidelity(self, x):
        """
        This is an additional train step that can be applied to the model, which implements a forward 
        pass of sorts, the computation of the gradients and the model weight update.
        
        This train step implements "fidelity training" where the fidelity property of the model itself 
        is being optimized.
        """
        
        with tf.GradientTape() as tape:
            fid_loss = 0.0
            
            # out_pred: ([B], C)
            out_pred, ni_pred, ei_pred = self(x, training=True)
           
            # helper data structures from which the node importance masks will later be assembled
            ones = tf.reduce_mean(tf.ones_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)
            zeros = tf.reduce_mean(tf.zeros_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)
            
            deviations = []
            for k, func in enumerate(self.fidelity_funcs):
                # Here we create a leave-one-out channel mask which will mask exactly the current channel k 
                # of the loop such that no information about that channel enters the final prediction network
                mask = [ones if i == k else zeros for i in range(self.importance_channels)]
                # mask: ([B], [V], K)
                mask = tf.concat(mask, axis=-1)
                # By applying this mask during another forward pass we can calculate the modified output prediction 
                # vector.
                # out_mod: ([B], C)
                out_mod, _, _ = self(x, training=True, node_importances_mask=mask)
                
                # Now for each channel it's is the users responsibility to pass in an appropriate function externally 
                # which will calculate the appropriate loss value for the difference of the original and the modified 
                # output vectors in the spirit of the multi-channel fidelity computation.
                # This is accomplished with arbitrary functions here instead of a hard-coded one, because the specific 
                # function to be applied to the prediction difference here depends on the mode of the network as well 
                # as the specific goal for the target fidelity distributions to be achieved in each case.
                diff = func(out_pred, out_mod)
                fid_loss += tf.reduce_mean(diff)
                
                # deviation: ([B], 1, C)
                deviation = tf.expand_dims(out_pred - out_mod, axis=-2)
                deviations.append(deviation)
                
            fid_loss *= self.fidelity_factor
            
            # deviations: ([B], K, C)
            deviations = tf.concat(deviations, axis=-2)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(fid_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return fid_loss, deviations
            
    
    def train_step(self, data):
        """
        This method will be called to perform each train step during the model training process, which is 
        executed once for every training batch. This method will do model forward passes within an 
        automatic differentiation environment, generate the loss gradients and perform model weight update.
        
        This is a customized train step function, which currently implements the following two training 
        objectives:
        
        1. The normal supervised predition loss. This primarily concerns the primary task predictions 
           but it is optionally also possible to train the node and edge explanation masks with some given 
           ground truth explanation masks!
        2. The second loss is the optional exlanation approximation loss. 
           This is only applied when importance_factor > 0. This additional loss tries to approximate the 
           primary task prediction by just using each channel's explanation masks in a specific way to promote 
           the several channels to produce explanations consistent with the pre-determined interpretations.
           
        Additions for v2 of the model:
        
        3. Constrastive Representation Learning. This is a method from the domain of unsupervised learning
           which aims to improve the properties of latent embedding spaces. In case of this model, this additional 
           loss term will promote clustering behavior within the latent space of graph explanation embeddings 
           which can then ultimately be interpreted as the global explanations.
        
        4. Fidelity Training. This is an additional training step which is applied separately after the main 
           training step of the model. This train step will promote the fidelity distributions of each channel 
           to behave according to some externally supplied target distribution properties with the goal of 
           reducing the number of samples with negative fidelity (where the effect on the network is not aligned 
           to the intended interpretation of that channel.)
        
        :returns: the metrics dictionary
        """
        # This is standard code to make it compatible to the default tensorflow training process
        if len(data) == 3:
            sample_weight = None
            x, y, clusters = data
        else:
            sample_weight = None
            x, y = data

        node_attributes, edge_attributes, edge_indices = x
        
        # ~ FIDELITY TRAINING
        # This section implements the fidelity training. The purpose of the fidelity training is to refine the 
        # multi-channel fidelity behavior of the model. It has been observed that while the default MEGAN model *generally* 
        # produces explanations that are faithful to their assigned interpretations according to the Fidelity* metric, 
        # there are quite some samples where this is not the case.
        # The aim of the fidelity training is to directly use the fidelity formula as an optimization objective during 
        # training to reduce the number of samples which are unfaithul to their channel's interpretation.
        
        # One might wonder why the fidelity training is implemented as a separate training step instead of a direct 
        # loss term such as all the other training modifications. The honest answer is that this has been tested out 
        # and I found it to work much better when it is a separate step.
        fid_loss = 0
        if self.fidelity_factor != 0:
            # This method will execute an entirely separate training step 
            fid_loss, deviations = self.train_step_fidelity(x)

        # Forward pass auto differentiation
        exp_metrics = {'exp_loss': 0}
        with tf.GradientTape() as tape:
            exp_loss = 0

            node_input, edge_input, edge_indices = x[:3]
            out_true, ni_true, ei_true = y

            # out_pred: ([B], C)
            # ni_pred: ([B], [V], K)
            # ei_pred: ([B], [E], K)
            # graph_embeddings: ([B], N, K)
            out_pred, ni_pred, ei_pred, graph_embeddings = self(x, training=True, return_importances=True, return_embeddings=True)
            
            # ~ PREDICTION LOSS
            # The following section implements the normal prediction loss that is determined by the tensorflow 
            # prediction function. Essentially in the fit() call of the model one has to define three separate 
            # loss functions for training the output, node_importance and edge_importances with their 
            # corresponding ground truth labels respectively.
            loss = self.compiled_loss(
                [out_true, ni_true, ei_true],
                [out_pred, ni_pred, ei_pred],
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            # ~ APPROX. EXPLANATION LOSS
            # The basic motivation for the explanation loss is that by default there is no method that assures that 
            # the designated importance channels actually produce explanations that are conceptionally consistent 
            # with the interpretations that we assign them.
            # (How does one channel know we want it to only represent negative evidence while the other is positive?)
            # Thus this explanation loss attempts to solve the primary task prediction performance using only the 
            # explanation channels as an approximation. 
            if self.importance_factor != 0:
                
                # First of all we need to assemble the approximated model output, which is simply calculated
                # by applying a global pooling operation on the corresponding slice of the node importances.
                # So for each slice (each importance channel) we get a single value, which we then
                # concatenate into an output vector with K dimensions.
                outs_approx: t.List[tf.Tensor] = []
                for k in range(self.importance_channels):
                    node_importances_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)
                    out = self.lay_pool_out(node_importances_slice)

                    outs_approx.append(out)

                # outs: ([batch], K)
                outs_approx = tf.concat(outs_approx, axis=-1)

                # How this approximation works in detail has to be different for regression and classification
                # problems since for regression problems we make the linearized assumption of positive and negative 
                # evidence for a single regression value, while for classification we need exactly one 
                # explanation per class
                if self.doing_regression:
                    # This method will return an augmented version of the true target lables such that this can be 
                    # directly trained with the given approximated output.
                    # The mask separates the training samples into the positive and negative ones for both the channels!
                    outs_regress, mask = self.regression_augmentation(out_true)
                    
                    # So we essentially try to solve a regression problem using the pooled explanation masks
                    # But split into the "positive" and "negative" parts of the current training batch with respect 
                    # to a given "reference" target value.
                    exp_loss = self.compiled_regression_loss(
                        outs_regress * mask,
                        outs_approx * mask,
                    )

                else:
                    outs_class = shifted_sigmoid(
                        outs_approx,
                        shift=self.importance_multiplier,
                        multiplier=1
                    ) * tf.cast(out_true, tf.float32)
                    exp_loss = self.compiled_classification_loss(out_true, outs_class)

                loss += self.importance_factor * exp_loss
                
            # graph_embeddings: ([B] * N, K)
            embeddings = self.embedd(x, training=True)
            pack_loss = self.get_packing_loss(embeddings, clusters)
            loss += pack_loss
                    
        # The rest of this is the standard keras train step code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(
            y,
            out_pred,
            sample_weight=sample_weight
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            **{m.name: m.result() for m in self.metrics},
            'loss': loss,
            'exp_loss': exp_loss,
            'fid_loss': fid_loss,
            'pack_loss': pack_loss,
        }
    
    # -- IMPLEMENT DeepEctMixin --
    
    def convert_elements(self, elements):
        return tensors_from_graphs(elements)


# == GRAPH MODELS (GRAPH DATA) == 

#class GraphPredictor(ks.models.Model, DeepEctMixin, ClusterPackingMixin):
class GraphPredictor(ks.models.Model, ClusterPackingMixin):
    
    def __init__(self,
                 units: t.List[int],
                 embedding_units: t.List[int],
                 final_units: t.List[int],
                 final_activation: str = 'linear',
                 embedding_activation: str = 'tanh',
                 num_heads: int = 2,
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        ClusterPackingMixin.__init__(self)
        
        self.units = units
        self.embedding_units = embedding_units
        self.final_units = final_units
        self.final_activation = final_activation
        self.num_heads = num_heads
        
        # ~ the message passing part of the network
        self.message_layers = []
        for k in self.units:
            lay = MultiHeadGATV2Layer(
                units=k,
                activation='swish',
                num_heads=num_heads,
                concat_heads=False,
                concat_self=True,
            )
            self.message_layers.append(lay)
        
        self.lay_pooling = PoolingNodes(pooling_method='sum')
        
        # ~ the projection network for the latent space
        self.embedding_layers = []
        self.embedding_activations = ['swish' for _ in self.embedding_units]
        self.embedding_activations[-1] = embedding_activation
        self.embedding_biases = [True for _ in self.embedding_units]
        self.embedding_biases[-1] = False
        for k, act, bias in zip(self.embedding_units, self.embedding_activations, self.embedding_biases):
            lay = DenseEmbedding(
                units=k,
                activation=act,
                use_bias=bias,
            )
            self.embedding_layers.append(lay)
            
        # ~ the final prediction part of the network    
        self.final_layers = []
        self.final_activations = ['relu' for _ in self.final_units]
        self.final_activations[-1] = self.final_activation
        for k, act in zip(self.final_units, self.final_activations):
            lay = DenseEmbedding(
                units=k,
                activation=act,
                use_bias=True,
            )
            self.final_layers.append(lay)
            
    def get_config(self) -> dict:
        config = {
            'units': self.units,
            'embedding_units': self.embedding_units,
            'final_units': self.final_units,
            'final_activation': self.final_activation,
            'num_heads': self.num_heads,
        }
        return config
        
    def embedd(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['inputs']
            
        node_input, edge_input, edge_index_input = inputs
        
        node_embeddings = node_input
        for lay in self.message_layers:
            node_embeddings, _ = lay([node_embeddings, edge_input, edge_index_input])
            
        graph_embeddings = self.lay_pooling(node_embeddings)
        for lay in self.embedding_layers:
            graph_embeddings = lay(graph_embeddings)
            
        return graph_embeddings
    
    def call(self, inputs):
        
        graph_embeddings = self.embedd(inputs)
        
        output = graph_embeddings
        for lay in self.final_layers:
            output = lay(output)
            
        # print(output.shape)
        return output
    
    def embedd_graphs(self, graphs):
        x = tensors_from_graphs(graphs)
        graph_embeddings = self.embedd(x)
        return graph_embeddings.numpy()
    
    def predict_graphs(self, graphs):
        x = tensors_from_graphs(graphs)
        out = self(x)
        return out.numpy()
    
    # -- IMPLEMENT DeepEctMixin --
    
    def convert_elements(self, elements: list):
        return tensors_from_graphs(elements)
    
    def train_step(self, data):
        
        if len(data) == 3:
            x, y, clusters = data
            
        else:
            x, y = data
            
        with tf.GradientTape() as tape:
            loss = 0.0
            
            # Forward pass
            y_pred = self(x, training=True)  # Call the model to get predictions
            embeddings = self.embedd(x)
            loss = self.compiled_loss(y, y_pred)  # Compute the loss
            
            loss_packing = self.get_packing_loss(embeddings, clusters)
            loss += loss_packing
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update model weights using the optimizer
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (if you have defined any)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dictionary of metrics to be displayed during training
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics.update({
            'loss_packing': loss_packing
        })
        return metrics
    

# == DENSE MODELS (TABULAR DATA) ==

class DensePredictor(ks.models.Model, DeepEctMixin):
    
    def __init__(self,
                 units: t.List[int],
                 final_units: t.List[int],
                 final_activation: str = 'softmax',
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        self.units = units
        self.final_units = final_units
        self.final_activation = final_activation
        
        # ~ The initial encoder network
        self.encoder_layers = []
        
        self.encoder_activations = ['relu' for _ in self.units]
        self.encoder_activations[-1] = 'linear'
        
        self.encoder_biases = [True for _ in self.units]
        self.encoder_biases[-1] = False
        
        for k, act, bias in zip(self.units, self.encoder_activations, self.encoder_biases):
            lay = ks.layers.Dense(
                units=k,
                activation=act,
                use_bias=bias
            )
            self.encoder_layers.append(lay)
        
        # ~ The final prediction network
        self.final_layers = []
        
        self.final_activations = ['relu' for _ in self.final_units]
        self.final_activations[-1] = self.final_activation
        
        for k, act in zip(self.final_units, self.final_activations):
            lay = ks.layers.Dense(
                units=k,
                activation=act,
                use_bias=True,
            )
            self.final_layers.append(lay)
        
    def get_config(self):
        config = {
            'units': self.units,
            'final_units': self.final_units,
        }
        return config

    def embedd(self, inputs):
        # inputs: ([B], N)
        embeddings = inputs
        for lay in self.encoder_layers:
            embeddings = lay(embeddings)
            
        # embeddings. ([B], D)
        return embeddings
        
    def call(self, 
             inputs, 
             training: bool = False,
             return_embeddings: bool = False):
        # inputs: ([B], N)
        # embeddings. ([B], D)
        embeddings = self.embedd()
            
        output = embeddings
        for lay in self.final_layers:
            output = lay(output)
            
        if return_embeddings:
            return output, embeddings
        else:
            return output

    def predict_vectors(self, vectors: list):
        vectors = np.array(vectors)
        pred = self(vectors)
        return pred.numpy()
    
    def embedd_vectors(self, vectors: list):
        vectors = np.array(vectors)
        emb = self.embedd(vectors)
        return emb.numpy()
    
    # -- IMPLEMENT DeepEctMixin --
    
    def convert_elements(self, elements: list) -> tf.Tensor:
        return tf.constant(np.array(elements))
        
        
class DenseAutoencoder(ks.models.Model):
    
    def __init__(self,
                 encoder_units: t.List[int],
                 decoder_units: t.List[int],
                 final_activation: str = 'linear',
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.final_activation = final_activation
        
        # ~ The initial encoder network
        self.encoder_layers = []
        
        self.encoder_activations = ['relu' for _ in self.encoder_units]
        self.encoder_activations[-1] = 'tanh'
        
        self.encoder_biases = [True for _ in self.encoder_units]
        self.encoder_biases[-1] = True
        
        for k, act, bias in zip(self.encoder_units, self.encoder_activations, self.encoder_biases):
            lay = ks.layers.Dense(
                units=k,
                activation=act,
                use_bias=bias
            )
            self.encoder_layers.append(lay)
        
        # ~ The decoder network
        self.decoder_layers = []
        
        self.decoder_activations = ['relu' for _ in self.decoder_units]
        self.decoder_activations[-1] = self.final_activation
        
        self.decoder_biases = [True for _ in self.decoder_units]
        self.decoder_biases[-1] = False
        
        for k, act, bias in zip(self.decoder_units, self.decoder_activations, self.decoder_biases):
            lay = ks.layers.Dense(
                units=k,
                activation=act,
                use_bias=bias
            )
            self.decoder_layers.append(lay)
        
    def embedd(self, inputs):
        # inputs: ([B], N)
        
        embeddings = inputs
        for lay in self.encoder_layers:
            embeddings = lay(embeddings)
            
        return embeddings
        
    def call(self, 
             inputs, 
             training: bool = False,
             return_embeddings: bool = False):
        # inputs: ([B], N)
        embeddings = self.embedd(inputs)
        # embeddings. ([B], D)
        
        output = embeddings
        for lay in self.decoder_layers:
            output = lay(output)
            
        if return_embeddings:
            return output, embeddings
        else:
            return output

    def predict_vectors(self, vectors: list):
        vectors = np.array(vectors)
        pred = self(vectors)
        return pred.numpy()
    
    def embedd_vectors(self, vectors: list):
        vectors = np.array(vectors)
        _, emb = self(vectors, return_embeddings=True)
        return emb.numpy()
    
    
CUSTOM_OBJECTS = {
    'EctMegan': EctMegan,
    'GraphPredictor': GraphPredictor,
    'DensePredictor': DensePredictor,
    'DenseAutoencoder': DenseAutoencoder,
    'NoLoss': NoLoss,
}


def load_model(path: str):
    with ks.utils.custom_object_scope(CUSTOM_OBJECTS):
        return ks.models.load_model(path)