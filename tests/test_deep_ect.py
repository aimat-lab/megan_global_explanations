import os
import json
import tempfile

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np

from visual_graph_datasets.data import NumericJsonEncoder
from megan_global_explanations.models import DensePredictor
from megan_global_explanations.deep_ect import DeepEctTrainer
from megan_global_explanations.deep_ect import CustomJsonEncoder


def test_deep_ect_trainer_construction_worlks():
    
    # The minimum additional information needed to construct a trainer object is a number of elements 
    # (in this case vectors) that will be used to representatively map the latent space and the 
    # actual model object to be subject ot the training.
    vectors = np.random.random(size=(100, 10))
    model = DensePredictor(
        units=[10, 2],
        final_units=[10]
    )
    # Then we can construct a minimal trainer object like this
    trainer = DeepEctTrainer(
        model=model,
        elements=vectors,
    )
    
    # After the construction the tree should still be empty
    assert isinstance(trainer, DeepEctTrainer)
    assert len(trainer.tree) == 0
    
    # Only after the construction there should be an initial root node be inserted into the tree!
    trainer.initialize()
    assert len(trainer.tree) == 1
    
    # Now we forcefully (prevents failure of the operation due to the splitting condittion not being met)
    # split the root node into two separate clusters.
    trainer.split_node(0, force=True)
    trainer.collect_leaves()
    
    # After this operation the tree should have 3 nodes in total, 2 of which are leaf nodes and one split 
    # node
    assert len(trainer.tree) == 3
    assert len(trainer.leafs) == 2
    assert len(trainer.splits) == 1
    

def test_deep_ect_trainer_predict_elements_works():
    
    # The minimum additional information needed to construct a trainer object is a number of elements 
    # (in this case vectors) that will be used to representatively map the latent space and the 
    # actual model object to be subject ot the training.
    vectors = np.random.random(size=(100, 10))
    model = DensePredictor(
        units=[10, 2],
        final_units=[10]
    )
    # Then we can construct a minimal trainer object like this
    trainer = DeepEctTrainer(
        model=model,
        elements=vectors,
    )
    
    # Here we initialize the cluster tree and perform an initial splitting so that we have 2 cluster leaf 
    # nodes in the tree.
    trainer.initialize()
    trainer.split_node(0, force=True)
    trainer.collect_leaves()
    
    # The "predict_elements" method is supposed to predict the clustering labels for a given set of elements 
    # this is not necessarily part of the set of elements initiall given to the trainer during construction
    # so here we generate some other vectors and use that method to predict the labels.
    num_all = 1000
    vectors_all = np.random.random(size=(num_all, 10))
    labels_all = trainer.predict_elements(vectors_all)
    
    assert isinstance(labels_all, np.ndarray)
    assert labels_all.shape == (num_all, )
    # We know that the cluster tree has to consist of exactly 2 leaf nodes
    clusters = set(labels_all)
    assert len(clusters) == 2 
    
    
def test_deep_ect_trainer_to_dict_from_dict_works():
    
    # The minimum additional information needed to construct a trainer object is a number of elements 
    # (in this case vectors) that will be used to representatively map the latent space and the 
    # actual model object to be subject ot the training.
    vectors = np.random.random(size=(100, 10))
    model = DensePredictor(
        units=[10, 2],
        final_units=[10]
    )
    # Then we can construct a minimal trainer object like this
    trainer = DeepEctTrainer(
        model=model,
        elements=vectors,
    )
    trainer.initialize()
    
    data = trainer.to_dict()
    
    # Now we test if it is possible to export this dictionary to a json file. This might not be the case 
    # if there are any non primitive data structures contained in it which we obviously want to avoid 
    # since the JSON export is the whole point of this function.
    content = json.dumps(data, cls=CustomJsonEncoder)
    assert isinstance(content, str)
    
    data = json.loads(content)
    trainer_loaded = DeepEctTrainer.from_dict(data)
    assert isinstance(trainer_loaded, DeepEctTrainer)
    
    