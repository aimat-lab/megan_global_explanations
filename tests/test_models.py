import os
import tempfile

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from megan_global_explanations.models import DensePredictor
from megan_global_explanations.models import GraphPredictor
from megan_global_explanations.models import load_model

from .util import load_mock_vgd

# -- GraphPredictor --

def test_graph_predictor_basically_works():
    
    index_data_map = load_mock_vgd()
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    
    model = GraphPredictor(
        units=[10, 10, 10], 
        embedding_units=[10, 10],
        final_units=[10, 10],
        final_activation='linear'
    )
    pred = model.predict_graphs(graphs)
    
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (51, 10)
    
    
def test_graph_predictor_saving_loading_works():
    
    index_data_map = load_mock_vgd()
    graphs = [data['metadata']['graph'] for data in index_data_map.values()]
    
    model = GraphPredictor(
        units=[10, 10, 10], 
        embedding_units=[10, 10],
        final_units=[10, 10],
        final_activation='linear'
    )
    pred = model.predict_graphs(graphs)
    
    with tempfile.TemporaryDirectory() as path:
        
        model.save(path)
        
        model_loaded = load_model(path)
        pred_loaded = model_loaded.predict_graphs(graphs)
        
        assert np.isclose(pred, pred_loaded, rtol=1e-4).all()
    

# -- DensePredictor --

def test_dense_predictor_basically_works():
    
    x = np.random.random(size=(100, 10))
    
    model = DensePredictor(
        units=[10, 10, 10],
        final_units=[10, 2]
    )
    y_pred = model(x)
    predictions = model.predict_vectors(x)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (100, 2)
    

def test_dense_predictor_saving_loading_works():
    
    x = np.random.random(size=(100, 10))
    model = DensePredictor(
        units=[10, 10, 10],
        final_units=[10, 2]
    )
    # We need to invoke the model with data once to actually build the layers
    y_pred = model.predict_vectors(x)
    
    with tempfile.TemporaryDirectory() as path:
        model.save(path)
        
        model_loaded = load_model(path)
        y_pred_loaded = model_loaded.predict_vectors(x)
        
        assert np.isclose(y_pred, y_pred_loaded).all()
        