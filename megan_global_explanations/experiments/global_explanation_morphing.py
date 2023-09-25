"""
This experiment assumes that there already exists a working global concept clustering result consisting of a 
fully trained MEGAN model and the HDBSCAN clusterer object that defines all the concept clusters.

Based on this existing result, the possibility of *global explanation morphing* is being explored. The idea here 
is that it should be possible to transform a given input graph such that it "uses" a different concept explanation.
For a given input graph we can use the model to generate the embeddings and then use the clusterer to determine which 
of the existing concept clusters is the most likely explanation for that given input sample. We can now make edits to 
this graph with the objective changing this cluster assignment to be a different one.
"""
import os
import joblib
import pathlib

import numpy as np
from hdbscan import HDBSCAN
from hdbscan.prediction import membership_vector
from pycomex.functional.experiment import Experiment
from pycomex.util import folder_path, file_namespace
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.processing import BaseProcessing

from megan_global_explanations.models import load_model
from megan_global_explanations.models import EctMegan

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')


# == MODEL PARAMETERS ==

MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'mutagenicity')

PROCESSING_PATH: str = os.path.join(ASSETS_PATH, 'models', 'mutagenicity', 'process.py')

HDBSCAN_PATH: str = os.path.join(ASSETS_PATH, 'models', 'mutagenicity', 'hdbscan.joblib')

# == INPUT PARAMETERS ==

# :param BASE_VALUE:
#       This is the domain specific value 
BASE_VALUE: str = 'C1=CC=CC=C1CCCO'

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')

    module = dynamic_import(e.PROCESSING_PATH)
    processing: BaseProcessing = module.processing

    e.log('loading model...')
    model: EctMegan = load_model(e.MODEL_PATH)
    
    e.log('loading hdbscan...')
    hdbscan = joblib.load(e.HDBSCAN_PATH)

    # ~ THE BASE VALUE
    # At first
    e.log('predicting for the base value...')
    base_graph = processing.process(e.BASE_VALUE)
    base_embeddings = model.embedd_graphs([base_graph])[0]

    # cluster_probabilities: (1, R) 
    cluster_probabilities = membership_vector(hdbscan, [base_embeddings])[0]
    
    print('soft cluster memberships')
    print(cluster_probabilities)
    print('top 3 clusters')
    print(np.argsort(-cluster_probabilities)[:3])
    
    