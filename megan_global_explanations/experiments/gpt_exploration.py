"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import openai

import numpy as np
import matplotlib.pyplot as plt

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True

OPENAI_KEY = ''

SYSTEM_MESSAGE = """
You are helpful teacher that explains property influenced of various strictural patterns for property prediction tasks on graph 
input data.

The current task is a graph regresssion task on color graphs. In color graphs each node is associated with an RGB color code and 
those nodes are connected by undirected edges. Each graph is associated with a continuous target value in the range from -3 to +3 
which are influenced by the existence of special sub-graph motifs.

User requests fpr explanations will be formated as desctibed within the delimiters. Users will provide information about the 

===

FIDELITY: This is a numeric value that determines the 

GLOBAL NAME: The name of the substructure to be explained. In your explanations you shall only refer to the structure by this name.

STRUCTURE 

INFLUENCE: The average effect of the given structure 



===

You produce explanations that are consistent in style and format.
"""

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    openai.api_key = OPENAI_KEY

    openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages={
            {'role': 'system', 'content': SYSTEM_MESSAGE},
            {'role': 'user', ''}
        }
    )

    e.apply_hook('hook', parameter=1)


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()