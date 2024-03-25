import os
import pathlib
import typing as t

import click
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem.AllChem
from rdkit import Chem
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import visual_graph_datasets.typing as tc
# processing
from visual_graph_datasets.processing.base import *
from visual_graph_datasets.processing.colors import *
from visual_graph_datasets.processing.molecules import *
# visualization
from visual_graph_datasets.visualization.base import *
from visual_graph_datasets.visualization.colors import *
from visual_graph_datasets.visualization.molecules import *

PATH = pathlib.Path(__file__).parent.absolute()

# -- custom imports --
"""
One way in which this class will be used is by copying its entire source code into a separate
python module, which will then be shipped with each visual graph dataset as a standalone input
processing functionality.

All the code of a class can easily be extracted and copied into a template using the "inspect"
module, but it may need to use external imports which are not present in the template by default.
This is the reason for this method.

Within this method all necessary imports for the class to work properly should be defined. The code
in this method will then be extracted and added to the top of the templated module in the imports
section.
"""
pass
# --


# -- The following class was dynamically inserted --
class CustomColorProcessing(ColorProcessing):
    pass

# --


# The data element pre-processing capabilities defined in the above class can either be accessed by
# importing this object from this module in other code and using the implementations of the methods
# "process", "visualize" and "create".
# Alternatively this module also acts as a command line tool (see below)
processing = CustomColorProcessing()

if __name__ == '__main__':
    # This class inherits from "click.MultiCommand" which means that it will directly work as a cli
    # entry point when using the __call__ method such as here. This will enable this python module to
    # expose the cli commands defined in the above class when invoking it from the command line.
    processing()
