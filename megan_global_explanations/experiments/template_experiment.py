"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')

    e.apply_hook('hook', parameter=1)


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()