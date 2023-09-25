"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

PATH = pathlib.Path(__file__).parent.absolute()

experiment = Experiment.extend(
    'template_experiment.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('hook')
def hook(e, parameter):
    e.log(f'parameter: {parameter}')


@experiment.analysis
def analysis(e):
    e.log('more analysis...')


experiment.run_if_main()
