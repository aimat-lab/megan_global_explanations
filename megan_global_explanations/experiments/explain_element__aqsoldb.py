"""
This is the specific sub-experiment which performs the element query and explanation for the aqsoldb
dataset using the corresponding model and concept clustering. Consequently, the domain representation 
for the query element in this experiment will be a SMILES string.
"""
import os
import pathlib

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == QUERY PARAMETERS ==
# The query parameters are the parameters that are used to query the model.

# :param QUERY_ELEMENT:
#       The string domain representation of the element with which to query the model. 
#       For this element the explanations will be created.
#QUERY_ELEMENT: str = 'C1(Cl)=C(Cl)C(Cl)=CC=C1CCCN'
QUERY_ELEMENT: str = 'C1=CC(=C(C(=C1CCN)Cl)Cl)Cl'
# :param QUERY_TYPE:
#       This string determines whether the given prediction task is a regression or a classification
#       problem, which in turn determines the details of the experiment such as how certain values 
#       will have to be calculates or what kinds of visualizations will be created.
QUERY_TYPE: str = 'regression' # 'classification'

# == DATASET PARAMETERS ==
# The dataset parameters are the parameters that are used to load the dataset. 

# :param VISUAL_GRAPH_DATASET:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'aqsoldb'
# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'aqsoldb.ckpt')
# :param CONCEPTS_PATH:
#       This has to be the absolute string path to the concept clustering data that is to be used for the
#       concept clustering. This data is typically created by the concept clustering process and then
#       stored on the disk as a folder.
CONCEPTS_PATH: str = os.path.join(ASSETS_PATH, 'concepts', 'aqsoldb')

# == VISUALIZATION PARAMETERS
# These parameters determine the details for the visualization of the results.

# :param NUM_EXAMPLES:
#       The number of examples to be shown in the visualization. This is the number of examples that will be
#       shown for each concept.
NUM_EXAMPLES: int = 6
# :param EXAMPLE_MODE:
#       This determines the mode in which the examples are selected. This may either be "centroid" or "random".
#       If "centroid" is selected, the examples will be selected based on the distance to the centroid of the
#       concept. If "random" is selected, the examples will be selected randomly from the concept.
EXAMPLE_MODE: str = 'centroid'  # 'random'


experiment = Experiment.extend(
    'explain_element.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()