import os
import pathlib

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')


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
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'aqsoldb2.ckpt')
# :param CONCEPTS_PATH:
#       This has to be the absolute string path to the concept clustering data that is to be used for the
#       concept clustering. This data is typically created by the concept clustering process and then
#       stored on the disk as a folder.
CONCEPTS_PATH: str = os.path.join(ASSETS_PATH, 'concepts', 'aqsoldb')

# == TRAINING PARAMETERS ==
# These parameters determine the details for the training of the simple interpretable proxy model.

# :param NUM_TEST:
#       The number of test examples to be used for the training of the proxy model. This is the number of examples
#       that will be used to evaluate the performance of the proxy model.
NUM_TEST: int = 1000


# == EXPERIMENT PARAMETERS ==
# The parameters for the experiment.

__DEBUG__ = True

experiment = Experiment.extend(
    'linear_concept_approximation.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()