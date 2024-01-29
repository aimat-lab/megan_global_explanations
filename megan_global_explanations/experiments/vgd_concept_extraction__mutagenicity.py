import os
import pathlib
import typing as t

import numpy as np
import visual_graph_datasets.typing as tv
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from graph_attention_student.torch.megan import Megan
from visual_graph_datasets.processing.base import ProcessingBase

from megan_global_explanations.prototype.optimize import genetic_optimize
from megan_global_explanations.prototype.optimize import embedding_distance_fitness
from megan_global_explanations.prototype.molecules import sample_from_smiles
from megan_global_explanations.prototype.molecules import mutate_remove_bond
from megan_global_explanations.prototype.molecules import mutate_remove_atom


PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# The parameters determine the details related to the dataset that should be used as the basis 
# of the concept extraction

# :param VISUAL_GRAPH_DATASETS:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either 
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this 
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote 
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'mutag'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or 
#       "classification"
DATASET_TYPE: str = 'classification'
# :param CHANNEL_INFOS:
#       This dictionary can optionally be given to supply additional information about the individual 
#       explanation channels. The key should be the index of the channel and the value should again be 
#       a dictionary that contains the information for the corresponding channel.
CHANNEL_INFOS: t.Dict[int, dict] = {
    0: {
        'name': 'non-mutagenic',
        'color': 'khaki',
    },
    1: {
        'name': 'mutagenic',
        'color': 'violet',
    }
}

# == MODEL PARAMETERS == 
# These parameters determine the details related to the model that should be used for the 
# concept extraction. For this experiment, the model should already be trained and only 
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the 
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'mutagenicity.ckpt')

# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a 
#       fidelity higher than this will be used as possible candidates for the clustering.
FIDELITY_THRESHOLD: float = 1
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially 
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
MIN_CLUSTER_SIZE: int = 20
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the 
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while 
#       lower values tend to result in more clusters.
MIN_SAMPLES: int = 5

# == PROTOTYPE OPTIMIZATION PARAMETERS ==
# These parameters configure the process of optimizing the cluster prototype representatation

# :param OPTIMIZE_CLUSTER_PROTOTYPE:
#       This boolean flag determines whether the prototype optimization should be executed at 
#       all or not. If this is False, the entire optimization routine will be skipped during the 
#       cluster discovery.
OPTIMIZE_CLUSTER_PROTOTYPE: bool = False


__DEBUG__ = True

experiment = Experiment.extend(
    'vgd_concept_extraction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('optimize_prototype', default=False, replace=True)
def optimize_prototype(e: Experiment,
                       model: Megan,
                       channel_index: int,
                       anchor: np.ndarray,
                       processing: ProcessingBase,
                       **kwargs,
                       ) -> tv.GraphDict:
    
    e.log('optimizing molecular prototype')
    element, history = genetic_optimize(
        fitness_func=lambda graphs: embedding_distance_fitness(
            graphs=graphs,
            model=model,
            channel_index=channel_index,
            anchor=anchor,
        ),
        sample_func=lambda: sample_from_smiles([
            'CC',
            'CCC',
            'C1=CC=CC=C1',
            'C1=CC=C2C=CC=CC2=C1',
            'C[N+](=O)[O-]',
            'N=[N+]=N',
            'CC(=O)O',
            'C1C2C=CC=CC2CC3=CC=CC=C31',
            'CN',
            'CO',
            'CCl',
            'C(F)(F)(F)',
            'CBr',
        ], processing=processing),
        mutation_funcs=[
            lambda element: mutate_remove_bond(element, processing=processing),
            lambda element: mutate_remove_atom(element, processing=processing),
        ],
        num_epochs=25,
        population_size=5_000,
        logger=e.logger,
    )
    
    return element

experiment.run_if_main()