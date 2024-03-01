"""
Extends the base experiment "vgd_concept_extraction". This experiment implements the concept extraction 
specifically for the aqsoldb dataset for the regression of logS water solubility values.
"""
import os
import pathlib
import random
import traceback
import typing as t
from copy import deepcopy

import numpy as np
import visual_graph_datasets.typing as tv
import rdkit.Chem as Chem
from scipy.spatial.distance import cosine
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.graph import copy_graph_dict
from graph_attention_student.torch.megan import Megan
from megan_global_explanations.gpt import query_gpt
from megan_global_explanations.gpt import describe_molecule
from megan_global_explanations.prototype.optimize import genetic_optimize
from megan_global_explanations.prototype.optimize import embedding_distances_fitness_mse
from megan_global_explanations.prototype.molecules import mutate_remove_atom
from megan_global_explanations.prototype.molecules import mutate_remove_bond
from megan_global_explanations.utils import EXPERIMENTS_PATH


PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(EXPERIMENTS_PATH, 'assets')

# == DATASET PARAMETERS ==
# The parameters determine the details related to the dataset that should be used as the basis 
# of the concept extraction

# :param VISUAL_GRAPH_DATASETS:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either 
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this 
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote 
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'aqsoldb'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or 
#       "classification"
DATASET_TYPE: str = 'regression'
# :param CHANNEL_INFOS:
#       This dictionary can optionally be given to supply additional information about the individual 
#       explanation channels. The key should be the index of the channel and the value should again be 
#       a dictionary that contains the information for the corresponding channel.
CHANNEL_INFOS: t.Dict[int, dict] = {
    0: {
        'name': 'negative',
        'color': 'lightskyblue',
    },
    1: {
        'name': 'positive',
        'color': 'lightcoral',
    }
}

# == MODEL PARAMETERS == 
# These parameters determine the details related to the model that should be used for the 
# concept extraction. For this experiment, the model should already be trained and only 
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the 
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'aqsoldb.ckpt')


# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a 
#       fidelity higher than this will be used as possible candidates for the clustering.
FIDELITY_THRESHOLD: float = 0.5
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially 
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
MIN_CLUSTER_SIZE: int = 10
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the 
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while 
#       lower values tend to result in more clusters.
MIN_SAMPLES: int = 10

# == PROTOTYPE OPTIMIZATION PARAMETERS ==
# These parameters configure the process of optimizing the cluster prototype representatation

# :param OPTIMIZE_CLUSTER_PROTOTYPE:
#       This boolean flag determines whether the prototype optimization should be executed at 
#       all or not. If this is False, the entire optimization routine will be skipped during the 
#       cluster discovery.
OPTIMIZE_CLUSTER_PROTOTYPE: bool = True
# :param INITIAL_POPULATION_SAMPLE:
#       This integer number determines the number of initial samples that are drawn from the cluster 
#       members as the initial population of the prototype optimization GA procedure.
INITIAL_POPULATION_SAMPLE: int = 200
# :param OPTIMIZE_PROTOTYPE_POPSIZE:
#       This integer number determines the population size of the genetic optimization algorithm
#       that is used to optimize the prototype representation.
OPTIMIZE_PROTOTYPE_POPSIZE: int = 1000
# :param OPTIMIZE_PROTOTYPE_EPOCHS:
#       This integer number determines the number of epochs that the genetic optimization algorithm
#       will be executed for the prototype optimization.
OPTIMIZE_PROTOTYPE_EPOCHS: int = 50
# :param OPENAI_KEY:
#       This string value has to be the OpenAI API key that should be used for the GPT-4 requests
#       that will be needed to generate the natural language descriptions of the prototypes.
OPENAI_KEY: str = os.getenv('OPENAI_KEY')
DESCRIBE_PROTOTYPE: bool = False
# :param HYPOTHESIZE_PROTOTYPE:
#       This boolean flag determines whether the prototype hypothesis should be generated at all
#       or not. If this is False, the entire hypothesis routine will be skipped during the
#       cluster discovery.
HYPOTHESIZE_PROTOTYPE: bool = True
# :param CONTRIBUTION_THRESHOLDS:
#       This dictionary determines the thresholds to be used when converting the contribution values 
#       of classification tasks into the strings such that they can be passed to the language model 
#       for the hypothesis generation. The keys are the contribution values and the values are the 
#       strings that will be used to describe the impact of these contributions in words.
#       Note that this will only be used for classification problems since for classification problems 
#       the contribution values are measured in classification logits which do not have a direct meaning 
#       to the language model. In contrast, regression contributions are measured directly in the 
#       target space and therefore do not need to be converted.

# == VISUALIZATION PARAMETERS ==
# These parameters determine the details of the visualizations that will be created as part of the 
# artifacts of this experiment.

# :param PLOT_UMAP:
#       This boolean flag determines whether the UMAP visualization of the graph embeddings should be
#       created or not. If this is True, the UMAP visualization will be created and saved as an additional 
#       artifact of the experiment.
PLOT_UMAP: bool = True


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
                       processing: ProcessingBase,
                       cluster_graphs: t.List[tv.GraphDict],
                       cluster_embeddings: np.ndarray,
                       **kwargs,
                       ) -> dict:
    
    # For the embedding objective function we need some kind of anchor location to which we want to 
    # minimize the distance. For this we are simply going to use the cluster centroid.
    anchor = np.mean(cluster_embeddings, axis=0)
    
    # In this section we assemble the initial population for the optimization of the prototype. In fact, 
    # in this use case, it is possible to already assemble a very good initial population by simply using 
    # the subgraphs which are already highlighted by the explanation masks of the cluster members. These 
    # should by themselves already be very close to the ideal cluster prototype and most likely only 
    # need minor refinements through the GA optimization.
    num_initial = min(e.INITIAL_POPULATION_SAMPLE, len(cluster_embeddings))
    
    anchor_distances = np.array([cosine(anchor, emb) for emb in cluster_embeddings])
    indices = np.argsort(anchor_distances).tolist()[:num_initial]
    
    violation_radius = np.percentile(anchor_distances, 90)
    print('MIN', anchor_distances[indices][:3], 'MEAN', np.mean(anchor_distances), 'MAX', np.max(anchor_distances), 'VIOL', violation_radius)
    
    elements_initial = []
    for index in indices:
        graph = copy_graph_dict(cluster_graphs[index])
        smiles = graph['graph_repr'].item()

        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)
        
        # We actually HAVE TO clear those here or else it will fail. This is because the molecular mutation operations 
        # do not preserve the additional graph attributes and it will cause issues to pass graphs through the model 
        # where some of them have this attribute and some of them dont.
        del graph['node_importances']
        del graph['edge_importances']

        elements_initial.append({
            'graph': graph,
            'value': smiles,
        })
    
    # This function will execute the actual genetic optimization algorithm to create graph prototypes
    # that are as close to the cluster centroid (==anchor) as possible.
    element, history = genetic_optimize(
        fitness_func=lambda elements: embedding_distances_fitness_mse(
            elements=elements,
            model=model,
            channel_index=channel_index,
            anchors=[anchor],
            violation_radius=0.2,
        ),
        # fitness_func=lambda graphs: graph_matching_embedding_fitness(
        #     graphs=graphs,
        #     model=model,
        #     channel_index=channel_index,
        #     anchor_graphs=anchor_graphs,
        #     processing=processing, 
        #     check_edges=True,
        # ),
        sample_func=lambda: deepcopy(random.choice(elements_initial)),
        mutation_funcs=[
            lambda element: mutate_remove_bond(element, processing=processing),
            lambda element: mutate_remove_atom(element, processing=processing),
            lambda element: mutate_remove_atom(mutate_remove_bond(element, processing=processing), processing=processing),
            lambda element: mutate_remove_atom(mutate_remove_atom(element, processing=processing), processing=processing),
        ],
        num_epochs=10,
        population_size=2000,
        logger=e.logger,
        elite_ratio=0.01,
        refresh_ratio=0.1,
    )
    
    return element


@experiment.hook('describe_prototype', default=False, replace=True)
def describe_prototype(e: Experiment,
                       value: str,
                       image_path: str,
                       ) -> str:
    
    print(value)
    e.log(' * generating description for the molecular prototype...')
    description, _ = describe_molecule(
        api_key=e.OPENAI_KEY,
        smiles=value,
        image_path=image_path,
        max_tokens=200,
    )
    print(description)
    
    return description


@experiment.hook('prototype_hypothesis', default=False, replace=True)
def prototype_hypothesis(e: Experiment,
                         value: str,
                         image_path: str,
                         channel_index: int,
                         contribution: float,
                         **kwargs,
                         ) -> t.Optional[str]:
    
    e.log(' * generating mutagenicity prototype hypothesis with GPT...')
    
    system_message = (
        f'You are a chemistry expert that is tasked to come up with possible hypotheses about the underlying '
        f'structure-property relationships of molecular properties.\n\n'
        f'The property in question is "Water Solubility" - meaning a molecules tendency to dissolve in water. \n\n'
        'You will be presented with a molecular substructure / fragment in SMILES representation and with some '
        'empirical evidence linked to that substructure. For your answer follow a structure that starts with an '
        'explanation of the hypothesized reason for the identified structure-property relationship, followed summary '
        'of the presented evidence and the hypothesis like this: \n\n'
        'Detailed Explanation: [Elaboration of the causal reasoning for  the suggested structure property relationship\n\n'
        'Hypothesis: [One sentence describing the structure and linked property. Two sentences about the hypothesized causal explanations.]\n\n'
        '- you do not use markdown syntax\n'
        '- you do not use enumerations\n'
        '- you language is accurate and concise\n'
    )
    
    if e.DATASET_TYPE == 'regression':
        impact: str = f'{contribution:.2f}'
        name = 'Water Solubility'
        
    elif e.DATASET_TYPE == 'classification':
        name: str = e.CHANNEL_INFOS[channel_index]["name"]
        impact: str = ''
        for threshold, description in e.CONTRIBUTION_THRESHOLDS.items():
            if contribution > threshold:
                impact = description.upper()
    
    user_message = (
        f'The structure given by the SMILES representation "{value}" has been linked to {impact} influence '
        f'towards "{name}"'
    )
    
    try:
        description, messages = query_gpt(
            api_key=e.OPENAI_KEY,
            system_message=system_message,
            user_message=user_message,
        )
        print(description)
        return description
    except Exception as exc:
        print(exc)
        traceback.print_exc()
        return None

experiment.run_if_main()