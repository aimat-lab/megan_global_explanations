"""
This script illustrates how to save and load the concept information to and from a folder 
representation on the disk. This is done using one of the already saved concept assets as 
an example.
"""
import os
import typing as t

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.torch.megan import Megan

from megan_global_explanations.utils import EXPERIMENT_PATH
from megan_global_explanations.data import ConceptReader
from megan_global_explanations.data import ConceptWriter

# ~ loading concept information
# The persistent representation of concept information is intentionally lightweight. The information 
# about a concept clustering is stored as a folder, which in turn contains subfolders for every concept. 
# These subfolders then contain metadata about the concept cluster and optionally about the concept 
# prototype graphs.
# The metadata information however only specifies the indices of the graph elements which make up that 
# concept. That means that the basic graph structure needs to be fetched from the orginal dataset.
# The concept information also does not include the model-specific information such as the explanation 
# subgraph embeddings. This information will have to be obtained from the model as part of the 
# loading process of the concept information.
# Therefore the loading process for the concept clustering information depends on a dataset and a model 
# which have to be passed to the constructor of the Reader object.

CONFIG = Config().load()
MODEL_PATH: str = os.path.join(EXPERIMENT_PATH, 'assets', 'models', 'rb_dual_motifs.ckpt')
DATASET_PATH: str = ensure_dataset('rb_dual_motifs')
CONCEPTS_PATH: str = os.path.join(EXPERIMENT_PATH, 'assets', 'concepts', 'rb_dual_motifs')

dataset_reader = VisualGraphDatasetReader(path=DATASET_PATH)
index_data_map = dataset_reader.read()
processing = dataset_reader.read_processing()

model = Megan.load_from_checkpoint(MODEL_PATH)

reader = ConceptReader(
    path=CONCEPTS_PATH,
    dataset=index_data_map,
    model=model,
)
# The "read" method will return the list of concept dictionaries, where each dictionary contains the full
# information about a single concept cluster.
concepts: t.List[dict] = reader.read()
print(f'loaded {len(concepts)} concepts.')
print(f'concept keys: {list(concepts[0].keys())}')

# ~ saving concept information
# 

writer = ConceptWriter(
    ConceptWriter.create_experiment_folder(
        EXPERIMENT_PATH,
        'assets',
        'concepts',
    ),
)