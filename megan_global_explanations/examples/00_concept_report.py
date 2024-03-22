"""
This example script will load a visual graph dataset and a pre-trained model and then 
extract the concept explanations by finding dense clusters in the model's latent space 
of subgraph explanations. The resulting concept clustering report PDF will be saved 
in the current working directory.
"""
import os
import sys
import pathlib
import logging
import typing as t

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from graph_attention_student.torch.megan import Megan

from megan_global_explanations.utils import EXPERIMENTS_PATH
from megan_global_explanations.main import extract_concepts
from megan_global_explanations.visualization import create_concept_cluster_report

PATH = pathlib.Path(__file__).parent.absolute()

log = logging.Logger('00')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter())
log.addHandler(handler)

# ~ required parameters

# The "ensure_dataset" method will try to download a dataset from the remote file 
# share server if it is not already present. If the dataset is already present, 
# the local version will be used. In any case, the function will return the absolute 
# string path to the dataset folder. 
DATASET_PATH: str = ensure_dataset('rb_dual_motifs', Config().load())
# Knowing the exact type of task (regression or classification) is important for 
# various operations during the concept clustering and report generation!
DATASET_TYPE: t.Literal['regression', 'classification'] = 'regression'
# We also need to load an existing model, from whose latent space the concept 
# explanations will be extracted.
MODEL_PATH: str = os.path.join(
    EXPERIMENTS_PATH, 
    'assets', 
    'models', 
    'rb_dual_motifs.ckpt'
)
# This is a dictionary that provides additional information about the channels that 
# the model uses.
# However, this dict is optional and does not necessarily have to be provided for the 
# concept clustering to work.
CHANNEL_INFOS: t.Dict[int, dict] = {
    0: {'name': 'negative', 'color': 'skyblue'},
    1: {'name': 'positive', 'color': 'coral'},
}

# ~ loading the dataset
# The dataset is assumed to be in the special "visual graph dataset (VGD)" format. 
# The special "VisualGraphDatasetReader" class will be used to load the dataset. 
# The "read" method will return a dictionary with the dataset elements and their 
# indices as keys.
reader = VisualGraphDatasetReader(path=DATASET_PATH)
index_data_map: t.Dict[int, dict] = reader.read()
processing = reader.read_process()
log.info(f'loaded dataset with {len(index_data_map)} elements.')

# ~ loading the model
# The model is assumed to be a MEGAN model. Therefore the "Megan" class will be 
# used to load the model from the given checkpoint file. The "load_from_checkpoint" 
# method will return the model instance.
model = Megan.load_from_checkpoint(MODEL_PATH)
log.info(f'loaded model {model.__class__.__name__} with {model.num_channels} channels.')

# ~ extracting the concept explanations
# The extract_concepts method will extract the concept explanations by finding 
# dense clusters in the the latent space of the model.
concepts: t.List[dict] = extract_concepts(
    model=model,
    index_data_map=index_data_map,
    processing=processing,
    # parameters for the HDBSCAN clustering algorithm. The smaller the "min_samples" 
    # parameter the more concept clusters will be found. However, this will also lead 
    # to more redundancy - there might be multiple clusters for the same true motif.
    min_samples=60,
    min_cluster_size=10,
    fidelity_threshold=0.5,
    dataset_type=DATASET_TYPE,
    channel_infos=CHANNEL_INFOS,
    # optimization of the cluster prototypes involves more effort.
    optimize_prototypes=False,
    sort_similarity=False,
    logger=log,
)
log.info(f'extracted {len(concepts)} concepts.')

# ~ creating the report
# The "create_concept_report" method will create a report PDF which visualizes 
# all the information from the concept clustering. For every concept several pages 
# with statistics, examples and descriptions will be created.

log.info(f'creating the concept clustering report...')
report_path: str = os.path.join(PATH, 'concept_report.pdf')
create_concept_cluster_report(
    cluster_data_list=concepts,
    path=report_path,
    dataset_type=DATASET_TYPE,
    examples_type='centroid',
    logger=log,
)
log.info(f'report @ {report_path}')