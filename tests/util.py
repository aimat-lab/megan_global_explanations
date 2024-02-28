import os
import sys
import json
import pathlib
import logging
import typing as t

from visual_graph_datasets.data import VisualGraphDatasetReader
from decouple import config

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')

LOG_TESTING = config('LOG_TESTING', cast=bool, default=True)
LOG = logging.getLogger('Testing')
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.NullHandler())
if LOG_TESTING:
    LOG.addHandler(logging.StreamHandler(sys.stdout))
    
# For some functions we require an OpenAI key to be able to query the OpenAI API, so we load it from the
# environment variables here. Specifically from the .env file that is located in the same folder as this
# module.
OPENAI_KEY = config('OPENAI_KEY', default=None)    

    
def load_mock_clusters() -> t.List[dict]:
    """
    This function loads a cluster data list from the file system. This cluster data list is 
    a mock data structure mainly intended for testing the creation of the concept cluster reports.
    
    :returns: A list of dicts which each contain the information about one cluster such that 
        it can be used to create the cluster report
    """
    folder_path = os.path.join(ASSETS_PATH, 'mock_clusters')
    
    data_path = os.path.join(folder_path, 'data.json')
    with open(data_path) as file:
        content = file.read()
        cluster_data_list = json.loads(content)
        
    # Now we still have to assemble the image paths for each cluster because we can't 
    # save the system-specifics paths statically into that file
    for data in cluster_data_list:
        cluster_index = data['index']
        image_paths = []
        for index, channel_index in data['index_tuples']:
            image_path = os.path.join(folder_path, f'{cluster_index}_{index}.png')
            image_paths.append(image_path)
            
        data['image_paths'] = image_paths
        
    return cluster_data_list


def load_mock_vgd() -> dict:
    path = os.path.join(ASSETS_PATH, 'mock_vgd')
    reader = VisualGraphDatasetReader(path=path)
    index_data_map = reader.read()
    return index_data_map


# Here I want to load the .env file in the same folder as this module and expose all the env variables
# as a dict ENV_VARS

