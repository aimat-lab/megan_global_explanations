import os
import pytest

from megan_global_explanations.gpt import query_gpt
from megan_global_explanations.gpt import describe_molecule
from megan_global_explanations.gpt import describe_color_graph
from .util import ASSETS_PATH
from .util import ARTIFACTS_PATH
from .util import OPENAI_KEY


def test_query_gpt_basically_works():
    """
    The query_gpt function is a generic function that wraps the functionality to send a query to ghe GPT API.
    """
    content, messages = query_gpt(
        api_key=OPENAI_KEY,
        system_message='',
        user_message='This is ',
    )
    assert isinstance(content, str)
    assert content != ''


def test_describe_color_graph_basically_works():
    """
    The function describe_color_graph should be able to generate a description for a given color graph image.
    """
    image_path = os.path.join(ASSETS_PATH, 'test_color_graph.png')
    description, messages = describe_color_graph(
        api_key=OPENAI_KEY,
        image_path=image_path,
    )
    assert isinstance(description, str)
    assert description != ''
    
    # Now we also want to save the description as a test artifact in a text file
    output_path = os.path.join(ARTIFACTS_PATH, 'test_describe_color_graph_basically_works.txt')
    with open(output_path, 'w') as f:    
        f.write(description)


def test_describe_molecule_smiles_only():
    """
    The function describe_molecule should be able to generate a description for a given molecule given 
    only the SMILES representation of the molecule.
    """
    smiles = 'CCC'
    
    description, messages = describe_molecule(
        api_key=OPENAI_KEY,
        smiles=smiles,
    )
    assert isinstance(description, str)
    assert description != ''
    
    # Now we also want to save the description as a test artifact in a text file
    output_path = os.path.join(ARTIFACTS_PATH, 'test_describe_molecule_smiles_only.txt')
    with open(output_path, 'w') as f:
        f.write(description)
    
    
def test_describe_molecule_with_image():
    """
    The function describe_molecule should be able to generate a description for a given molecule given.
    """
    image_path = os.path.join(ASSETS_PATH, 'test_molecule.png')
    smiles = 'C1=CC=C(C=C1)C(=O)O'
    
    description, messages = describe_molecule(
        api_key=OPENAI_KEY,
        smiles=smiles,
        image_path=image_path,
    )
    assert isinstance(description, str)
    assert description != ''
    
    # Now we also want to save the description as a test artifact in a text file
    output_path = os.path.join(ARTIFACTS_PATH, 'test_describe_molecule_with_image.txt')
    with open(output_path, 'w') as f:
        f.write(description)    
    