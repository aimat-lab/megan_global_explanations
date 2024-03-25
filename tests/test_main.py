import os
import pytest
import tempfile

from megan_global_explanations.testing import MockModel
from megan_global_explanations.main import generate_concept_prototypes

from .util import load_mock_clusters
from .util import load_mock_vgd
from .util import load_mock_processing
from .util import LOG


def test_generate_concept_prototypes_basically_works():
    """
    The "generate_concept_prototypes" function is supposed to generate the prototype graphs for a given,
    already existing concept clustering. This test checks if the function can be called without raising
    any exceptions and whether the prototypes are added to the concept data as expected.
    """
    # ~ preparation
    # The "generate_concept_prototypes" function needs quite a lot of different parameters, such as an
    # already existing concept clustering, a model to work with, a dataset instance and a processing 
    # instance, which will all have to be set up for the test.
    embedding_dim = 10
    
    concepts: list[dict] = load_mock_clusters(embedding_dim=embedding_dim)  
    index_data_map: dict = load_mock_vgd()
    processing = load_mock_processing()
    model = MockModel(embedding_dim=embedding_dim)
    
     # ~ testing
    # Now we can check if the prototype generation has worked as intended
    
    with tempfile.TemporaryDirectory() as path:
        generate_concept_prototypes(
            concepts=concepts,
            model=model,
            index_data_map=index_data_map,
            processing=processing,
            mutate_funcs=[lambda element: element],
            path=path,
            logger=LOG,
            num_epochs=1,
            population_size=10,
        )
        
        # The major thing is to check if the concept information has now actually been extended with the 
        # "prototypes" attribute, which is supposed to be a list of dicts where each dict is a visual graph 
        # element dict that represents the prototype graph - including a valid "image_path"!
        for concept in concepts:
            assert 'prototypes' in concept
            assert len(concept['prototypes']) == 1
            
            prototype = concept['prototypes'][0]
            assert 'image_path' in prototype
            # path should exists
            assert os.path.exists(prototype['image_path'])