import os
import pytest
import tempfile
import visual_graph_datasets.typing as tv

import numpy as np
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.util import dynamic_import
from megan_global_explanations.utils import safe_int
from megan_global_explanations.testing import create_mock_concepts
from megan_global_explanations.data import ConceptWriter
from megan_global_explanations.data import ConceptReader

from .util import ASSETS_PATH, ARTIFACTS_PATH


@pytest.mark.parametrize('num,dim,prototype_value,num_graphs',[
    (3, 32, None, 0),
    (5, 64, 'R-1RR-1', 5),
])
def test_concept_reader_works(num, dim, prototype_value, num_graphs):
    
    # Before being able to write the concept data to the disk we actually need some concept data to 
    # begin with in the first place so this helper function will create a concept data list with 
    # mock entries.
    concepts = create_mock_concepts(
        num=num,
        dim=dim,
        prototype_value=prototype_value,
        num_graphs=num_graphs
    )
    
    with tempfile.TemporaryDirectory() as tempdir:
        
        # Also before we can read the data, we need to actually write it to the disk first.
        # NOTE: This assumes that the ConceptWriter works as expected!
        processing = ColorProcessing()        
        writer = ConceptWriter(path=tempdir, processing=processing)
        writer.write(concepts)
            
        # Now we can actually test the ConceptReader class by reading the data from the disk
        # which was written in the previous step.
        reader = ConceptReader(path=tempdir)
        concept_data = reader.read()
        assert len(concept_data) == num
        
        for concept, concept_true in zip(concept_data, concepts):
            # check if the centroid np arrays are the same
            assert np.isclose(concept['centroid'], concept_true['centroid']).all()
            
            if prototype_value:
                assert 'prototypes' in concept
            
            if num_graphs > 0:
                assert 'graphs' in concept


@pytest.mark.parametrize('num,dim,prototype_value,num_graphs',[
    (10, 32, None, 0),
    (20, 64, 'R-1RR-1', 10),
])
def test_concept_writer_works(num, dim, prototype_value, num_graphs):
    """
    The ConceptWriter class is used to write concept data to the disk. This test checks if the concept writer
    works as expected using some sample cases and mostly surface level checks for the existence of files.
    """
    # Before being able to write the concept data to the disk we actually need some concept data to 
    # begin with in the first place so this helper function will create a concept data list with 
    # mock entries.
    processing = ColorProcessing()
    concepts = create_mock_concepts(
        num=num,
        dim=dim,
        prototype_value=prototype_value,
        num_graphs=num_graphs,
    )
    
    with tempfile.TemporaryDirectory() as tempdir:
        
        assert len(os.listdir(tempdir)) == 0 
        
        # Setting up the writer instance itself.
        writer = ConceptWriter(
            path=tempdir,
            processing=processing,    
        )
        
        for index, concept in enumerate(concepts):
            writer.write_concept(
                index=index,
                concept=concept,
            )
        
        # at the very least the folder should not be empty anymore
        elements = os.listdir(tempdir)
        assert len(elements) != 0
        # actually there should be as many elements as there are concepts
        assert len(elements) == len(concepts)
        
        # Every concept should have a folder with the index as name
        for index, (element, concept) in enumerate(zip(elements, concepts)):
            assert safe_int(element) is not None
            
            # Now we will check if the concept folder contains the expected files
            concept_folder = os.path.join(tempdir, element)
            print('concept folder', os.listdir(concept_folder))
            assert os.path.isdir(concept_folder)
            
            metadata_path = os.path.join(concept_folder, 'metadata.json')
            assert os.path.exists(metadata_path)
            
            # There should be a separate folder in which both the prototypes and the graphs of the 
            # concept cluster in general are being stored into, which is what is being checked here.
            if prototype_value:
                prototypes_path = os.path.join(concept_folder, 'prototypes')
                print('prototypes path', os.listdir(prototypes_path))
                assert os.path.exists(prototypes_path)
                assert len(os.listdir(prototypes_path)) != 0
            
            if num_graphs > 0:
                graphs_path = os.path.join(concept_folder, 'graphs')
                print('graphs path', os.listdir(graphs_path))
                assert os.path.exists(graphs_path)
                assert len(os.listdir(graphs_path)) != 0
                
        # ~ Saving of the processing
        # At the very end we test the functionality of the writer to save the processing instance 
        # it was constructed with into a standalone module file.
        writer.write_processing()
        
        process_path = os.path.join(tempdir, 'process.py')
        assert os.path.exists(process_path)
        module = dynamic_import(process_path)
        assert isinstance(module.processing, ProcessingBase)
        
        
@pytest.mark.parametrize('num,dim,prototype_value',[
    (10, 32, None),
    (20, 64, 'R-1RR-1'),   
])
def test_create_mock_concepts(num, dim, prototype_value):
    """
    create_mock_concepts is itself a helper function for testing. It is used to create a list of mock concepts.
    this function tests if that creation works.
    """
    concept_data = create_mock_concepts(
        num=num,
        dim=dim,
        prototype_value=prototype_value,
    )
    assert isinstance(concept_data, list)
    assert len(concept_data) == num
    for concept in concept_data:
        assert 'centroid' in concept
        
        if prototype_value:
            assert 'prototypes' in concept
            assert 'metadata' in concept['prototype']
            assert 'image' in concept['prototype']
            tv.assert_graph_dict(concept['prototype']['metadata']['graph'])