from megan_global_explanations.prototype.molecules import MOLECULE_PROCESSING
from megan_global_explanations.prototype.molecules import mutate_remove_bond
from megan_global_explanations.prototype.molecules import mutate_remove_atom


def test_mutate_remove_atom_basically_works():
    
    smiles = 'C1=CC=CC=C1'
    element = {
        'value': smiles,
        'graph': MOLECULE_PROCESSING.process(smiles),
    }
    
    result: dict = mutate_remove_atom(
        element,
    )
    assert 'value' in result
    assert 'graph' in result
    
    print(result['value'])
    assert result['value'] != smiles
    assert len(result['graph']['node_indices']) == 5

    result = mutate_remove_atom(result)
    assert isinstance(result, dict)


def test_mutate_remove_bond_basically_works():
    
    # This is a single carbon ring, so in other words every possible bond removal is 
    # always valid because no removal would lead to a disconnected graph.
    smiles = 'C1=CC=CC=C1'
    element = {
        'value': smiles,
        'graph': MOLECULE_PROCESSING.process(smiles),
    }
    
    result: dict = mutate_remove_bond(
        element
    )
    assert 'value' in result
    assert 'graph' in result
    
    # Since we know that every removal is valid, the result has to be different from the input
    assert result['value'] != smiles
    assert len(result['graph']['node_indices']) == 6
    
    result = mutate_remove_bond(result)
    assert isinstance(result, dict)