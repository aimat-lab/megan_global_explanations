from megan_global_explanations.prototype.molecules import MOLECULE_PROCESSING
from megan_global_explanations.prototype.molecules import mutate_remove_bond
from megan_global_explanations.prototype.molecules import mutate_remove_atom


def test_mutate_remove_atom_basically_works():

    # Use a simple aliphatic chain — aromatic rings break when atoms are removed,
    # causing the function to exhaust max_tries and return the original.
    smiles = 'CCCCCC'
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

    result = mutate_remove_atom(result)
    assert isinstance(result, dict)


def test_mutate_remove_bond_basically_works():

    # Use a simple aliphatic chain — removing any C-C bond splits it into two
    # valid fragments, so the mutation always succeeds.
    smiles = 'CCCCCC'
    element = {
        'value': smiles,
        'graph': MOLECULE_PROCESSING.process(smiles),
    }

    result: dict = mutate_remove_bond(
        element
    )
    assert 'value' in result
    assert 'graph' in result

    assert result['value'] != smiles

    result = mutate_remove_bond(result)
    assert isinstance(result, dict)