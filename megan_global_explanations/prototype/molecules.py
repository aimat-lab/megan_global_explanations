import os
import random
import typing as t

from rdkit import Chem
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import mol_from_smiles
from vgd_counterfactuals.generate.molecules import DEFAULT_ATOM_VALENCE_MAP
from vgd_counterfactuals.generate.molecules import get_free_valence_map
from vgd_counterfactuals.generate.molecules import get_valid_atom_additions
from vgd_counterfactuals.generate.molecules import get_valid_bond_additions
from vgd_counterfactuals.generate.molecules import get_valid_bond_removals


MOLECULE_PROCESSING = MoleculeProcessing()

ELEMENT_INFOS: t.List[int] = [
    {
        'symbol':   'C',
        'valence':  4,
        'weight':   10,
    },
    {
        'symbol':   'O',
        'valence':  2,
        'weight':   2,
    },
    {
        'symbol':   'N',
        'valence':  3,
        'weight':   2,
    },
]



def mutate_remove_bond(element: dict,
                       processing: ProcessingBase = MOLECULE_PROCESSING,
                       max_tries: int = 5,
                       ) -> dict:
    
    smiles = element['value']
    mol = Chem.MolFromSmiles(smiles)
    
    # smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)
    # mol = Chem.MolFromSmiles(smiles)
    
    if not mol:
        print(False, smiles)
        return element
    
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    
    for _ in range(max_tries):
        
        bond = random.choice(bonds)
        
        temp_mol = Chem.Mol(mol)
        temp_mol = Chem.EditableMol(temp_mol)
        temp_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        temp_mol = temp_mol.GetMol()
        
        temp_smiles = Chem.MolToSmiles(temp_mol, allBondsExplicit=True, isomericSmiles=False)
        temp_mol = Chem.MolFromSmiles(temp_smiles)
        
        if temp_mol:
            
            if '.' in temp_smiles:
                temp_smiles = random.choice(temp_smiles.split('.'))
                temp_mol = Chem.MolFromSmiles(temp_smiles)
                if not temp_mol or len(temp_mol.GetAtoms()) < 2:
                    continue
                
            try:
                _ = Chem.MolToSmiles(temp_mol, kekuleSmiles=True)
                damaged = False
            except:
                damaged = True
            
            return {
                'value': temp_smiles,
                'graph': processing.process(temp_smiles),
                'damaged': damaged,
            }
        
    return element


def mutate_remove_atom(element: dict,
                       processing: ProcessingBase = MOLECULE_PROCESSING,
                       max_tries: int = 10,
                       ) -> dict:

    smiles = element['value']
    mol = Chem.MolFromSmiles(smiles)
    
    # smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)
    # mol = Chem.MolFromSmiles(smiles)
    
    if not mol:
        print(True, smiles)
        return element
    
    if len(mol.GetAtoms()) < 3:
        return element
    
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    
    for _ in range(max_tries):
        
        atom = random.choice(atoms)
        
        temp_mol = Chem.EditableMol(mol)
        temp_mol.RemoveAtom(atom.GetIdx())
        temp_mol = temp_mol.GetMol()
        
        # for atom in temp_mol.GetAtoms():
        #     atom.SetIsAromatic(False)
        
        temp_smiles = Chem.MolToSmiles(temp_mol, allBondsExplicit=True, isomericSmiles=False)
        temp_mol = Chem.MolFromSmiles(temp_smiles)
        
        if temp_mol:
            
            if '.' in temp_smiles:
                temp_smiles = random.choice(temp_smiles.split('.'))
                temp_mol = Chem.MolFromSmiles(temp_smiles)
                if not temp_mol or len(temp_mol.GetAtoms()) < 2:
                    continue
                
            try:
                _ = Chem.MolToSmiles(temp_mol, kekuleSmiles=True)
                damaged = False
            except:
                damaged = True
            
            return {
                'value': temp_smiles,
                'graph': processing.process(temp_smiles),
                'damaged': damaged,
            }

    return element


def mutate_modify_atom(element: dict,
                       processing: ProcessingBase = MOLECULE_PROCESSING,
                       ) -> dict:
    
    smiles = element['value']
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return element
    
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    
    atom = random.choice(atoms)
    
    replacements: t.List[dict] = []
    for element_info in ELEMENT_INFOS:
        if atom.GetExplicitValence() <= element_info['valence']:
            replacements.append(element_info)

    if not replacements:
        return element
    
    replacement = random.choices(replacements, weights=[info['weight'] for info in replacements])[0]
    
    atom.SetAtomicNum(Chem.Atom(replacement['symbol']).GetAtomicNum())
    
    smiles = Chem.MolToSmiles(mol, allBondsExplicit=True, isomericSmiles=False)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return element
    
    return {
        'value': smiles,
        'graph': processing.process(smiles),
    }


def sample_from_smiles(smiles_list: t.List[str],
                       processing: ProcessingBase = MOLECULE_PROCESSING,
                       ) -> dict:
    smiles = random.choice(smiles_list)
    graph = processing.process(smiles)
    
    return {
        'value': smiles,
        'graph': graph,
    }