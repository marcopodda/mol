from rdkit import Chem


def clean_smiles(smi):
    mol = mol_from_smiles(smi)
    Chem.RemoveStereochemistry(mol)
    smi = mol_to_smiles(mol)
    return Chem.CanonSmiles(smi)


def mol_to_smiles(mol):
    return Chem.MolToSmiles(mol)


def mol_from_smiles(smi):
    return Chem.MolFromSmiles(smi)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]


def mols_from_smiles(smiles):
    return [mol_from_smiles(s) for s in smiles]


def process_smiles(smiles):
    for smi in smiles:
        mol = mol_from_smiles(smi)
        Chem.RemoveStereochemistry(mol)
        smi = mol_to_smiles(mol)
        yield mol, Chem.CanonSmiles(smi)
