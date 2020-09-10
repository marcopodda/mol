import networkx as nx
import torch

from rdkit import Chem
from rdkit.Chem import rdmolops

from core.mols.utils import mol_from_smiles


ATOMS = ["C", "N", "S", "O", "F", "Cl", "Br"]


ATOM_FEATURES = {
    'atomic_num': [6, 7, 8, 9, 16, 17, 35, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83],  # [6, 7, 8, 9, 16, 17, 35],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values())

BOND_FEATURES = {'stereo': [0, 1, 2, 3, 4, 5]}
BOND_FDIM = 13

FINGERPRINT_DIM = 2048


def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def get_atom_features(atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features += onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs'])
    features += onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    features += [1 if atom.GetIsAromatic() else 0]
    return features


def get_bond_features(bond):
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    fbond += onek_encoding_unk(int(bond.GetStereo()), BOND_FEATURES['stereo'])
    return fbond


def mol2nx(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)

    edge_index = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))  # add for reverse edge

    G = nx.DiGraph()
    G.add_edges_from(edge_index)

    for bond in mol.GetBonds():
        bond_features = get_bond_features(bond)
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.edges[(start, end)]["edge_attr"] = [float(f) for f in bond_features]
        G.edges[(end, start)]["edge_attr"] = [float(f) for f in bond_features]

    for node in G.nodes():
        atom = mol.GetAtomWithIdx(node)
        atom_features = get_atom_features(atom)
        G.nodes[node]["x"] = [float(f) for f in atom_features]

    return G
