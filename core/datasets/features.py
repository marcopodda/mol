import torch

from rdkit import Chem
from rdkit.Chem import rdmolops

from torch_geometric.utils import dense_to_sparse


ATOMS = {
    '*': 0,
    'Br': 1,
    'C': 2,
    'Cl': 3,
    'F': 4,
    'H': 5,
    'I': 6,
    'N': 7,
    'O': 8,
    'P': 9,
    'S': 10
}


ATOM_FEATURES = {
    'atomic_num': [6, 7, 8, 9, 16, 17, 35],
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
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2

BOND_FEATURES = {'stereo': [0, 1, 2, 3, 4, 5]}
BOND_FDIM = 13


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
    features += [atom.GetMass() * 0.01]  # scaled to about the same range as other features
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


def get_features(mol):
    A = rdmolops.GetAdjacencyMatrix(mol)
    node_features, edge_features = [], []

    for idx in range(A.shape[0]):
        atom = mol.GetAtomWithIdx(idx)
        atom_features = get_atom_features(atom)
        node_features.append(atom_features)

    for bond in mol.GetBonds():
        bond_features = get_bond_features(bond)
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_features.append(bond_features)
        edge_features.append(bond_features)  # add for reverse edge

    edge_index, eattr = dense_to_sparse(torch.Tensor(A))
    node_features = torch.Tensor(node_features)
    edge_features = torch.Tensor(edge_features)
    return edge_index, node_features, edge_features