import torch
import networkx as nx

from rdkit import Chem

from core.mols.utils import mol_from_smiles


ATOMS = ["C", "N", "S", "O", "F", "Cl", "Br"]


ATOM_FEATURES = {
    'atomic_num': [6, 7, 8, 9, 16, 17, 35], # 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83],  # [6, 7, 8, 9, 16, 17, 35],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
}
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values())  + 1

BOND_FEATURES = {'stereo': [0, 1, 2, 3, 4, 5]}
BOND_FDIM = 12

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
    features += [1 if atom.GetIsAromatic() else 0]
    return features


def get_bond_features(bond):
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
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


def tensorize(mol_batch):
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
        in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                af = torch.Tensor(get_atom_features(atom))
                fatoms.append(af)
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds)
                all_bonds.append((x,y))
                bf = torch.Tensor(get_bond_features(bond))
                fbonds.append( torch.cat([fatoms[x], bf], 0) )
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y,x))
                bf = torch.Tensor(get_bond_features(bond))
                fbonds.append( torch.cat([fatoms[y], bf], 0) )
                in_bonds[x].append(b)

            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms, 6).long()
        bgraph = torch.zeros(total_bonds, 6).long()

        for a in range(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(1, total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1,i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)