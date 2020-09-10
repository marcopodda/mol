from rdkit import Chem
from rdkit.Chem import BRICS

from core.mols.utils import mol_to_smiles


DUMMY = Chem.MolFromSmiles('*')


def strip_dummy_atoms(mol):
    hydrogen = Chem.MolFromSmiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, DUMMY, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol


def break_on_bond(mol, bond):
    broken = Chem.FragmentOnBonds(mol, bondIndices=[bond], dummyLabels=[(0, 0)])
    res = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    data = Chem.GetMolFrags(broken, asMols=False)
    return res, data


def get_bonds(mol):
    bond_data = list(BRICS.FindBRICSBonds(mol))
    try:
        idxs, labs = zip(*bond_data)
        return idxs
    except Exception:
        return []


def get_frags(mol):
    bonds = get_bonds(mol)

    if bonds == []:
        return [mol]

    bonds_list = []
    for a1, a2 in bonds:
        bond = mol.GetBondBetweenAtoms(a1, a2).GetIdx()
        broken, data = break_on_bond(mol, bond)
        bonds_list.append([broken, len(data[0])])

    res = min(bonds_list, key=lambda x: x[1])
    return res[0]


def fragment(mol):
    frags = []
    temp = get_frags(mol)

    while len(temp) == 2:
        frags.append(temp[0])
        temp = get_frags(temp[1])

    frags.extend(temp)

    if len(frags) > 1:
        smi = mol_to_smiles(mol)
        rec = reconstruct(frags)
        assert mol_to_smiles(rec) == smi

    return frags, len(frags)


def join_fragments(fragA, fragB):
    for atom in fragB.GetAtoms():
        if atom.GetAtomicNum() == 0:
            break

    markedB = atom.GetIdx()

    ed = Chem.EditableMol(fragB)
    ed.RemoveAtom(markedB)
    fragB = ed.GetMol()

    joined = Chem.ReplaceSubstructs(fragA, DUMMY, fragB)[0]

    return joined


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def reconstruct(frags):
    try:
        if count_dummies(frags[0]) != 1:
            return None

        if count_dummies(frags[-1]) != 1:
            return None

        for f in frags[1:-1]:
            if count_dummies(f) != 2:
                return None

        mol = join_fragments(frags[0], frags[1])
        # print(f"[[['{mol_to_smiles(frags[0])}', '{mol_to_smiles(frags[1])}'], '{mol_to_smiles(mol)}'],")
        for i, frag in enumerate(frags[2:]):
            # print(f"[['{mol_to_smiles(mol)}', '{mol_to_smiles(frag)}'],", end=" ")
            mol = join_fragments(mol, frag)
            # print(f"'{mol_to_smiles(mol)}'],")
        # print("]")

        # see if there are kekulization/valence errors
        mol_to_smiles(mol)
        Chem.SanitizeMol(mol)

        return mol
    except Exception:
        return None
