from rdkit import Chem
import numpy as np


# Fragmenting and building the encoding
MOL_SPLIT_START = 70


# Atom numbers of noble gases (should not be used as dummy atoms)
NOBLE_GASES = set([2, 10, 18, 36, 54, 86])


def ok_to_break(bond):
    """Check if it is ok to break a bond.
       It is ok to break a bond if:
       1. It is a single bond
       2. Either the start or the end atom is in a ring, but not both of them."""

    if bond.IsInRing():
        return False

    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False

    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    if not(begin_atom.IsInRing() or end_atom.IsInRing()):
        return False
    elif begin_atom.GetAtomicNum() >= MOL_SPLIT_START or \
            end_atom.GetAtomicNum() >= MOL_SPLIT_START:
        return False
    else:
        return True


def split_molecule(mol):
    """Divide a molecule into fragments."""

    split_id = MOL_SPLIT_START

    res = []
    to_check = [mol]
    while len(to_check) > 0:
        ms = spf(to_check.pop(), split_id)
        if len(ms) == 1:
            res += ms
        else:
            to_check += ms
            split_id += 1

    return create_chain(res)


def spf(mol, split_id):
    """Function for doing all the nitty gritty splitting work."""

    bonds = mol.GetBonds()
    for i in range(len(bonds)):
        if ok_to_break(bonds[i]):
            mol = Chem.FragmentOnBonds(mol, [i], addDummies=True, dummyLabels=[(0, 0)])
            # Dummy atoms are always added last
            n_at = mol.GetNumAtoms()
            mol.GetAtomWithIdx(n_at-1).SetAtomicNum(split_id)
            mol.GetAtomWithIdx(n_at-2).SetAtomicNum(split_id)
            return Chem.rdmolops.GetMolFrags(mol, asMols=True)

    # If the molecule could not been split, return original molecule
    return [mol]


def create_chain(splits):
    """Build up a chain of fragments from a molecule.
       This is required so that a given list of fragments can be rebuilt into the same
       molecule as was given when splitting the molecule."""
    
    splits_ids = np.asarray(
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits])
    

    splits_ids = \
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits]

    splits2 = []
    mv = np.max(splits_ids)
    look_for = [mv if isinstance(mv, np.int64) else mv[0]]
    join_order = []

    mols = []

    for i in range(len(splits_ids)):
        l = splits_ids[i]
        if l[0] == look_for[0] and len(l) == 1:
            mols.append(splits[i])
            splits2.append(splits_ids[i])
            splits_ids[i] = []

    while len(look_for) > 0:
        sid = look_for.pop()
        join_order.append(sid)
        next_mol = [i for i in range(len(splits_ids))
                      if sid in splits_ids[i]]

        if len(next_mol) == 0:
            break
        next_mol = next_mol[0]

        for n in splits_ids[next_mol]:
            if n != sid:
                look_for.append(n)
        mols.append(splits[next_mol])
        splits2.append(splits_ids[next_mol])
        splits_ids[next_mol] = []

    return [simplify_splits(mols[i], splits2[i], join_order) for i in range(len(mols))]


def simplify_splits(mol, splits, join_order):
    """Split and keep track of the order on how to rebuild the molecule."""

    td = {}
    n = 0
    for i in splits:
        for j in join_order:
            if i == j:
                td[i] = MOL_SPLIT_START + n
                n += 1
                if n in NOBLE_GASES:
                    n += 1

    for a in mol.GetAtoms():
        k = a.GetAtomicNum()
        if k in td:
            a.SetAtomicNum(td[k])

    return mol


def get_join_list(mol):
    """Go through a molecule and find attachment points and
       define in which order they should be re-joined."""

    join = []
    rem = []
    bonds = []

    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        if an >= MOL_SPLIT_START:
            while len(join) <= (an - MOL_SPLIT_START):
                rem.append(None)
                bonds.append(None)
                join.append(None)

            b = a.GetBonds()[0]
            ja = b.GetBeginAtom() if b.GetBeginAtom().GetAtomicNum() < MOL_SPLIT_START else b.GetEndAtom()
            join[an - MOL_SPLIT_START] = ja.GetIdx()
            rem[an - MOL_SPLIT_START] = a.GetIdx()
            bonds[an - MOL_SPLIT_START] = b.GetBondType()
            a.SetAtomicNum(0)

    return [x for x in join if x is not None],\
           [x for x in bonds if x is not None],\
           [x for x in rem if x is not None]


def join_fragments(fragments):
    """Join a list of fragments toghether into a molecule
       Throws an exception if it is not possible to join all fragments."""

    to_join = []
    bonds = []
    pairs = []
    del_atoms = []
    new_mol = fragments[0]

    j, b, r = get_join_list(fragments[0])
    to_join += j
    del_atoms += r
    bonds += b
    offset = fragments[0].GetNumAtoms()

    for f in fragments[1:]:
        j, b, r = get_join_list(f)
        p = to_join.pop()
        pb = bonds.pop()

        # Check bond types if b[:-1] == pb
        if b[:-1] != pb:
            assert("Can't connect bonds")

        pairs.append((p, j[-1] + offset,pb))

        for x in j[:-1]:
            to_join.append(x + offset)
        
        for x in r:
            del_atoms.append(x + offset)
        
        bonds += b[:-1]
        offset += f.GetNumAtoms()
        new_mol = Chem.CombineMols(new_mol, f)

    new_mol =  Chem.EditableMol(new_mol)

    for a1, a2, b in pairs:
        new_mol.AddBond(a1,a2, order=b)

    # Remove atom with greatest number first:
    for s in sorted(del_atoms, reverse=True):
        new_mol.RemoveAtom(s)
    
    return new_mol.GetMol()