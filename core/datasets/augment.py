import numpy as np

from core.mols.utils import mol_from_smiles, mols_from_smiles, mol_to_smiles, mols_to_smiles
from core.mols.split import split_molecule, join_fragments
from core.mols.props import similarity


def corrupt(seq, vocab, max_length, reps=1):
    seq = seq[:]

    for _ in range(reps):
        # deletion
        if np.random.rand() > 0.1 and len(seq) > 2:
            delete_index = np.random.choice(len(seq)-1)
            seq.pop(delete_index)

        # replacement
        if  np.random.rand() > 0.1:
            mask_index = np.random.choice(len(seq)-1)
            probs = vocab.condition(seq[mask_index])
            seq[mask_index] = vocab.sample() # probs=probs)

        # insertion
        if np.random.rand() > 0.1 and len(seq) + 2 <= max_length:
            add_index = np.random.choice(len(seq)-1)
            probs = vocab.condition(seq[add_index])
            seq.insert(add_index, vocab.sample()) # probs=probs))

    return seq


def augment(data, vocab, num):
    target_smiles = data[data.target != "*"].target.tolist()
    max_length = data.length.max() + 1
    augmented = []

    for target in target_smiles:
        try:
            mol = mol_from_smiles(target)
            frags = mols_to_smiles(split_molecule(mol))

            new_frags = corrupt(frags, vocab, max_length)
            new_mol = join_fragments(mols_from_smiles(new_frags))
            new_smiles = mol_to_smiles(new_mol)
            sim = similarity(target, new_smiles)

            while sim < 0.4:
                new_frags = corrupt(frags, vocab, max_length)
                new_mol = join_fragments(mols_from_smiles(new_frags))
                new_smiles = mol_to_smiles(new_mol)
                sim = similarity(target, new_smiles)

            augmented.append(new_smiles)
            print(len(augmented))
        except:
            continue
    return augmented