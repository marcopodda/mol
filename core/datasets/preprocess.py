import ast
import warnings
from pathlib import Path

import pandas as pd

from joblib import Parallel, delayed

from rdkit import Chem
from molvs import standardize_smiles

from core.datasets.download import fetch_dataset
from core.datasets.utils import load_csv_data
from core.datasets.features import ATOM_FEATURES
from core.utils.vocab import Vocab
from core.mols.props import get_props_data
from core.mols.fragmentation import fragment
from core.mols.split import split_molecule
from core.utils.os import get_or_create_dir, dir_is_empty
from core.utils.misc import get_n_jobs
from core.utils.serialization import load_yaml

import core.datasets.clean as clean_functions


ROOT_DIR = Path(".")
DATA_DIR = Path("DATA")
CONFIG_DIR = DATA_DIR / "CONFIG"


def get_dataset_info(name):
    path = CONFIG_DIR / f'{name}.yml'
    return load_yaml(path)


def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        element_list = ATOM_FEATURES["atomic_num"]
        num_heavy = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        return num_heavy and elements
    return False


def clean_molecule(smi):
    datadict = {}

    try:
        smi = standardize_smiles(smi)
        mol = Chem.MolFromSmiles(smi)
        if filter_mol(mol):
            frags = split_molecule(mol)
            length = len(frags)

            if length > 1:
                datadict.update(smiles=smi)
                datadict.update(**get_props_data(mol))
                datadict.update(frags=[Chem.MolToSmiles(f) for f in frags])
                datadict.update(length=length)
    except:
        print(f"Couldn't process {smi}.")
       
    return datadict


def process_data(smiles, n_jobs):
    P = Parallel(n_jobs=n_jobs, verbose=1)
    data = P(delayed(clean_molecule)(smi) for smi in smiles)
    return pd.DataFrame(data).dropna().reset_index(drop=True)


def populate_vocab(df, n_jobs):
    print("Create vocab...", end=" ")
    vocab = Vocab()
    
    for _, frags in df.frags.iteritems():
        [vocab.update(f) for f in frags]
    print("Done.")
    
    return vocab


def get_vocab(df, path):
    vocab = Vocab.from_file(path)
    new_vocab = Vocab()
    
    for _, frags in df.frags.iteritems():
        [new_vocab.update(f) for f in frags]
    
    return new_vocab
    

def run_preprocess(dataset_name):
    info = get_dataset_info(dataset_name)
    raw_dir = get_or_create_dir(DATA_DIR / dataset_name / "RAW")

    if not raw_dir.exists() or dir_is_empty(raw_dir):
        fetch_dataset(info, dest_dir=raw_dir)

    processed_dir = get_or_create_dir(DATA_DIR / dataset_name / "PROCESSED")
    processed_data_path = processed_dir / "data.csv"
    processed_vocab_path = processed_dir / "vocab.csv"

    if not processed_data_path.exists():
        n_jobs = get_n_jobs()
        clean_fn = getattr(clean_functions, f"clean_{dataset_name}")
        data = clean_fn(raw_dir, info)
        cleaned_data = process_data(data[info["smiles_col"]], n_jobs)
        postprocess_fn = getattr(clean_functions, f"postprocess_{dataset_name}")
        cleaned_data = postprocess_fn(cleaned_data, data)
        cleaned_data = cleaned_data.dropna().reset_index(drop=True)
        cleaned_data.to_csv(processed_data_path)
        if not processed_vocab_path.exists():
            vocab = populate_vocab(cleaned_data, n_jobs)
            vocab.save(processed_vocab_path)


def get_data(dest_dir, dataset_name, num_samples=None):
    dest_dir = Path(dest_dir)

    processed_dir = DATA_DIR / dataset_name / "PROCESSED"
    processed_data_path = processed_dir / "data.csv"
    processed_vocab_path = processed_dir / "vocab.csv"

    if not processed_data_path.exists():
        print("Preprocess your dataset first!")
        exit(1)

    if not processed_vocab_path.exists():
        print("Create the vocabulary first!")
        exit(1)

    dest_dir = get_or_create_dir(dest_dir / "DATA")
    dest_data_path = dest_dir / "data.csv"
    dest_vocab_path = dest_dir / "vocab.csv"

    n_jobs = get_n_jobs()

    if dest_data_path.exists():
        data = load_csv_data(dest_data_path, convert=["frags"], cast={"length": int})
        vocab = Vocab.from_file(dest_vocab_path)
        if num_samples is not None and num_samples != data.shape[0]:
            warnings.warn(f"Got num samples: {num_samples} != data size: {data.shape[0]}. Overriding.")
            if num_samples is not None:
                data = data.sample(n=num_samples)
                data = data.reset_index(drop=True)
            data.to_csv(dest_data_path)
            
            vocab = get_vocab(data, processed_vocab_path)
            vocab.save(dest_vocab_path)
    else:
        data = load_csv_data(processed_data_path, convert=["frags"], cast={"length": int})
        if num_samples is not None:
            data = data.sample(n=num_samples)
            data = data.reset_index(drop=True)
        data.to_csv(dest_data_path)
        
        vocab = get_vocab(data, processed_vocab_path)
        vocab.save(dest_vocab_path)

    data = load_csv_data(dest_data_path, convert=["frags"], cast={"length": int})
    vocab = Vocab.from_file(dest_vocab_path)
    return data, vocab
