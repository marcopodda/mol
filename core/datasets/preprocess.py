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


ROOT_DIR = Path(".")
DATA_DIR = Path("DATA")
CONFIG_DIR = DATA_DIR / "CONFIG"


def get_dataset_info(name):
    path = CONFIG_DIR / f'{name}.yml'
    return load_yaml(path)


def clean_ZINC(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data = raw_data.replace(r'\n', '', regex=True)
    return raw_data[:500]


def postprocess_ZINC(cleaned_data, raw_data):
    return cleaned_data


def clean_drd2(raw_dir, info):
    file_names = ["train_pairs.txt", "valid.txt", "test.txt"]
    
    
    raw_data_path = raw_dir / "train_pairs.txt"
    raw_data = pd.read_csv(raw_data_path, names=["x", "y"], **info["parse_args"])[:100]
    
    all_smiles, is_x, is_y, is_valid, is_test = [], [], [], [], []
    
    all_smiles += raw_data.x.tolist()
    is_x += [1] * raw_data.shape[0]
    is_y += [0] * raw_data.shape[0]
    is_valid += [0] * raw_data.shape[0]
    is_test += [0] * raw_data.shape[0]
    
    all_smiles += raw_data.y.tolist()
    is_x += [0] * raw_data.shape[0]
    is_y += [1] * raw_data.shape[0]
    is_valid += [0] * raw_data.shape[0]
    is_test += [0] * raw_data.shape[0]
    
    raw_data_path = raw_dir / "valid.txt"
    raw_data = pd.read_csv(raw_data_path, names=["smiles"], **info["parse_args"])[:100]
    
    all_smiles += raw_data.smiles.tolist()
    is_x += [0] * raw_data.shape[0]
    is_y += [0] * raw_data.shape[0]
    is_valid += [1] * raw_data.shape[0]
    is_test += [0] * raw_data.shape[0]
    
    raw_data_path = raw_dir / "test.txt"
    raw_data = pd.read_csv(raw_data_path, names=["smiles"], **info["parse_args"])[:100]
    
    all_smiles += raw_data.smiles.tolist()
    is_x += [0] * raw_data.shape[0]
    is_y += [0] * raw_data.shape[0]
    is_valid += [0] * raw_data.shape[0]
    is_test += [1] * raw_data.shape[0]
    
    return pd.DataFrame({"smiles": all_smiles, "is_x": is_x, "is_y": is_y, "is_valid": is_valid, "is_test": is_test})


def postprocess_drd2(cleaned_data, raw_data):
    cleaned_data["is_x"] = raw_data.is_x
    cleaned_data["is_y"] = raw_data.is_y
    cleaned_data["is_valid"] = raw_data.is_valid
    cleaned_data["is_test"] = raw_data.is_test
    return cleaned_data


def clean_MolPort(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data.columns = ["smiles", "molPortID", "url"]
    return raw_data


def postprocess_MolPort(cleaned_data, raw_data):
    return cleaned_data


def clean_ChEMBL(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    raw_data.columns = ["smiles"]
    return raw_data


def postprocess_CHEMBL(cleaned_data, raw_data):
    return cleaned_data


def clean_moses(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    return raw_data


def postprocess_moses(cleaned_data, raw_data):
    return cleaned_data


def clean_AID1706(raw_dir, info):
    raw_data_path = raw_dir / info["filename"]
    raw_data = pd.read_csv(raw_data_path, **info["parse_args"])
    return raw_data


def postprocess_AID1706(cleaned_data, raw_data):
    cleaned_data["orig_smiles"] = raw_data.smiles
    cleaned_data["activity"] = raw_data.activity
    return cleaned_data


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
        clean_fn = globals()[f"clean_{dataset_name}"]
        data = clean_fn(raw_dir, info)
        cleaned_data = process_data(data[info["smiles_col"]], n_jobs)
        postprocess_fn = globals()[f"postprocess_{dataset_name}"]
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
