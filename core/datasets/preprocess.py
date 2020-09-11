import warnings
import pandas as pd

from joblib import Parallel, delayed
from molvs import standardize_smiles

from core.datasets import clean as clean_functions
from core.datasets.download import fetch_dataset
from core.datasets.features import ATOM_FEATURES
from core.datasets.utils import load_dataset_info
from core.datasets.vocab import Vocab
from core.mols.props import get_props_data
from core.mols.split import split_molecule
from core.mols.utils import mol_from_smiles, mols_to_smiles
from core.utils.os import get_or_create_dir, dir_is_empty
from core.utils.misc import get_n_jobs
from core.datasets.settings import DATA_DIR


def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        element_list = ATOM_FEATURES["atomic_num"]
        num_heavy = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        return num_heavy and elements
    return False


def clean_mol(smi):
    datadict = {}

    try:
        smi = standardize_smiles(smi)
        mol = mol_from_smiles(smi)
        if filter_mol(mol):
            frags = split_molecule(mol)
            length = len(frags)

            if length > 1:
                datadict.update(smiles=smi)
                datadict.update(**get_props_data(mol))
                datadict.update(frags=mols_to_smiles(frags))
                datadict.update(length=length)
    except Exception:
        pass
        # print(f"Couldn't process {smi}:", e)

    return datadict


def process_data(smiles, n_jobs):
    P = Parallel(n_jobs=n_jobs, verbose=1)
    data = P(delayed(clean_mol)(smi) for smi in smiles)
    return pd.DataFrame(data, dtype=object).dropna().reset_index(drop=True)


def run_preprocess(dataset_name):
    info = load_dataset_info(dataset_name)
    raw_dir = get_or_create_dir(DATA_DIR / dataset_name / "RAW")

    if not raw_dir.exists() or dir_is_empty(raw_dir):
        fetch_dataset(info, dest_dir=raw_dir)

    processed_dir = get_or_create_dir(DATA_DIR / dataset_name / "PROCESSED")
    processed_data_path = processed_dir / "data.csv"
    processed_vocab_path = processed_dir / "vocab.csv"

    if not processed_data_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            n_jobs = get_n_jobs()
            clean_fn = getattr(clean_functions, f"clean_{dataset_name}")
            data = clean_fn(raw_dir, info)
            cleaned_data = process_data(data[info["smiles_col"]], n_jobs)
            postprocess_fn = getattr(clean_functions, f"postprocess_{dataset_name}")
            cleaned_data = postprocess_fn(cleaned_data, data)
            cleaned_data = cleaned_data.dropna().reset_index(drop=True)
            cleaned_data.to_csv(processed_data_path)
            if not processed_vocab_path.exists():
                vocab = Vocab.from_df(cleaned_data)
                vocab.save(processed_vocab_path)
