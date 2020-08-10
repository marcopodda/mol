import networkx as nx

from rdkit import DataStructs, Chem
from rdkit.Chem import Crippen, QED, AllChem

from core.mols import drd2_scorer
from core.mols.utils import mol_from_smiles, mols_from_smiles
from core.mols.sascorer.sascorer import calculateScore


def _num_long_cycles(mol):
    """Calculate the number of long cycles.
    Args:
    mol: Molecule. A molecule.
    Returns:
    negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if not cycle_list:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return -cycle_length


def sas(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return calculateScore(mol) if mol else None


def penalized_logp(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)

    if mol is None:
        return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Chem.Crippen.MolLogP(mol)
    SA = -calculateScore(mol)

    cycle_score = _num_long_cycles(mol)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


def drd2(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return drd2_scorer.get_score(mol)


def mr(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return Crippen.MolMR(mol) if mol else None


def qed(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return QED.qed(mol) if mol else None


def get_fingerprint(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


def similarity(mol1, mol2):
    fp1 = get_fingerprint(mol1)
    fp2 = get_fingerprint(mol2)
    return DataStructs.FingerprintSimilarity(fp1, fp2)


def bulk_tanimoto(mol, mols):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)

    if isinstance(mols[0], str):
        mols = mols_from_smiles(mols)

    fp = get_fingerprint(mol)
    fps = [get_fingerprint(m) for m in mols]
    return DataStructs.BulkTanimotoSimilarity(fp, fps)


def get_props_data(mol):
    return {
        "qed": qed(mol),
        "plogp": penalized_logp(mol),
        # "mr": mr(mol),
        # "fp": get_fingerprint(mol)
    }


def compare(smi1, smi2):
    props1 = get_props_data(smi1)
    props2 = get_props_data(smi2)
    return similarity(smi1, smi2)