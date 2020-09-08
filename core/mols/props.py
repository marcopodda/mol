import networkx as nx

from rdkit import DataStructs, Chem
from rdkit.Chem import Crippen, QED, AllChem, Descriptors

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


def penalized_logp(mol, logP=None, SAS=None):
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

    if not logP:
        logP = Chem.Crippen.MolLogP(mol)
    if not SAS:
        SAS = calculateScore(mol)

    cycle_score = _num_long_cycles(mol)

    normalized_logp = (logP - logP_mean) / logP_std
    normalized_SA = ((-SAS) - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_logp + normalized_SA + normalized_cycle


def logp(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return Crippen.MolLogP(mol) if mol else 0.0


def drd2(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return drd2_scorer.get_score(mol) if mol else 0.0


def mr(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return Crippen.MolMR(mol) if mol else 0.0

def mw(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return Descriptors.MolWt(mol) if mol else 0.0


def qed(mol):
    if isinstance(mol, str):
        mol = mol_from_smiles(mol)
    return QED.qed(mol) if mol else 0.0


def get_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)


def similarity(mol1, mol2):
    if isinstance(mol1, str):
        mol1 = mol_from_smiles(mol1)
    if isinstance(mol2, str):
        mol2 = mol_from_smiles(mol2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
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
    logP, SAS = logp(mol), sas(mol)
    return {
        "qed": qed(mol),
        "logP": logP,
        "SAS": SAS,
        "plogP": penalized_logp(mol, logP=logP, SAS=SAS),
        "mw": mw(mol),
        "mr": mr(mol)
    }