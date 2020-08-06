from pathlib import Path

from rdkit import Chem
from moses import get_all_metrics

from core.mols.split import join_fragments
from core.utils.vocab import Vocab
from core.utils.serialization import load_yaml


def score(output_dir, tokens_path, n_jobs=40):
    vocab = Vocab.from_file(Path(output_dir) / "DATA" / "vocab.csv")
    tokens_list = load_yaml(Path(tokens_path))
    
    mols = []
    for tokens in tokens_list:
        frags = [Chem.MolFromSmiles(vocab[t]) for t in tokens]
        try:
            mols.append(join_fragments(frags))
        except:
            pass
    
    smis = [Chem.MolToSmiles(m) for m in mols]
    return get_all_metrics(smis, n_jobs=n_jobs)
        
    
    
    