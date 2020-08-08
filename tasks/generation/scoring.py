from pathlib import Path

from rdkit import Chem
from moses import get_all_metrics

from core.mols.split import join_fragments
from core.utils.vocab import Vocab
from core.utils.serialization import load_yaml


def score(output_dir, epoch, n_jobs=40):
    output_dir = Path(output_dir)
    samples_dir = output_dir / "generation" / "samples"
    vocab = Vocab.from_file(output_dir / "DATA" / "vocab.csv")
    samples_path = samples_dir / f"samples_{epoch}.yml"
    
    if samples_path.exists():
        tokens_list = load_yaml(samples_path)
        
        mols = []
        for tokens in tokens_list:
            frags = [Chem.MolFromSmiles(vocab[t]) for t in tokens]
            try:
                mols.append(join_fragments(frags))
            except:
                pass
        
        smis = [Chem.MolToSmiles(m) for m in mols]
        print(f"number of correct SMILES samples: {len(smis)}")
        
        return get_all_metrics(smis, n_jobs=n_jobs)