from pathlib import Path

from rdkit import Chem

from core.mols.split import join_fragments
from core.utils.metrics import get_all_metrics
from core.utils.serialization import load_yaml, save_yaml
from core.utils.vocab import Vocab


def convert_metrics_dict(metrics_dict):
    for k in metrics_dict.keys():
        metrics_dict[k] = float(metrics_dict[k])
    return metrics_dict


def score(output_dir, epoch, n_jobs=40):
    assert epoch >= 1
    output_dir = Path(output_dir)
    samples_dir = output_dir / "generation" / "samples"
    vocab = Vocab.from_file(output_dir / "DATA" / "vocab.csv")
    samples_path = samples_dir / f"samples_{epoch}.yml"
    
    if samples_path.exists():
        smiles = load_yaml(samples_path)
        
        metrics = get_all_metrics(smiles, n_jobs=n_jobs)
        metrics = convert_metrics_dict(metrics)
        metrics["NumSamples"] = len(smiles)
        
        path = samples_path.parent / f"{samples_path.stem}_results.yml"
        save_yaml(metrics, path)
        return metrics
        