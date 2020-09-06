import numpy as np
from pathlib import Path
from argparse import Namespace


from core.utils.serialization import load_yaml
from core.mols.props import drd2, qed, logp
from tasks import TRANSLATION


SCORING = {
    "drd2": {"prop": drd2, "score_fun": lambda x: x >= 0.5},
    "logp4": {"prop": logp, "score_fun": lambda x: x >= 0.4},
    "logp6": {"prop": logp, "score_fun": lambda x: x >= 0.6},
    "qed": {"prop": qed, "score_fun": lambda x: x >= 0.9},
}


def score(output_dir, dataset_name, epoch=1):
    output_dir = Path(output_dir)
    samples_dir = output_dir / TRANSLATION / "samples"
    samples_filename = f"samples_{epoch}.yml"
    samples = load_yaml(samples_dir / samples_filename)
    
    ref_samples = [s["ref"] for s in samples]
    gen_samples = [s["gen"] for s in samples]
    prop = SCORING[dataset_name]["prop"]
    score_fun = SCORING[dataset_name]["score_fun"]
    
    scored_samples = [prop(g) for g in gen_samples]
    valid_scored_samples = [s for s in scored_samples if s]
    num_valid_scored_samples = len(valid_scored_samples)
    score = [score_fun(s) for s in valid_scored_samples]
    
    return {
        "num_samples": len(samples),
        "dataset_name": dataset_name,
        "score": np.mean(score),
        "valid": len(valid_scored_samples) / len(gen_samples),
        "unique": len(set(valid_scored_samples)) / num_valid_scored_samples,
        "novel": len([s in ref_samples for s in valid_scored_samples]) / num_valid_scored_samples
    }
    
    
        
    