import numpy as np
from pathlib import Path
from argparse import Namespace


from core.utils.serialization import load_yaml
from core.mols.props import drd2, qed, logp, similarity
from tasks import TRANSLATION


SR_KWARGS = {
    "drd2": {"prop_fun": drd2, "similarity_thres": 0.4, "improvement_thres": 0.5},
    "qed": {"prop_fun": qed, "similarity_thres": 0.4, "improvement_thres": 0.9},
    "logp4": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 1.2},
    "logp6": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 1.2},
}


def success_rate(x, y, prop_fun, similarity_thres, improvement_thres):
    sim, prop = similarity(x, y), prop_fun(y)
    return sim >= similarity_thres and prop >= improvement_thres    


def score(output_dir, dataset_name, epoch=1):
    output_dir = Path(output_dir)
    samples_dir = output_dir / TRANSLATION / "samples"
    samples_filename = f"samples_{epoch}.yml"
    samples = load_yaml(samples_dir / samples_filename)
    
    gen = [s["gen"] for s in samples]
    ref = [s["ref"] for s in samples]
    num_samples = len(samples)
    
    # valid samples
    valid_samples = [(x, y) for (x, y) in zip(ref, gen) if y]
    num_valid = len(valid_samples)
    validity_rate = num_valid / num_samples
    
    # novel samples
    novel_samples = [y not in ref for (x, y) in valid_samples]
    novelty_rate = len(novel_samples) / num_valid
    
    # unique samples
    unique_samples = set([y for (x, y) in valid_samples])
    uniqueness_rate = len(unique_samples) / num_valid

    # success rate    
    sr = [success_rate(x, y, **SR_KWARGS[dataset_name]) for (x, y) in valid_samples]
    
    return {
        "num_samples": len(samples),
        "success_rate": np.mean(sr),
        "valid": validity_rate,
        "unique": uniqueness_rate,
        "novel": novelty_rate,
        "scoring": dataset_name
    }
    
    
        
    