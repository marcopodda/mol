import numpy as np
from pathlib import Path
from moses import get_all_metrics

from core.utils.serialization import load_yaml
from tasks import PRETRAINING


def convert_metrics_dict(metrics_dict):
    for k in metrics_dict.keys():
        metrics_dict[k] = float(metrics_dict[k])
    return metrics_dict


def reconstruction_accuracy(ref, gen):
    total = len(ref)
    correct = sum([x==y for (x, y) in zip(ref, gen)])
    return correct / total


def score(output_dir, dataset_name, epoch=1, n_jobs=40):
    output_dir = Path(output_dir)
    samples_dir = output_dir / PRETRAINING / "samples"
    samples_path = samples_dir / f"samples_{epoch}.yml"
    samples = load_yaml(samples_path)
    
    ref_samples = [s["smi"] for s in samples]
    gen_samples = [s["gen"] for s in samples]
    
    scores = get_all_metrics(gen_samples, n_jobs=n_jobs)
    scores["rec"] = reconstruction_accuracy(ref_samples, gen_samples)
    return scores
        