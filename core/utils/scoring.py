import numpy as np
from pathlib import Path

from moses import get_all_metrics

from core.datasets.utils import load_data
from core.utils.serialization import load_yaml
from core.mols.props import drd2, qed, logp, similarity
from core.mols.utils import mol_from_smiles


SR_KWARGS = {
    "moses": {"prop_fun": drd2, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "ZINC": {"prop_fun": drd2, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "drd2": {"prop_fun": drd2, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "qed": {"prop_fun": qed, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "logp04": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 0.8},
    "logp06": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 0.8},
}


def is_similar(x, y, similarity_thres):
    return similarity(x, y) >= similarity_thres


def is_improved(y, prop_fun, improvement_thres):
    return prop_fun(y) >= improvement_thres


def success_rate(x, y, prop_fun, similarity_thres, improvement_thres):
    sim, prop = similarity(x, y), prop_fun(y)
    return sim >= similarity_thres and prop >= improvement_thres


def score(exp_dir, dataset_name, epoch=0):
    exp_dir = Path(exp_dir)
    samples_dir = exp_dir / "samples"
    samples_filename = f"samples_{epoch}.yml"

    samples = load_yaml(samples_dir / samples_filename)
    num_samples = len(samples)

    ref = [s["ref"] for s in samples]
    gen = [s["gen"] for s in samples]

    # valid samples
    valid_samples = [(x, y) for (x, y) in zip(ref, gen) if y and mol_from_smiles(y)]
    num_valid = len(valid_samples)

    # novel samples
    data, _, _ = load_data(dataset_name)
    training_set = set(data[data.is_train == True].smiles.tolist())
    novel_samples = [y not in training_set for (_, y) in valid_samples]

    # unique samples
    unique_samples = set([y for (_, y) in valid_samples])

    # similarity
    kw = SR_KWARGS[dataset_name].copy()
    similarities = [similarity(x, y) for (x, y) in valid_samples]
    similar = [s > kw["similarity_thres"] for s in similarities]

    # property
    fun = kw["prop_fun"]
    ref_prop = [fun(x) for (x, _) in valid_samples]
    gen_prop = [fun(y) for (_, y) in valid_samples]

    # improvement
    improvement = [g - r for (r, g) in zip(ref_prop, gen_prop)]
    improved = [p > kw["improvement_thres"] for p in gen_prop]

    # success
    success = [x and y for (x, y) in zip(similar, improved)]

    # reconstructed
    reconstructed = [x == y for (x, y) in valid_samples]

    return {
        "scoring": dataset_name,
        "num_samples": len(samples),
        "valid": num_valid / num_samples,
        "unique": len(unique_samples) / num_valid,
        "novel": sum(novel_samples) / num_valid,
        "property": (np.mean(gen_prop), np.std(gen_prop)),
        "similar": sum(similar) / num_valid,
        "avg_similarity": (np.mean(similarities), np.std(similarities)),
        "improved": sum(improved) / num_valid,
        "avg_improvement": (np.mean(improvement), np.std(improvement)),
        "success_rate": sum(success) / num_valid,
        "recon_rate": sum(reconstructed) / num_valid,
    }


def convert_metrics_dict(metrics_dict):
    for k in metrics_dict.keys():
        metrics_dict[k] = float(metrics_dict[k])
    return metrics_dict


def moses_score(exp_dir, epoch=0, n_jobs=40):
    exp_dir = Path(exp_dir)
    samples_dir = exp_dir / "samples"
    samples_path = samples_dir / f"samples_{epoch}.yml"
    samples = load_yaml(samples_path)

    ref_samples = [s["ref"] for s in samples]
    gen_samples = [s["gen"] for s in samples]

    scores = get_all_metrics(gen_samples, test=ref_samples, n_jobs=n_jobs)
    return convert_metrics_dict(scores)
