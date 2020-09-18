import numpy as np
from pathlib import Path

from moses import get_all_metrics

from core.datasets.utils import load_data
from core.utils.serialization import load_yaml
from core.mols.props import drd2, qed, logp, similarity as sim
from core.mols.utils import mol_from_smiles


SR_KWARGS = {
    "moses": {"prop_fun": drd2, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "ZINC": {"prop_fun": drd2, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "drd2": {"prop_fun": drd2, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "qed": {"prop_fun": qed, "similarity_thres": 0.3, "improvement_thres": 0.6},
    "logp04": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 0.8},
    "logp06": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 0.8},
}


def validity(ref, gen):
    valid = [(x, y) for (x, y) in zip(ref, gen) if y and mol_from_smiles(y)]
    return valid, round(len(valid) / len(ref), 4)


def novelty(dataset_name, valid_gen):
    data, _, _ = load_data(dataset_name)
    training_set = set(data[data.is_train == True].smiles.tolist())
    novel = [g not in training_set for g in valid_gen]
    return round(sum(novel) / len(valid_gen), 4)


def uniqueness(valid_gen):
    unique = set(valid_gen)
    return round(len(unique) / len(valid_gen), 4)


def similarity(valid_samples, kwargs):
    scores = np.array([sim(x, y) for (x, y) in valid_samples])
    mean, std = scores.mean(), scores.std()
    similar = scores > kwargs["similarity_thres"]
    score = sum(similar) / len(similar)
    return similar, {
        "score": round(score, 4),
        "mean": round(mean, 4),
        "std": round(std, 4)
    }


def diversity(valid_samples):
    diverse = np.array([1.0 - sim(x, z) for (x, y) in valid_samples for (_, z) in valid_samples])
    mean, std = diverse.mean(), diverse.std()
    return diverse, {
        "mean": round(mean, 4),
        "std": round(std, 4)
    }


def improvement(valid_samples, kwargs):
    prop_fun = kwargs["prop_fun"]
    ref_score = np.array([prop_fun(v[0]) for v in valid_samples])
    gen_score = np.array([prop_fun(v[1]) for v in valid_samples])
    scores = gen_score - ref_score
    mean, std = scores.mean(), scores.std()
    improved = scores > kwargs["improvement_thres"]
    score = sum(improved) / len(improved)
    return improved, {
        "score": round(score, 4),
        "mean": round(mean, 4),
        "std": round(std, 4)
    }


def success_rate(similar, improved):
    scores = similar & improved
    return round(sum(scores) / len(scores), 4)


def score(exp_dir, dataset_name, epoch=0):
    exp_dir = Path(exp_dir)
    samples_dir = exp_dir / "samples"
    samples_filename = f"samples_{epoch}.yml"

    samples = load_yaml(samples_dir / samples_filename)

    ref = [s["ref"] for s in samples]
    gen = [s["gen"] for s in samples]
    kwargs = SR_KWARGS[dataset_name]

    valid_samples, validity_score = validity(ref, gen)
    valid, valid_gen = zip(*valid_samples)
    novelty_score = novelty(dataset_name, valid_gen)
    uniqueness_score = uniqueness(valid_gen)
    similar, similarity_scores = similarity(valid_samples, kwargs)
    diverse, diversity_scores = diversity(valid_samples)
    improved, improvement_scores = improvement(valid_samples, kwargs)
    success_rate_score = success_rate(similar, improved)

    return {
        "scoring": dataset_name,
        "num_samples": len(samples),
        "valid": validity_score,
        "unique": uniqueness_score,
        "novel": novelty_score,
        "similar": similarity_scores,
        "diverse": diversity_scores,
        "improved": improvement_scores,
        "success_rate": success_rate_score,
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
