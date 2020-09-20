import numpy as np
from pathlib import Path

from moses import get_all_metrics

from core.datasets.utils import load_data
from core.utils.serialization import load_yaml, save_yaml
from core.mols.props import drd2, qed, logp, similarity as sim, bulk_tanimoto
from core.mols.utils import mol_from_smiles


SR_KWARGS = {
    "moses": {"prop_fun": drd2, "similarity_thres": 0.4, "improvement_thres": 0.6},
    "ZINC": {"prop_fun": drd2, "similarity_thres": 0.4, "improvement_thres": 0.6},
    "drd2": {"prop_fun": drd2, "similarity_thres": 0.4, "improvement_thres": 0.6},
    "qed": {"prop_fun": qed, "similarity_thres": 0.4, "improvement_thres": 0.9},
    "logp04": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 0.8},
    "logp06": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 0.8},
}


def validity(ref, gen):
    valid = [(x, y) for (x, y) in zip(ref, gen) if y and y != "*" and mol_from_smiles(y)]
    return valid, round(len(valid) / len(ref), 4)


def novelty(valid_gen, dataset_name):
    data, _, _ = load_data(dataset_name)
    training_set = set(data[data.is_train == True].smiles.tolist())
    novel = [g not in training_set for g in valid_gen]
    return round(sum(novel) / len(valid_gen), 4)


def uniqueness(valid_gen):
    unique = set(valid_gen)
    return round(len(unique) / len(valid_gen), 4)


def similarity(valid_ref, valid_gen, kwargs):
    scores = np.array([sim(x, y) for (x, y) in zip(valid_ref, valid_gen)])
    similar = scores > kwargs["similarity_thres"]
    return similar, round(similar.mean().item(), 4)


def diversity(valid_gen):
    cumsum, tot = 0, 0
    num_samples = len(valid_gen)

    for i in range(num_samples):
        for j in range(i+1, num_samples):
            cumsum += 1.0 - sim(valid_gen[i], valid_gen[j])
            tot += 1

    return round(cumsum / tot, 4)


def improvement(valid_gen, kwargs):
    prop_fun = kwargs["prop_fun"]
    gen_score = np.array([prop_fun(g) for g in valid_gen])
    improved = gen_score > kwargs["improvement_thres"]
    return improved, round(improved.mean().item(), 4)


def success_rate(similar, improved):
    scores = similar & improved
    return round(scores.mean().item(), 4)


def score(exp_dir, dataset_name, fun=None, epoch=0):
    exp_dir = Path(exp_dir)
    fun = fun or dataset_name
    samples_dir = exp_dir / "samples"
    samples_filename = f"samples_{epoch}.yml"
    scores_filename = f"samples_{epoch}_scores_{fun}.yml"
    scores_path = samples_dir / scores_filename

    if not scores_path.exists():
        samples = load_yaml(samples_dir / samples_filename)

        ref = [s["ref"] for s in samples]
        gen = [s["gen"] for s in samples]

        kwargs = SR_KWARGS[fun]

        valid_samples, validity_score = validity(ref, gen)
        valid_ref, valid_gen = zip(*valid_samples)
        novelty_score = novelty(valid_gen, dataset_name)
        uniqueness_score = uniqueness(valid_gen)
        diversity_scores = diversity(valid_gen)
        similar, similarity_score = similarity(valid_ref, valid_gen, kwargs)
        improved, improvement_score = improvement(valid_gen, kwargs)
        success_rate_score = success_rate(similar, improved)

        data = {
            "scoring": dataset_name,
            "num_samples": len(samples),
            "valid": validity_score,
            "unique": uniqueness_score,
            "novel": novelty_score,
            "similar": similarity_score,
            "diverse": diversity_scores,
            "improved": improvement_score,
            "success_rate": success_rate_score,
        }

        save_yaml(data, samples_dir / scores_filename)

    return load_yaml(scores_path)


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
    valid_gen = [s["gen"] for s in samples]

    scores = get_all_metrics(valid_gen, test=ref_samples, n_jobs=n_jobs)
    return convert_metrics_dict(scores)
