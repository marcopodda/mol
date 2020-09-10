from pathlib import Path

from moses import get_all_metrics

from core.utils.serialization import load_yaml
from core.mols.props import drd2, qed, logp, similarity
from core.mols.utils import mol_from_smiles
from tasks import PRETRAINING, TRANSLATION


SR_KWARGS = {
    "ZINC": {"prop_fun": drd2, "similarity_thres": 0.4, "improvement_thres": 0.5},
    "drd2": {"prop_fun": drd2, "similarity_thres": 0.4, "improvement_thres": 0.5},
    "qed": {"prop_fun": qed, "similarity_thres": 0.4, "improvement_thres": 0.9},
    "logp4": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 1.2},
    "logp6": {"prop_fun": logp, "similarity_thres": 0.4, "improvement_thres": 1.2},
}


def is_similar(x, y, similarity_thres):
    return similarity(x, y) >= similarity_thres


def is_improved(y, prop_fun, improvement_thres):
    return prop_fun(y) >= improvement_thres


def success_rate(x, y, prop_fun, similarity_thres, improvement_thres):
    sim, prop = similarity(x, y), prop_fun(y)
    return sim >= similarity_thres and prop >= improvement_thres


def score(results_dir, epoch=1):
    results_dir = Path(results_dir)
    dataset_name = results_dir.parts[-2]
    samples_dir = results_dir / "samples"
    samples_filename = f"samples_{epoch}.yml"
    samples = load_yaml(samples_dir / samples_filename)

    gen = [s["gen"] for s in samples]
    ref = [s["ref"] for s in samples]
    num_samples = len(samples)

    # valid samples
    valid_samples = [(x, y) for (x, y) in zip(ref, gen) if y and mol_from_smiles(y)]
    num_valid = len(valid_samples)
    validity_rate = num_valid / num_samples

    # novel samples
    novel_samples = [y not in ref for (x, y) in valid_samples]
    novelty_rate = len(novel_samples) / num_valid

    # unique samples
    unique_samples = set([y for (x, y) in valid_samples])
    uniqueness_rate = len(unique_samples) / num_valid

    # success rate
    kw = SR_KWARGS[dataset_name].copy()
    similar = [is_similar(x, y, kw["similarity_thres"]) for (x, y) in valid_samples]
    improved = [is_improved(y, kw["prop_fun"], kw["improvement_thres"]) for (x, y) in valid_samples]
    success = [x and y for (x, y) in zip(similar, improved)]

    return {
        "num_samples": len(samples),
        "similar": sum(similar) / num_valid,
        "improved": sum(improved) / num_valid,
        "success_rate": sum(success) / num_valid,
        "valid": validity_rate,
        "unique": uniqueness_rate,
        "novel": novelty_rate,
        "scoring": dataset_name
    }


def convert_metrics_dict(metrics_dict):
    for k in metrics_dict.keys():
        metrics_dict[k] = float(metrics_dict[k])
    return metrics_dict


def moses_score(results_dir, epoch=1, n_jobs=40):
    results_dir = Path(results_dir)
    samples_dir = results_dir / "samples"
    samples_path = samples_dir / f"samples_{epoch}.yml"
    samples = load_yaml(samples_path)

    ref_samples = [s["ref"] for s in samples]
    gen_samples = [s["gen"] for s in samples]

    scores = get_all_metrics(gen_samples, n_jobs=n_jobs)
    return convert_metrics_dict(scores)