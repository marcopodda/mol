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

    ref = [s["ref"] for s in samples]
    gen = [s["gen"] for s in samples]

    # valid samples
    valid = [(x, y) for (x, y) in zip(ref, gen) if y and mol_from_smiles(y)]
    ref, gen = zip(*valid)

    # novel samples
    data, _, _ = load_data(dataset_name)
    training_set = set(data[data.is_train == True].smiles.tolist())
    novel = [g not in training_set for g in gen]

    # unique samples
    unique = set(gen)

    # similarity
    kw = SR_KWARGS[dataset_name].copy()
    sims = [similarity(x, y) for (x, y) in valid]
    similar = [s >= kw["similarity_thres"] for s in sims]

    # property
    fun = kw["prop_fun"]

    # improvement
    impr = [fun(g) - fun(r) for (r, g) in zip(ref, gen)]
    improved = [fun(g) >= kw["improvement_thres"] for g in gen]

    # success
    success = [x and y for (x, y) in zip(similar, improved)]

    # reconstructed
    recon = [x == y for (x, y) in valid]

    return {
        "scoring": dataset_name,
        "num_samples": len(samples),
        "valid": len(valid) / len(samples),
        "unique": len(unique) / len(valid),
        "novel": sum(novel) / len(valid),
        "property": f"{np.mean(gen)} +/- {np.std(gen)}",
        "similar": sum(similar) / len(similar),
        "avg_similarity": f"{np.mean(sims)} +/- {np.std(sims)}",
        "improved": sum(improved) / len(improved),
        "avg_improvement": f"{np.mean(impr)} +/- {np.std(impr)}",
        "success_rate": sum(success) / len(success),
        "recon_rate": sum(recon) / len(recon),
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
