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
    sim_mean, sim_std = np.mean(sims), np.std(sims)
    similar = [s >= kw["similarity_thres"] for s in sims]

    # property
    fun = kw["prop_fun"]

    # improvement
    gen_prop, ref_prop = [fun(g) for g in gen], [fun(r) for r in ref]
    gen_mean, gen_std = np.mean(gen_prop), np.std(gen_prop)
    impr = [g - r for (g, r) in zip(gen_prop, ref_prop)]
    impr_mean, impr_std = np.mean(impr), np.std(impr)
    improved = [fun(g) >= kw["improvement_thres"] for g in gen]

    # success
    success = [x and y for (x, y) in zip(similar, improved)]

    # reconstructed
    recon = [x == y for (x, y) in valid]

    return {
        "scoring": dataset_name,
        "num_samples": len(samples),
        "valid": f"{len(valid) / len(samples):.4f}",
        "unique": f"{len(unique) / len(valid):.4f}",
        "novel": f"{sum(novel) / len(valid):.4f}",
        "property": f"{gen_mean:.4f} +/- {gen_std:.4f}",
        "similar": f"{sum(similar) / len(similar):.4f}",
        "avg_similarity": f"{sim_mean:.4f} +/- {sim_std:.4f}",
        "improved": f"{sum(improved) / len(improved):.4f}",
        "avg_improvement": f"{impr_mean:.4f} +/- {impr_std:.4f}",
        "success_rate": f"{sum(success) / len(success):.4f}",
        "recon_rate": f"{sum(recon) / len(recon):.4f}"
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
