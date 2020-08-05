import numpy as np
import pandas as pd

import torch

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from scipy.spatial.kdtree import KDTree

from core.mols.props import qed, penalized_logp


def get_similar(emb, vocab, idx, num_similar=3):
    tree = KDTree(emb)
    e = emb[idx, :]
    idxs = (tree.query(e, k=num_similar)[1]).tolist()
    smiles = [vocab[i] for i in idxs]
    return idxs, ".".join(smiles)


def dotprod(emb, i, j):
    emb = emb[4:, :]
    point1 = emb[i,:]
    point2 = emb[j,:]
    return np.dot(point1, point2)


def visualize_emb(emb, vocab, num_points=5000):
    X = emb[4:, :num_points]
    X = StandardScaler().fit_transform(X)
    X = UMAP().fit_transform(X)

    qeds = [qed(vocab[i]) for i in range(X.shape[0])]

    cmap = sns.cubehelix_palette(as_cmap=True, n_colors=5)
    plt.scatter(X[:, 0], X[:, 1], c=qeds, s=5, cmap=cmap)
    plt.show()


def visualize_covid(emb, vocab, acts):
    X = emb[4:, :]
    X = StandardScaler().fit_transform(X)
    X = UMAP().fit_transform(X)

    cs = ["blue"] * X.shape[0]
    for a in acts:
        cs[a] = "red"

    plt.scatter(X[:, 0], X[:, 1], c=cs, s=5)
    plt.show()