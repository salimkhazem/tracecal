"""Label-free (k-means) and score-supervised (regression tree) partitions of a feature space.

Partitions are fit on the *fit* split only, so the Mondrian categories never depend on the calibration split
(category-conditional validity then follows from exchangeability within the calibration + test points).
A minimum cell size (on the fit split) is enforced by merging the smallest cell into its nearest neighbour.
"""

from __future__ import annotations

import numpy as np


class Standardizer:
    def fit(self, z: np.ndarray) -> Standardizer:
        self.mu = z.mean(0)
        self.sd = z.std(0) + 1e-8
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mu) / self.sd


class KMeansCells:
    def __init__(self, n_cells: int, min_fit: int, seed: int = 0):
        self.n_cells, self.min_fit, self.seed = int(n_cells), int(min_fit), int(seed)

    def fit(self, z_fit: np.ndarray, y_fit: np.ndarray | None = None) -> KMeansCells:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=self.n_cells, n_init=10, random_state=self.seed).fit(z_fit)
        self.centers = km.cluster_centers_
        orig = km.labels_
        group = np.arange(self.n_cells)
        while True:
            g = group[orig]
            ids, counts = np.unique(g, return_counts=True)
            if len(ids) <= 1 or counts.min() >= self.min_fit:
                break
            small = ids[np.argmin(counts)]
            cents = {i: z_fit[g == i].mean(0) for i in ids}
            others = [i for i in ids if i != small]
            dist = [np.linalg.norm(cents[small] - cents[o]) for o in others]
            group[group == small] = others[int(np.argmin(dist))]
        uniq = np.unique(group)
        self.remap = {int(u): i for i, u in enumerate(uniq)}
        self.group = np.array([self.remap[int(group[i])] for i in range(self.n_cells)])
        self.n_final = len(uniq)
        return self

    def assign(self, z: np.ndarray) -> np.ndarray:
        d = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return self.group[np.argmin(d, axis=1)]


class TreeCells:
    def __init__(self, n_cells: int, min_fit: int, seed: int = 0):
        self.n_cells, self.min_fit, self.seed = int(n_cells), int(min_fit), int(seed)

    def fit(self, z_fit: np.ndarray, y_fit: np.ndarray | None = None) -> TreeCells:
        from sklearn.tree import DecisionTreeRegressor

        if y_fit is None:
            raise ValueError("tree cells need the fit-split conformal scores")
        self.tree = DecisionTreeRegressor(max_leaf_nodes=self.n_cells, min_samples_leaf=self.min_fit,
                                          random_state=self.seed).fit(z_fit, y_fit)
        leaves = np.unique(self.tree.apply(z_fit))
        self.remap = {int(v): i for i, v in enumerate(leaves)}
        self.n_final = len(leaves)
        return self

    def assign(self, z: np.ndarray) -> np.ndarray:
        leaf = self.tree.apply(z)
        return np.array([self.remap[int(v)] for v in leaf])


def make_cells(kind: str, n_cells: int, min_fit: int, seed: int = 0):
    if kind == "kmeans":
        return KMeansCells(n_cells, min_fit, seed)
    if kind == "tree":
        return TreeCells(n_cells, min_fit, seed)
    raise ValueError(kind)
