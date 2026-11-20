"""Conformal outlier p-values (Bates et al. 2023) for the support gate.

The outlier score A is fit on the *fit* split (k-NN mean distance in standardised feature space, or a scalar score
such as -max softmax); p(x) = (1 + #{j in cal : A_j >= A(x)}) / (n_cal + 1). If the test point is exchangeable with
the calibration points, P(p(x) <= beta) <= beta.
"""

from __future__ import annotations

import numpy as np


class KNNScore:
    def __init__(self, k: int = 10):
        self.k = int(k)

    def fit(self, z_fit: np.ndarray) -> KNNScore:
        from sklearn.neighbors import NearestNeighbors

        self.nn = NearestNeighbors(n_neighbors=self.k).fit(z_fit)
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(z, n_neighbors=self.k)
        return d.mean(1)


class MahalanobisScore:
    def fit(self, z_fit: np.ndarray) -> MahalanobisScore:
        self.mu = z_fit.mean(0)
        cov = np.cov(z_fit, rowvar=False) + 1e-4 * np.eye(z_fit.shape[1])
        self.prec = np.linalg.inv(cov)
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        d = z - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


def conformal_pvalues(a_cal: np.ndarray, a_test: np.ndarray) -> np.ndarray:
    s = np.sort(np.asarray(a_cal, dtype=np.float64))
    n = s.size
    n_ge = n - np.searchsorted(s, np.asarray(a_test, dtype=np.float64), side="left")
    return (1.0 + n_ge) / (n + 1.0)


def energy(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Free energy -T logsumexp(z/T); larger = more outlying."""
    z = logits / t
    m = z.max(1, keepdims=True)
    return -t * (m[:, 0] + np.log(np.exp(z - m).sum(1)))
