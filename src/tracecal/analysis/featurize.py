"""Conditioning features for the Mondrian cells (ours and the ablation/baseline feature sets)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundles import Bundle


@dataclass
class Aux:
    """White-box helpers fit on the fit split (all 9 pool versions; uses program labels, never class labels)."""

    pca64: object | None = None
    pca11: object | None = None
    shiftclf: object | None = None

    @classmethod
    def fit(cls, feats_fit: np.ndarray, version_fit: np.ndarray, seed: int = 0) -> Aux:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        pca64 = PCA(n_components=min(64, feats_fit.shape[1]), random_state=seed).fit(feats_fit)
        pca11 = PCA(n_components=11, random_state=seed).fit(feats_fit)
        z = pca64.transform(feats_fit)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit((z - z.mean(0)) / (z.std(0) + 1e-8), version_fit)
        aux = cls(pca64, pca11, clf)
        aux._mu, aux._sd = z.mean(0), z.std(0) + 1e-8
        return aux

    def emb64(self, feats: np.ndarray) -> np.ndarray:
        return self.pca64.transform(feats)  # type: ignore[union-attr]

    def shift_probs(self, feats: np.ndarray) -> np.ndarray:
        z = (self.pca64.transform(feats) - self._mu) / self._sd  # type: ignore[union-attr]
        return self.shiftclf.predict_proba(z)  # type: ignore[union-attr]


FEATURE_SETS = ("trace", "scalar_s", "conf", "embed", "shiftpred")


def features(b: Bundle, kind: str, aux: Aux | None = None) -> np.ndarray:
    base = [b.conf[:, None], b.entropy[:, None]]
    if kind == "trace":
        return np.concatenate([b.R] + base, axis=1)
    if kind == "trace_unsigned":
        return np.concatenate([b.KL] + base, axis=1)
    if kind == "scalar_s":
        return np.concatenate([b.s[:, None]] + base, axis=1)
    if kind == "conf":
        return np.concatenate(base, axis=1)
    if kind == "embed":
        assert aux is not None and b.feats is not None
        return np.concatenate([aux.pca11.transform(b.feats)] + base, axis=1)  # type: ignore[union-attr]
    if kind == "shiftpred":
        assert aux is not None and b.feats is not None
        return np.concatenate([aux.shift_probs(b.feats)] + base, axis=1)
    if kind.startswith("trace_sub:"):
        cols = [int(c) for c in kind.split(":", 1)[1].split(",")]
        return np.concatenate([b.R[:, cols]] + base, axis=1)
    raise ValueError(kind)
