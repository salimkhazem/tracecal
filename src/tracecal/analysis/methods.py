"""All conformal methods behind one interface.

``fit(ctx)`` sees only calibration-pool data (fit / tune / cal splits); ``predict(test)`` returns per-example
(covered, size, threshold). Transductive baselines additionally use the *unlabeled* test condition inside
``predict`` (QTC, SAC, KS-adjust, weighted CP); they never see test labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..conformal.core import conformal_quantile, evaluate_sets, mondrian_thresholds, score_fns, weighted_thresholds
from ..gate.pvalues import KNNScore, MahalanobisScore, conformal_pvalues, energy
from ..partition.cells import Standardizer, make_cells
from ..utils.seed import derive_seed
from .bundles import Bundle, softmax
from .featurize import Aux, features


@dataclass
class Ctx:
    """Calibration-side data for one (model, re-split, protocol, p0)."""

    family: str
    alpha: float
    score: str
    pfit: Bundle            # P' rows of the fit split
    pcal: Bundle            # P' rows of the cal split
    ptune: Bundle           # P' rows of the tune split
    clean_fit: Bundle
    clean_cal: Bundle
    clean_tune: Bundle
    aux: Aux | None
    min_fit: int
    sac_sets: list = field(default_factory=list)       # list of (name, Bundle) surrogate sets from the cal split
    pcal_views: np.ndarray | None = None               # (n_cal, V+1, C') view logits of P' cal (TTA)
    ptune_views: np.ndarray | None = None
    clean_cal_views: np.ndarray | None = None
    clean_tune_views: np.ndarray | None = None
    raps: tuple = (0.0, 0)                             # (lam, k_reg)


def _u(n: int, key: str) -> np.ndarray:
    return np.random.default_rng(derive_seed(0, "aps-u", key, n)).random(n)


def scores_of(b: Bundle, score: str, key: str, raps=(0.0, 0), probs: np.ndarray | None = None):
    lam, k_reg = raps
    all_fn, true_fn = score_fns(score, lam, k_reg)
    p = b.probs if probs is None else probs
    u = _u(len(b), key)
    return all_fn(p, u), true_fn(p, b.labels, u)


class Method:
    name = "base"
    transductive = False

    def fit(self, ctx: Ctx) -> Method:
        raise NotImplementedError

    def predict(self, test: Bundle, key: str) -> dict:
        raise NotImplementedError


class SplitCP(Method):
    """Split CP with a single threshold calibrated on clean data ('clean') or on P' ('pprime')."""

    def __init__(self, cal: str):
        self.cal = cal
        self.name = "split_clean" if cal == "clean" else "pprime_marginal"

    def fit(self, ctx: Ctx) -> SplitCP:
        b = ctx.clean_cal if self.cal == "clean" else ctx.pcal
        _, t = scores_of(b, ctx.score, f"cal-{self.cal}", ctx.raps)
        self.q = conformal_quantile(t, ctx.alpha)
        self.ctx = ctx
        return self

    def predict(self, test: Bundle, key: str) -> dict:
        a, _ = scores_of(test, self.ctx.score, key, self.ctx.raps)
        cov, size = evaluate_sets(a, test.labels, self.q)
        return {"covered": cov, "size": size, "q": np.full(len(test), self.q)}


class Mondrian(Method):
    """Mondrian split CP on cells of a feature space fit on P' (TRACE = feature set 'trace')."""

    def __init__(self, feat: str, partition: str, n_cells: int):
        self.feat, self.partition, self.n_cells = feat, partition, int(n_cells)
        self.name = f"mondrian_{feat}"

    def fit(self, ctx: Ctx) -> Mondrian:
        self.ctx = ctx
        zf = features(ctx.pfit, self.feat, ctx.aux)
        self.std = Standardizer().fit(zf)
        _, tf = scores_of(ctx.pfit, ctx.score, "fit", ctx.raps)
        self.cells = make_cells(self.partition, self.n_cells, ctx.min_fit).fit(self.std(zf), tf)
        zc = self.std(features(ctx.pcal, self.feat, ctx.aux))
        cc = self.cells.assign(zc)
        _, tc = scores_of(ctx.pcal, ctx.score, "cal-pprime", ctx.raps)
        self.q = mondrian_thresholds(cc, tc, self.cells.n_final, ctx.alpha)
        self.cal_counts = np.bincount(cc, minlength=self.cells.n_final)
        return self

    def cell_of(self, test: Bundle) -> np.ndarray:
        return self.cells.assign(self.std(features(test, self.feat, self.ctx.aux)))

    def predict(self, test: Bundle, key: str) -> dict:
        cells = self.cell_of(test)
        q = self.q[cells]
        a, _ = scores_of(test, self.ctx.score, key, self.ctx.raps)
        cov, size = evaluate_sets(a, test.labels, q)
        return {"covered": cov, "size": size, "q": q, "cell": cells}


class TTALearned(Method):
    """Shanmugam et al. (CVPR'25): learned logit weights over the 13 views (base + 11 probes + hflip), learned on
    the tune split by cross-entropy, then split CP on the aggregated probabilities (cal = clean or P')."""

    def __init__(self, cal: str, learned: bool = True):
        self.cal, self.learned = cal, learned
        self.name = f"tta_{'learned' if learned else 'avg'}_{cal}"

    def fit(self, ctx: Ctx) -> TTALearned:
        from scipy.optimize import minimize

        self.ctx = ctx
        tune_v = ctx.clean_tune_views if self.cal == "clean" else ctx.ptune_views
        tune_b = ctx.clean_tune if self.cal == "clean" else ctx.ptune
        cal_v = ctx.clean_cal_views if self.cal == "clean" else ctx.pcal_views
        cal_b = ctx.clean_cal if self.cal == "clean" else ctx.pcal
        m = tune_v.shape[1]
        theta0 = np.full(m, 1.0 / m)
        if self.learned:
            y = tune_b.labels

            def loss_grad(theta):
                z = np.einsum("nmc,m->nc", tune_v, theta)
                z = z - z.max(1, keepdims=True)
                lse = np.log(np.exp(z).sum(1))
                p = np.exp(z - lse[:, None])
                loss = float(np.mean(lse - z[np.arange(len(y)), y]))
                g = p.copy()
                g[np.arange(len(y)), y] -= 1.0
                grad = np.einsum("nc,nmc->m", g, tune_v) / len(y)
                return loss, grad

            res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
            self.theta = res.x
        else:
            self.theta = theta0
        probs = softmax(np.einsum("nmc,m->nc", cal_v, self.theta))
        _, t = scores_of(cal_b, ctx.score, f"cal-tta-{self.cal}", ctx.raps, probs=probs)
        self.q = conformal_quantile(t, ctx.alpha)
        return self

    def predict(self, test: Bundle, key: str) -> dict:
        probs = softmax(np.einsum("nmc,m->nc", test.all_view_logits(), self.theta))
        a, _ = scores_of(test, self.ctx.score, key, self.ctx.raps, probs=probs)
        cov, size = evaluate_sets(a, test.labels, self.q)
        return {"covered": cov, "size": size, "q": np.full(len(test), self.q)}


class QTC(Method):
    """Yilmaz & Heckel: beta = min(beta_T, beta_S) from max-softmax quantiles; recalibrate on clean cal at beta."""

    name = "qtc"
    transductive = True

    def fit(self, ctx: Ctx) -> QTC:
        self.ctx = ctx
        self.src_conf = ctx.clean_cal.conf
        _, self.src_t = scores_of(ctx.clean_cal, ctx.score, "cal-clean", ctx.raps)
        return self

    @staticmethod
    def _q(s: np.ndarray, c: float) -> float:
        """inf{p : mean(1{s < p}) >= c} = the ceil(c n)-th smallest value."""
        srt = np.sort(s)
        k = max(1, math.ceil(c * len(srt)))
        return float(srt[min(k, len(srt)) - 1])

    def predict(self, test: Bundle, key: str) -> dict:
        a = self.ctx.alpha
        tgt = test.conf
        beta_t = float(np.mean(self.src_conf < self._q(tgt, a)))
        beta_s = 1.0 - float(np.mean(tgt < self._q(self.src_conf, 1.0 - a)))
        beta = min(beta_t, beta_s)
        q = conformal_quantile(self.src_t, beta) if beta > 0 else math.inf
        al, _ = scores_of(test, self.ctx.score, key, self.ctx.raps)
        cov, size = evaluate_sets(al, test.labels, q)
        return {"covered": cov, "size": size, "q": np.full(len(test), q), "beta": beta}


class SAC(Method):
    """Salvador et al. 2021 adapted to CP: pick the surrogate set (family x severity quintile, or clean) whose mean
    max-softmax is closest to the unlabeled target's; split CP threshold on that set."""

    name = "sac"
    transductive = True

    def fit(self, ctx: Ctx) -> SAC:
        self.ctx = ctx
        self.sets = []
        for nm, b in ctx.sac_sets:
            _, t = scores_of(b, ctx.score, f"sac-{nm}", ctx.raps)
            self.sets.append((nm, float(b.conf.mean()), conformal_quantile(t, ctx.alpha)))
        return self

    def predict(self, test: Bundle, key: str) -> dict:
        mu = float(test.conf.mean())
        nm, _, q = min(self.sets, key=lambda s: abs(s[1] - mu))
        a, _ = scores_of(test, self.ctx.score, key, self.ctx.raps)
        cov, size = evaluate_sets(a, test.labels, q)
        return {"covered": cov, "size": size, "q": np.full(len(test), q), "chosen": nm}


class KSAdjust(Method):
    """Doula 2026 (TCC-KS) adapted: P' cal plays the transported calibration; surrogate T = 1 - max prob;
    delta+ = sup(F_cal - F_tgt)_+ + DKW terms; alpha* = max(0, alpha - delta+); 2-fold cross-fitting on the target."""

    name = "ks_adjust"
    transductive = True

    def __init__(self, eta: float = 0.10):
        self.eta = float(eta)

    def fit(self, ctx: Ctx) -> KSAdjust:
        self.ctx = ctx
        self.t_cal = np.sort(1.0 - ctx.pcal.conf)
        _, self.cal_scores = scores_of(ctx.pcal, ctx.score, "cal-pprime", ctx.raps)
        return self

    def _delta_plus(self, t_tgt: np.ndarray) -> float:
        t_tgt = np.sort(t_tgt)
        grid = np.concatenate([self.t_cal, t_tgt])
        f_cal = np.searchsorted(self.t_cal, grid, side="right") / len(self.t_cal)
        f_tgt = np.searchsorted(t_tgt, grid, side="right") / len(t_tgt)
        d_hat = max(0.0, float(np.max(f_cal - f_tgt)))
        lg = math.log(4.0 / self.eta)
        return d_hat + math.sqrt(lg / (2 * len(t_tgt))) + math.sqrt(lg / (2 * len(self.t_cal)))

    def predict(self, test: Bundle, key: str) -> dict:
        n = len(test)
        perm = np.random.default_rng(derive_seed(0, "ks", key)).permutation(n)
        halves = [perm[: n // 2], perm[n // 2 :]]
        q = np.empty(n)
        t = 1.0 - test.conf
        for h in range(2):
            a_star = max(0.0, self.ctx.alpha - self._delta_plus(t[halves[1 - h]]))
            q[halves[h]] = conformal_quantile(self.cal_scores, a_star) if a_star > 0 else math.inf
        a, _ = scores_of(test, self.ctx.score, key, self.ctx.raps)
        cov, size = evaluate_sets(a, test.labels, q)
        return {"covered": cov, "size": size, "q": q}


class WeightedCP(Method):
    """Tibshirani et al. 2019 with domain-classifier likelihood ratios (P' fit vs target), 2-fold cross-fitting."""

    transductive = True

    def __init__(self, feat: str):
        self.feat = feat
        self.name = f"weighted_{feat}"

    def _z(self, b: Bundle) -> np.ndarray:
        if self.feat == "embed":
            return self.ctx.aux.emb64(b.feats)  # type: ignore[union-attr]
        return features(b, "trace")

    def fit(self, ctx: Ctx) -> WeightedCP:
        self.ctx = ctx
        self.z_src = self._z(ctx.pfit)
        self.z_cal = self._z(ctx.pcal)
        _, self.cal_scores = scores_of(ctx.pcal, ctx.score, "cal-pprime", ctx.raps)
        return self

    def predict(self, test: Bundle, key: str) -> dict:
        from sklearn.linear_model import LogisticRegression

        z_t = self._z(test)
        n = len(test)
        perm = np.random.default_rng(derive_seed(0, "wcp", key)).permutation(n)
        halves = [perm[: n // 2], perm[n // 2 :]]
        q = np.empty(n)
        for h in range(2):
            train_t = z_t[halves[1 - h]]
            x = np.concatenate([self.z_src, train_t])
            y = np.r_[np.zeros(len(self.z_src)), np.ones(len(train_t))]
            std = Standardizer().fit(x)
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(std(x), y)
            ratio = len(self.z_src) / len(train_t)

            def w(z):
                lo = clf.decision_function(std(z))
                return np.exp(np.clip(lo, -30, 30)) * ratio

            q[halves[h]] = weighted_thresholds(self.cal_scores, w(self.z_cal), w(z_t[halves[h]]), self.ctx.alpha)
        a, _ = scores_of(test, self.ctx.score, key, self.ctx.raps)
        cov, size = evaluate_sets(a, test.labels, q)
        return {"covered": cov, "size": size, "q": q}


# ----------------------------------------------------------------------------------------------------------------
# gate (support test)
# ----------------------------------------------------------------------------------------------------------------
class Gate:
    """Conformal p-values of an outlier score fit on P' fit, referenced to P' cal. kind in
    {trace_knn, maxsoftmax, energy, embed_maha}."""

    def __init__(self, kind: str, k: int = 10):
        self.kind, self.k = kind, int(k)

    def _a(self, b: Bundle) -> np.ndarray:
        if self.kind == "trace_knn":
            return self.score(self.std(features(b, "trace")))
        if self.kind == "maxsoftmax":
            return -b.conf
        if self.kind == "energy":
            return energy(b.logits)
        if self.kind == "embed_maha":
            return self.score(self.std(self.ctx.aux.emb64(b.feats)))  # type: ignore[union-attr]
        raise ValueError(self.kind)

    def fit(self, ctx: Ctx) -> Gate:
        self.ctx = ctx
        if self.kind == "trace_knn":
            z = features(ctx.pfit, "trace")
            self.std = Standardizer().fit(z)
            self.score = KNNScore(self.k).fit(self.std(z))
        elif self.kind == "embed_maha":
            z = ctx.aux.emb64(ctx.pfit.feats)  # type: ignore[union-attr]
            self.std = Standardizer().fit(z)
            self.score = MahalanobisScore().fit(self.std(z))
        self.a_cal = self._a(ctx.pcal)
        return self

    def pvalues(self, test: Bundle) -> np.ndarray:
        return conformal_pvalues(self.a_cal, self._a(test))
