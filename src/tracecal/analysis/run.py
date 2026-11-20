"""Run every method for one (model, pool re-split, protocol): selection on synthetic tune conditions only, then
evaluation on all test conditions. Writes per-example coverage / set-size arrays for the statistics.

Protocols: "A" (all families in P'), "B:<category>" (leave-family-out: the families mapped to that corruption
category are removed from P' and from the tune conditions; evaluation restricted to that category).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import families, protocol
from ..data.splits import pool_resplit
from ..paths import get_paths
from ..utils.io import atomic_write_json
from .bundles import Bundle, load_bundle
from .featurize import Aux
from .methods import QTC, SAC, Ctx, Gate, KSAdjust, Mondrian, SplitCP, TTALearned, WeightedCP
from .pprime import _composed, assemble, assemble_view_logits, pprime_versions, single_family_severity

MONDRIAN_FEATS = ("trace", "scalar_s", "conf", "embed", "shiftpred")
GATES = ("trace_knn", "maxsoftmax", "energy", "embed_maha")


def dataset_family(dataset: str) -> str:
    return "cifar" if dataset.startswith("cifar") else "imagenet"


def test_conditions(dataset: str, category: str | None = None) -> list[tuple[str, str]]:
    corr = protocol()["corruptions"]
    names = corr["categories"][category] if category else corr["all"]
    if dataset in ("cifar10", "cifar100"):
        clean, cpref = f"{dataset}:test", f"{dataset}c"
    else:
        clean, cpref = "imagenet:test10k", "imagenetc"
    out = [] if category else [("clean", clean)]
    out += [(f"{c}:{s}", f"{cpref}:{c}:{s}") for c in names for s in corr["severities"]]
    return out


def allowed_families(proto: str) -> list[str]:
    fams = families()
    if proto == "A":
        return fams
    cat = proto.split(":", 1)[1]
    removed = set(protocol()["corruptions"]["lfo_removed_families"][cat])
    return [f for f in fams if f not in removed]


def mcs(cov: list[float], alpha: float) -> float:
    return float(np.mean([max(0.0, 1.0 - alpha - c) for c in cov]))


class PoolData:
    """All 9 pool versions of one model + the split positions of one re-split."""

    def __init__(self, model_tag: str, dataset: str, resplit: int, with_views: bool, allow_scaled: bool = False):
        self.model_tag, self.dataset, self.family = model_tag, dataset, dataset_family(dataset)
        src = f"{dataset}:pool"
        self.versions = [load_bundle(model_tag, src, v, with_feats=True) for v in range(len(families()) + 2)]
        ids0 = self.versions[0].ids
        for b in self.versions[1:]:
            if not np.array_equal(b.ids, ids0):
                raise RuntimeError("pool versions are not row-aligned")
        self.ids = ids0
        self.pos = pool_resplit(len(ids0), self.family, resplit, allow_scaled)
        self.with_views = with_views
        self._sev: dict[int, np.ndarray] = {}

    def severity(self, version: int) -> np.ndarray:
        """Severity of the single-family program of every cal-split image for ``version`` (cached)."""
        if version not in self._sev:
            self._sev[version] = single_family_severity(self.ids[self.pos["cal"]], version, self.family)
        return self._sev[version]

    def tune_conditions(self, allowed: list[str], subset=None) -> list[tuple[str, Bundle]]:
        fams = families()
        pos = self.pos["tune"]
        out = [("v0", self.versions[0].take(pos).restrict(subset))]
        for f in allowed:
            v = fams.index(f) + 1
            out.append((f"v{v}", self.versions[v].take(pos).restrict(subset)))
        ok = np.array([all(f in allowed for f in _composed(int(i), self.family)) for i in self.ids[pos]])
        if ok.any():
            out.append(("v8", self.versions[len(fams) + 1].take(pos[ok]).restrict(subset)))
        return out

    def ctx(self, p0: float, allowed: list[str], alpha: float, score: str, aux: Aux, subset=None,
            views: bool = False) -> Ctx:
        def pp(split):
            pos = self.pos[split]
            ver = pprime_versions(self.ids[pos], p0, allowed, self.family)
            return assemble(self.versions, pos, ver), ver, pos

        (pfit, _, _), (pcal, vcal, poscal), (ptune, vtune, postune) = pp("fit"), pp("cal"), pp("tune")
        clean = {k: self.versions[0].take(self.pos[k]) for k in ("fit", "cal", "tune")}
        min_cal = int(protocol()["methods"]["mondrian"]["min_cell_cal"][self.family])
        min_fit = int(np.ceil(min_cal * len(self.pos["fit"]) / len(self.pos["cal"])))
        ctx = Ctx(self.family, alpha, score, pfit.restrict(subset), pcal.restrict(subset), ptune.restrict(subset),
                  clean["fit"].restrict(subset), clean["cal"].restrict(subset), clean["tune"].restrict(subset), aux,
                  min_fit)
        if views:
            def vl(bund_versions, pos, ver):
                arr = assemble_view_logits(self.versions, pos, ver)
                if subset is not None:
                    keep = np.isin(self.versions[0].labels[pos], subset)
                    arr = arr[keep][:, :, subset]
                return arr

            ctx.pcal_views = vl(None, poscal, vcal)
            ctx.ptune_views = vl(None, postune, vtune)
            ctx.clean_cal_views = ctx.clean_cal.all_view_logits()
            ctx.clean_tune_views = ctx.clean_tune.all_view_logits()
        # SAC surrogate sets from the cal split: clean + family x severity quintile
        fams = families()
        sets = [("clean", clean["cal"].restrict(subset))]
        nb = int(protocol()["methods"]["sac"]["severity_bins"])
        for f in allowed:
            v = fams.index(f) + 1
            sev = self.severity(v)
            edges = np.quantile(sev, np.linspace(0, 1, nb + 1))
            binid = np.clip(np.searchsorted(edges, sev, side="right") - 1, 0, nb - 1)
            for k in range(nb):
                sel = self.pos["cal"][binid == k]
                if len(sel):
                    sets.append((f"{f}-q{k}", self.versions[v].take(sel).restrict(subset)))
        ctx.sac_sets = sets
        return ctx


def fit_aux(pd: PoolData, allowed: list[str]) -> Aux:
    """White-box helpers on the fit split, using ONLY the allowed families (no leakage of a held-out family)."""
    fams = families()
    fit_pos = pd.pos["fit"]
    feats, vers = [], []
    for v in [0] + [fams.index(f) + 1 for f in allowed]:
        feats.append(pd.versions[v].take(fit_pos).feats)
        vers.append(np.full(len(fit_pos), v))
    ok = np.array([all(f in allowed for f in _composed(int(i), pd.family)) for i in pd.ids[fit_pos]])
    if ok.any():
        feats.append(pd.versions[len(fams) + 1].take(fit_pos[ok]).feats)
        vers.append(np.full(int(ok.sum()), len(fams) + 1))
    return Aux.fit(np.concatenate(feats), np.concatenate(vers))


def _eval_on(method, conds: list[tuple[str, Bundle]], alpha: float) -> tuple[float, float]:
    covs, sizes = [], []
    for name, b in conds:
        r = method.predict(b, f"tune-{name}")
        covs.append(float(r["covered"].mean()))
        sizes.append(float(r["size"].mean()))
    return mcs(covs, alpha), float(np.mean(sizes))


def select(pd: PoolData, allowed: list[str], alpha: float, score: str, aux: Aux, subset=None) -> dict[str, Any]:
    """Pre-registered selection on the tune split's synthetic conditions (never a test shift)."""
    m = protocol()["methods"]
    tune = pd.tune_conditions(allowed, subset)
    log: dict[str, Any] = {"pprime_marginal": [], "mondrian": {}}
    best_pm = None
    for p0 in m["p0_grid"]:
        ctx = pd.ctx(p0, allowed, alpha, score, aux, subset)
        meth = SplitCP("pprime").fit(ctx)
        mc, sz = _eval_on(meth, tune, alpha)
        log["pprime_marginal"].append({"p0": p0, "mcs": mc, "size": sz})
        if best_pm is None or (mc, sz) < (best_pm["mcs"], best_pm["size"]):
            best_pm = {"p0": p0, "mcs": mc, "size": sz}
    assert best_pm is not None
    size_cap = 1.15 * best_pm["size"]
    chosen = {"pprime_marginal": best_pm, "size_cap": size_cap}
    ctxs = {p0: pd.ctx(p0, allowed, alpha, score, aux, subset) for p0 in m["p0_grid"]}
    for feat in MONDRIAN_FEATS:
        rows = []
        for p0 in m["p0_grid"]:
            for part in m["mondrian"]["partition_grid"]:
                for b in m["mondrian"]["B_grid"]:
                    meth = Mondrian(feat, part, b).fit(ctxs[p0])
                    mc, sz = _eval_on(meth, tune, alpha)
                    rows.append({"p0": p0, "partition": part, "B": b, "n_cells": int(meth.cells.n_final),
                                 "mcs": mc, "size": sz, "feasible": sz <= size_cap})
        feas = [r for r in rows if r["feasible"]]
        pool_rows = feas if feas else rows
        key = (lambda r: (r["mcs"], r["B"], r["partition"] != "kmeans", r["p0"])) if feas else \
              (lambda r: (r["size"], r["mcs"], r["B"]))
        chosen[f"mondrian_{feat}"] = dict(min(pool_rows, key=key), any_feasible=bool(feas))
        log["mondrian"][feat] = rows
    return {"chosen": chosen, "log": log}


def build_methods(pd: PoolData, sel: dict, allowed, alpha, score, aux, subset=None, with_tta=True):
    ch = sel["chosen"]
    p0_pm = ch["pprime_marginal"]["p0"]
    ctx_pm = pd.ctx(p0_pm, allowed, alpha, score, aux, subset, views=with_tta)
    eta = float(protocol()["methods"]["ks_adjust"]["eta"])
    methods = [SplitCP("clean").fit(ctx_pm), SplitCP("pprime").fit(ctx_pm), QTC().fit(ctx_pm), SAC().fit(ctx_pm),
               KSAdjust(eta).fit(ctx_pm), WeightedCP("embed").fit(ctx_pm), WeightedCP("trace").fit(ctx_pm)]
    if with_tta:
        methods += [TTALearned("clean", True).fit(ctx_pm), TTALearned("pprime", True).fit(ctx_pm),
                    TTALearned("clean", False).fit(ctx_pm)]
    ctx_by_p0: dict[float, Ctx] = {p0_pm: ctx_pm}
    for feat in MONDRIAN_FEATS:
        c = ch[f"mondrian_{feat}"]
        if c["p0"] not in ctx_by_p0:
            ctx_by_p0[c["p0"]] = pd.ctx(c["p0"], allowed, alpha, score, aux, subset)
        methods.append(Mondrian(feat, c["partition"], c["B"]).fit(ctx_by_p0[c["p0"]]))
    trace_ctx = ctx_by_p0[ch["mondrian_trace"]["p0"]]
    gates = [Gate(g).fit(trace_ctx) for g in GATES]
    return methods, gates


def run(model_tag: str, dataset: str, resplit: int, proto: str = "A", alpha: float | None = None,
        score: str | None = None, out_root: Path | None = None, with_tta: bool = True,
        naturals: bool = False, conditions: list[tuple[str, str]] | None = None,
        allow_scaled: bool = False) -> Path:
    t0 = time.time()
    alpha = float(protocol()["alpha_primary"] if alpha is None else alpha)
    score = score or protocol()["methods"]["score_primary"]
    tag = f"{proto.replace(':', '-')}_a{alpha:g}_{score}"
    out = (out_root or get_paths().raw_results_root / "runs") / model_tag / f"r{resplit}" / tag
    if (out / ".ok").exists():
        print(f"skip (done): {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    pd = PoolData(model_tag, dataset, resplit, with_views=with_tta, allow_scaled=allow_scaled)
    allowed = allowed_families(proto)
    aux = fit_aux(pd, allowed)
    sel = select(pd, allowed, alpha, score, aux)
    atomic_write_json(out / "selection.json", sel)
    methods, gates = build_methods(pd, sel, allowed, alpha, score, aux, with_tta=with_tta)
    category = None if proto == "A" else proto.split(":", 1)[1]
    conds = conditions if conditions is not None else test_conditions(dataset, category)
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"methods": {}, "gates": {}, "conditions": [c for c, _ in conds]}
    for cname, src in conds:
        test = load_bundle(model_tag, src, 0, with_feats=True, cache=False)
        for m in methods:
            r = m.predict(test, cname)
            arrays[f"{m.name}|{cname}|cov"] = r["covered"].astype(bool)
            arrays[f"{m.name}|{cname}|size"] = r["size"].astype(np.int16)
            summary["methods"].setdefault(m.name, {})[cname] = {
                "coverage": float(r["covered"].mean()), "size": float(r["size"].mean())}
            if m.name == "mondrian_trace":
                arrays[f"{m.name}|{cname}|cell"] = r["cell"].astype(np.int16)
        for g in gates:
            p = g.pvalues(test).astype(np.float32)
            arrays[f"gate_{g.kind}|{cname}|p"] = p
        test.meta = {}
    if naturals and dataset == "imagenet" and proto == "A":
        _run_naturals(pd, sel, allowed, alpha, score, aux, arrays, summary, with_tta)
    np.savez_compressed(out / "arrays.npz", **arrays)
    summary.update({"model": model_tag, "dataset": dataset, "resplit": resplit, "protocol": proto,
                    "alpha": alpha, "score": score, "seconds": time.time() - t0,
                    "n_cells_trace": int(next(m for m in methods if m.name == "mondrian_trace").cells.n_final)})
    atomic_write_json(out / "summary.json", summary)
    (out / ".ok").write_text("ok")
    print(f"done {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _run_naturals(pd, sel, allowed, alpha, score, aux, arrays, summary, with_tta):
    from ..data.imagenet import natural_index

    for name in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
        subset = natural_index(name)["class_subset"]
        methods, gates = build_methods(pd, sel, allowed, alpha, score, aux, subset=subset, with_tta=with_tta)
        test = load_bundle(pd.model_tag, name, 0, with_feats=True, cache=False).restrict(subset)
        for m in methods:
            r = m.predict(test, name)
            arrays[f"{m.name}|{name}|cov"] = r["covered"].astype(bool)
            arrays[f"{m.name}|{name}|size"] = r["size"].astype(np.int16)
            summary["methods"].setdefault(m.name, {})[name] = {
                "coverage": float(r["covered"].mean()), "size": float(r["size"].mean())}
        for g in gates:
            arrays[f"gate_{g.kind}|{name}|p"] = g.pvalues(test).astype(np.float32)
