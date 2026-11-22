#!/usr/bin/env python3
"""Classical benchmark runner with an explicit SEARCH / FINAL split.

SEARCH (once per model, seed 42): evaluate the hyperparameter grid on val only,
freeze the winner to <model>/results/<dataset>/best_config.json.

FINAL (seeds 42..46): load the frozen config and train exactly once per seed.
Seeds change only init/shuffle -- NEVER lr/C/feature. This keeps seed variance
separate from tuning variance (required for a publishable benchmark).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import load_records, to_arrays  # noqa: E402
from common.io import (  # noqa: E402
    RunLogger,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_json,
    sha256_text,
    utcnow,
    write_status,
)
from common.metrics import compute_metrics  # noqa: E402
from common.paths import CACHE, SPLITS, model_dir  # noqa: E402
from common.runner import Timer, set_seed  # noqa: E402

GRIDS: dict[str, list[dict]] = {
    "dummy": [{}],
    "multinomial_nb": [{"alpha": a} for a in (0.1, 0.5, 1.0)],
    "logistic_regression": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "linear_svc": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "svm_rbf": [{"C": c, "gamma": g} for c in (1, 10) for g in ("scale", 1e-3)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": l} for d in (None, 64) for l in (1, 3)],
    "extra_trees": [{"max_depth": d, "min_samples_leaf": l} for d in (None, 64) for l in (1, 3)],
    "hist_gradient_boosting": [{"learning_rate": lr, "max_iter": mi} for lr in (0.05, 0.1) for mi in (200, 400)],
    "knn": [{"n_neighbors": k} for k in (5, 15, 30)],
    "soft_voting": [{}],
    "xgboost": [
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    ],
    "lightgbm": [
        {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 300},
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300},
    ],
}

FEATURES = ("word11", "word12", "char25", "union")


def make_vectorizer(feature: str):
    if feature == "word11":
        return TfidfVectorizer(min_df=2, sublinear_tf=True)
    if feature == "word12":
        return TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))
    if feature == "char25":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               sublinear_tf=True, max_features=400000)
    if feature == "union":
        return FeatureUnion([
            ("w", TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                  sublinear_tf=True, max_features=400000)),
        ])
    raise ValueError(feature)


def build_features(texts: dict, feature: str, tag: str):
    key = sha256_json({"feature": feature, "tag": tag,
                       "n": {s: len(texts[s]) for s in SPLITS},
                       "train_hash": sha256_text("\n".join(texts["train"]))[:16]})
    d = CACHE / "features" / key
    files = {s: d / f"{s}.npz" for s in SPLITS}
    if all(f.exists() for f in files.values()) and (d / "meta.json").exists():
        return {s: sparse.load_npz(f) for s, f in files.items()}, load_json(d / "meta.json"), True
    vec = make_vectorizer(feature)
    X = {"train": vec.fit_transform(texts["train"])}
    X["val"] = vec.transform(texts["val"])
    X["test"] = vec.transform(texts["test"])
    d.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        sparse.save_npz(files[s], X[s])
    meta = {"feature": feature, "n_features": int(X["train"].shape[1])}
    atomic_write_json(d / "meta.json", meta)
    return X, meta, False


def _hgb(params: dict, seed: int):
    """HistGradientBoosting needs dense input; reduce to top-k chi2 features first."""
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline([
        ("select", SelectKBest(chi2, k=2000)),
        ("dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
        ("hgb", HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"], max_iter=params["max_iter"], random_state=seed)),
    ])


def build_model(name: str, params: dict, seed: int):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "multinomial_nb":
        return MultinomialNB(alpha=params.get("alpha", 1.0))
    if name == "logistic_regression":
        return LogisticRegression(C=params["C"], max_iter=2000, random_state=seed)
    if name == "linear_svc":
        return CalibratedClassifierCV(LinearSVC(C=params["C"], max_iter=5000, random_state=seed),
                                      cv=3, method="sigmoid")
    if name == "svm_rbf":
        return SVC(C=params["C"], gamma=params["gamma"], probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"],
                                      n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=params["max_depth"],
                                    min_samples_leaf=params["min_samples_leaf"],
                                    n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return _hgb(params, seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"], metric="cosine", n_jobs=-1)
    if name == "soft_voting":
        return VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
                ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)),
                ("et", ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed)),
                ("hgb", _hgb({"learning_rate": 0.1, "max_iter": 200}, seed)),
            ], voting="soft", n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=0.9, colsample_bytree=0.8,
            tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=params["n_estimators"], num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"], n_jobs=-1, random_state=seed, verbose=-1)
    raise ValueError(name)


def fit_and_score(name, params, seed, X, y):
    clf = build_model(name, params, seed).fit(X["train"], y["train"])
    val_pred = clf.predict(X["val"])
    val_f1 = float(f1_score(y["val"], val_pred, average="macro", zero_division=0))
    return clf, val_f1


def main(default_model: str | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model, required=default_model is None)
    ap.add_argument("--dataset", default="CMTE-NITANP")
    ap.add_argument("--split-name", default="B", choices=("A", "B"))
    ap.add_argument("--text-field", default="text_clean")
    ap.add_argument("--feature", choices=FEATURES, default="word11")
    ap.add_argument("--phase", choices=("search", "final"), required=True)
    ap.add_argument("--config", default=None, help="results subfolder (defaults to phase name)")
    ap.add_argument("--best-config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config = args.config or args.phase
    model_root = model_dir("classical", args.model)
    best_path = args.best_config or (model_root / "results" / args.dataset / "best_config.json")
    rd = model_root / "results" / args.dataset / config / f"seed{args.seed}"

    if (rd / "run_status.json").exists() and not args.force:
        if load_json(rd / "run_status.json").get("status") == "done":
            print(f"[skip] {args.model} phase={args.phase} seed={args.seed} already done")
            return

    set_seed(args.seed)
    rd.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    with RunLogger(rd) as log:
        write_status(rd, "running", model=args.model, phase=args.phase, seed=args.seed)
        try:
            recs = load_records(args.dataset, args.split_name)
            arr = {s: to_arrays(recs[s], args.text_field) for s in SPLITS}
            texts = {s: arr[s][0] for s in SPLITS}
            y = {s: arr[s][1] for s in SPLITS}
            ids = {s: arr[s][3] for s in SPLITS}

            if args.phase == "final":
                if not best_path.exists():
                    raise FileNotFoundError(f"no frozen config at {best_path}; run --phase search first")
                frozen = load_json(best_path)
                feature = frozen["feature"]
                params = frozen["params"]
                log.log(f"FINAL: frozen params={params} feature={feature} "
                        f"from val_f1={frozen['val_macro_f1']:.4f}")
            else:
                feature = args.feature
                params = None

            X, meta, cached = build_features(texts, feature,
                                             f"{args.dataset}:{args.split_name}:{args.text_field}")
            log.log(f"features={feature} n_features={meta['n_features']:,} cached={cached}")

            history, best_params, clf = [], None, None
            if args.phase == "search":
                for candidate in GRIDS[args.model]:
                    clf_i, val_f1 = fit_and_score(args.model, candidate, args.seed, X, y)
                    history.append({"params": candidate, "val_macro_f1": round(val_f1, 4)})
                    log.log(f"  config {candidate} -> val_macro_f1={val_f1:.4f}")
                # choose the winner explicitly (val only)
                best_i = int(np.argmax([h["val_macro_f1"] for h in history]))
                best_params = history[best_i]["params"]
                clf, best_val = fit_and_score(args.model, best_params, args.seed, X, y)
                frozen = {
                    "model": args.model, "dataset": args.dataset, "split_name": args.split_name,
                    "text_field": args.text_field, "feature": feature, "params": best_params,
                    "val_macro_f1": round(best_val, 4), "search_seed": args.seed,
                    "grid": GRIDS[args.model], "frozen_at": utcnow(),
                }
                frozen["hash"] = sha256_json({k: frozen[k] for k in ("feature", "params")})
                atomic_write_json(best_path, frozen)
                log.log(f"FROZE {best_path.name}: params={best_params} val_f1={best_val:.4f} hash={frozen['hash'][:12]}")
            else:
                clf, best_val = fit_and_score(args.model, params, args.seed, X, y)
                best_params = params

            val_proba = clf.predict_proba(X["val"])
            test_proba = clf.predict_proba(X["test"])
            test_pred = test_proba.argmax(1)
            metrics = compute_metrics(y["test"], test_pred, probs=test_proba)
            val_metrics = compute_metrics(y["val"], val_proba.argmax(1), probs=val_proba)

            np.savez_compressed(rd / "preds.npz", test_pred=test_pred, test_proba=test_proba,
                                test_gold=y["test"], val_pred=val_proba.argmax(1),
                                val_proba=val_proba, val_gold=y["val"],
                                test_ids=np.array([str(i) for i in ids["test"]], dtype=object))
            result = {
                "run_id": f"classical/{args.model}/{args.dataset}/{args.phase}/seed{args.seed}",
                "family": "classical", "model": args.model, "phase": args.phase,
                "config": config, "seed": args.seed, "dataset": args.dataset,
                "split_name": args.split_name, "text_field": args.text_field,
                "feature": feature, "n_features": meta["n_features"],
                "rows": {s: len(y[s]) for s in SPLITS},
                "params": best_params, "frozen_hash": frozen.get("hash") if args.phase == "final" else None,
                "best_val_macro_f1": round(float(best_val), 4),
                "val": val_metrics, "test": metrics, "history": history,
                "elapsed_s": round(timer.elapsed(), 1),
            }
            atomic_write_json(rd / "metrics.json", result)
            atomic_write_json(rd / "history.json", history)
            atomic_write_text(rd / "per_class.txt", metrics["report"])
            atomic_write_text(rd / "report.md", build_report(result))
            write_status(rd, "done", elapsed_s=round(timer.elapsed(), 1),
                         test_macro_f1=metrics["macro_f1"], test_accuracy=metrics["accuracy"])
            log.log(f"DONE test_macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} elapsed={timer.fmt()}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            write_status(rd, "failed", error=str(exc))
            raise


def build_report(r: dict) -> str:
    t, v = r["test"], r["val"]
    lines = [f"# {r['model']} ({r['feature']}) — {r['phase']}", "",
             f"seed: {r['seed']} | dataset: {r['dataset']} | split: {r['split_name']} | "
             f"text: `{r['text_field']}` | features: {r['n_features']:,}", "",
             f"params: `{r['params']}` | frozen hash: `{r.get('frozen_hash')}` | val macro-F1: {r['best_val_macro_f1']:.4f}", "",
             "| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in (("val", v), ("test", t)):
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                     f"{m['weighted_f1']:.4f} | {m['mcc']:.4f} | {m.get('ece', float('nan')):.4f} |")
    lines += ["", "## Per-class (test)", "", "```", t["report"].rstrip(), "```", ""]
    if r["history"]:
        lines += ["## Search history (val)", "", "| params | val macro-F1 |", "|---|---:|"]
        lines += [f"| `{h['params']}` | {h['val_macro_f1']:.4f} |" for h in r["history"]]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
