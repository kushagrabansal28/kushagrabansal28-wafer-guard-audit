"""Rolling out-of-time backtest: does any model carry signal across time, and is the repaired model's score a fluke of one split?

Five consecutive future windows (about 12% of the lots each, starting at 40% of the timeline). For every window the
preprocessor and every model are fitted only on lots that came before it, and the last 25% of those lots are held out as
validation for the autoencoder's epoch and the logistic regression's regularisation. Rank metrics (ROC-AUC, PR-AUC) are
reported per window because they need no threshold. The windows hold only a few defects each, so the spread is wide.

Run:  python backtest.py   ->  writes backtest_results.json
"""
import json
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import wafer_guard as wg

MODELS = ("repaired_sae", "logistic_regression", "random_forest", "gradient_boosting")


def _tie_safe(ts, c):
    while 0 < c < len(ts) and ts[c] == ts[c - 1]:
        c += 1
    return c


def backtest_windows(ts, start=0.40, n_windows=5, val_frac=0.25):
    """Return a list of (train_idx, val_idx, test_idx), each window strictly later than everything it is fitted on."""
    ts = np.asarray(ts)
    n = len(ts)
    edges = [_tie_safe(ts, int(round(n * (start + (1 - start) * k / n_windows)))) for k in range(n_windows + 1)]
    edges[-1] = n
    out = []
    for k in range(n_windows):
        tr_end, te_end = edges[k], edges[k + 1]
        split = _tie_safe(ts, int(tr_end * (1 - val_frac)))
        idx = np.arange(n)
        itr, iva, ite = idx[:split], idx[split:tr_end], idx[tr_end:te_end]
        if min(len(itr), len(iva), len(ite)) == 0:
            raise ValueError("empty window; adjust start or n_windows")
        assert ts[itr].max() < ts[iva].min() and ts[iva].max() < ts[ite].min(), "windows must be strictly time ordered"
        out.append((itr, iva, ite))
    return out


def _fit_score(name, Ztr, ytr, Zva, yva, Zte, seed):
    if name == "repaired_sae":
        model, _ = wg.train_model(Ztr, ytr, Zva, yva, seed=seed, verbose=False)
        return wg.predict_proba(model, Zte)
    if name == "logistic_regression":
        c = max((average_precision_score(yva, LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
                                         .fit(Ztr, ytr).predict_proba(Zva)[:, 1]), C) for C in (1e-4, 1e-3, 1e-2, 1e-1))[1]
        return LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, class_weight="balanced",
                                              random_state=0).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
    raise ValueError(name)


def _mean_auc(ys, ps):
    vals = [roc_auc_score(y, p) for y, p in zip(ys, ps) if 0 < y.sum() < len(y)]
    return float(np.mean(vals)) if vals else float("nan")


def run_backtest(csv_path, models=MODELS, seed=wg.SEED, n_boot=1000, verbose=False):
    warnings.filterwarnings("ignore")
    X, y, ts = wg.load_telemetry(csv_path)
    wins = backtest_windows(ts)
    scores = {m: [] for m in models}
    ytest = []
    t0 = time.time()
    for w, (itr, iva, ite) in enumerate(wins):
        prep = wg.WaferPreprocessor().fit(X[itr])                  # past lots only
        Ztr, Zva, Zte = (prep.transform(X[i]) for i in (itr, iva, ite))
        ytest.append(y[ite])
        for m in models:
            scores[m].append(_fit_score(m, Ztr, y[itr], Zva, y[iva], Zte, seed))
        if verbose:
            print(f"window {w + 1}/{len(wins)} done in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(seed)
    res = dict(windows=[dict(n=int(len(i[2])), defects=int(y[i[2]].sum()), prevalence=float(y[i[2]].mean()),
                             start=str(ts[i[2]].min())[:10], end=str(ts[i[2]].max())[:10], train_lots=int(len(i[0]) + len(i[1])))
                        for i in wins], models={})
    for m in models:
        per = [dict(roc_auc=float(roc_auc_score(yt, p)) if 0 < yt.sum() < len(yt) else None,
                    pr_auc=float(average_precision_score(yt, p)) if yt.sum() else None)
               for yt, p in zip(ytest, scores[m])]
        boots = []
        for _ in range(n_boot):
            ii = [rng.integers(0, len(yt), len(yt)) for yt in ytest]
            boots.append(_mean_auc([yt[i] for yt, i in zip(ytest, ii)], [p[i] for p, i in zip(scores[m], ii)]))
        boots = np.array([b for b in boots if np.isfinite(b)])
        res["models"][m] = dict(per_window=per, mean_roc_auc=float(np.mean([r["roc_auc"] for r in per if r["roc_auc"] is not None])),
                                mean_pr_auc=float(np.mean([r["pr_auc"] for r in per if r["pr_auc"] is not None])),
                                roc_auc_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
    res["mean_prevalence"] = float(np.mean([w["prevalence"] for w in res["windows"]]))
    res["total_defects"] = int(sum(w["defects"] for w in res["windows"]))
    return res


def print_table(res):
    print(f"windows: {[w['n'] for w in res['windows']]} lots, defects {[w['defects'] for w in res['windows']]} "
          f"(total {res['total_defects']}), mean prevalence {res['mean_prevalence']:.3f}")
    print(f"{'model':22s} {'mean ROC-AUC':>12s} {'95% CI':>16s} {'mean PR-AUC':>12s}   per-window ROC-AUC")
    for m, r in res["models"].items():
        ci = r["roc_auc_ci"]
        print(f"{m:22s} {r['mean_roc_auc']:12.3f} {f'{ci[0]:.2f} to {ci[1]:.2f}':>16s} {r['mean_pr_auc']:12.3f}   "
              + " ".join(f"{p['roc_auc']:.2f}" if p["roc_auc"] is not None else " n/a" for p in r["per_window"]))


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "secom_telemetry_modified.csv"
    out = run_backtest(csv, verbose=True)
    print_table(out)
    with open("backtest_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote backtest_results.json")
