"""Build index.html: an interactive report of the audit, using only numbers produced by this repo's code.

Runs the repaired pipeline (five seeds), reads audit_evidence.json (from audit_evidence.py) and the
dataset, and embeds everything into dashboard/template.html. The page needs no network and does not
retrain anything in the browser.

Run from the repository root:  python dashboard/build_dashboard.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wafer_guard as wg  # noqa: E402

CSV = ROOT / "secom_telemetry_modified.csv"
EVIDENCE = ROOT / "audit_evidence.json"
if not EVIDENCE.exists():
    sys.exit("audit_evidence.json not found: run `python audit_evidence.py` first")
ev = json.loads(EVIDENCE.read_text())
BACKTEST = ROOT / "backtest_results.json"
if not BACKTEST.exists():
    sys.exit("backtest_results.json not found: run `python backtest.py` first")
bt_res = json.loads(BACKTEST.read_text())

X, y, ts = wg.load_telemetry(CSV)
legacy = wg.WaferPreprocessor.legacy_mask(X)
res = wg.run_experiment(CSV)
tr, va, te = res["splits"]
p_te, thr = res["p_test"], res["threshold"]
test = res["test"]
ci = wg.bootstrap_ci(y[te], p_te, thr, n_boot=2000, seed=wg.SEED)

# baselines (same features and threshold rule as the notebook)
Ztr, Zva, Zte = (res["prep"].transform(X[i]) for i in (tr, va, te))
best_c = max((average_precision_score(y[va], LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
                                      .fit(Ztr, y[tr]).predict_proba(Zva)[:, 1]), C) for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0))[1]
lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000).fit(Ztr, y[tr])
lr_m = wg.evaluate(y[te], lr.predict_proba(Zte)[:, 1], wg.select_threshold(y[va], lr.predict_proba(Zva)[:, 1]))
prev = float(y[te].mean())

seeds = []
for s in (wg.SEED, 1, 2, 3, 4):
    r = res if s == wg.SEED else wg.run_experiment(CSV, seed=s)
    t = r["test"]
    seeds.append(dict(seed=s, threshold=r["threshold"], precision=t["precision"], recall=t["recall"],
                      f1=t["f1"], pr_auc=t["pr_auc"], roc_auc=t["roc_auc"]))

month = pd.Series(ts).dt.to_period("M")
months = [dict(m=str(k), lots=int(len(g)), rate=float(y[g.index].mean()), legacy=float(legacy[g.index].mean()))
          for k, g in pd.Series(range(len(y))).groupby(month)]
day0 = ts.min()
lots = [[round(float((t - day0) / np.timedelta64(1, "D")), 2), int(y[i]), int(legacy[i])] for i, t in enumerate(ts)]
splits = [dict(name=n, n=int(len(i)), defects=int(y[i].sum()), prevalence=float(y[i].mean()), legacy=int(legacy[i].sum()),
               start=str(ts[i].min())[:10], end=str(ts[i].max())[:10]) for n, i in (("train", tr), ("validation", va), ("test", te))]
ladder = [dict(id="A", label="Original notebook", sub="leaky pipeline, random split, 50% defective test set", **{k: ev["A_original"][k] for k in ("f1", "pr_auc", "roc_auc")}),
          dict(id="B", label="Leakage fixed", sub="chronological split, training-only fit, natural prevalence", **{k: ev["B_leakage_fixed_only"][k] for k in ("f1", "pr_auc", "roc_auc")}),
          dict(id="C", label="Loss fixed", sub="per-sample reconstruction loss, no broadcasting", **{k: ev["C_plus_loss_fixed"][k] for k in ("f1", "pr_auc", "roc_auc")}),
          dict(id="D", label="Legacy handling (final)", sub="sentinel flag, firmware codes one-hot, class-weighted loss", **{k: ev["D_full_repair"][k] for k in ("f1", "pr_auc", "roc_auc")})]

data = dict(
    meta=dict(lots=int(len(y)), defects=int(y.sum()), prevalence=float(y.mean()), sensors=int(X.shape[1]), legacy=int(legacy.sum()),
              codes={str(int(c)): int((X[legacy, 1] == c).sum()) for c in wg.FIRMWARE_CODES},
              legacy_defects=int(y[legacy].sum()), other_rate=float(y[~legacy].mean()), features=int(Ztr.shape[1]),
              dropped=int(X.shape[1] - len(res["prep"].keep_))),
    ladder=ladder, splits=splits, months=months, lots=lots,
    history=[dict(epoch=h["epoch"], clf=h["clf"], recon=h["recon"], val_ap=h["val_ap"]) for h in res["history"]],
    scatter=dict(s0=[None if not np.isfinite(v) else round(float(v), 1) for v in X[:, 0]],
                 s1=[None if not np.isfinite(v) else round(float(v), 1) for v in X[:, 1]]),
    test=dict(y=y[te].tolist(), p=[round(float(v), 5) for v in p_te], t=[str(v)[:16].replace("T", " ") for v in ts[te]],
              threshold=thr, metrics=test,
              f1_ci=list(ci["f1"]), pr_ci=list(ci["pr_auc"]), n_boot=ci["n_valid"], prevalence=prev),
    baselines=dict(flag=dict(f1=2 * prev / (1 + prev), precision=prev, recall=1.0, pr_auc=prev),
                   logreg=dict(f1=lr_m["f1"], precision=lr_m["precision"], recall=lr_m["recall"], pr_auc=lr_m["pr_auc"], roc_auc=lr_m["roc_auc"])),
    seeds=seeds, evidence=dict(loss=ev["loss_evidence"], invariant=ev["invariant"], leak=ev["leak_stats"], nn=ev["leak_nn"],
                               original=ev["A_original"], extras=ev["extras"]),
    backtest=bt_res,
)
html = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
assert "/*DATA*/null" in html
(ROOT / "index.html").write_text(html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
print(f"wrote index.html | test F1 {test['f1']:.3f} at threshold {thr:.4f} | {len(lots)} lots")
