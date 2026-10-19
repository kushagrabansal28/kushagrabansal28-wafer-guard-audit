"""Reproducible evidence for the three silent defects in the original notebook (bnypi.ipynb).

Replicates the ORIGINAL pipeline (mean-impute + z-score on all rows, SMOTE, random stratified
split, [B,1,F] targets) and then repairs one defect at a time (ablation ladder).
Run:  python audit_evidence.py   -> prints results and writes audit_evidence.json
"""
import json
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wafer_guard as wg

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "secom_telemetry_modified.csv"
SEED, HP = wg.SEED, dict(wg.HP)
X, y, ts = wg.load_telemetry(CSV)
legacy = wg.WaferPreprocessor.legacy_mask(X)
out = {}


class OriginalSAE(wg.WaferSAE):
    """Original forward pass: silently flattens [B, C, F] -> [B, C*F] (no shape guard)."""

    def forward(self, x):
        z = self.encoder(x.flatten(start_dim=1))
        return self.decoder(z), self.head(z).squeeze(-1), z


def buggy_loader(Xa, ya, shuffle):
    """Original loader: explicit channel axis -> [N, 1, F]."""
    g = torch.Generator().manual_seed(SEED)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xa).float().unsqueeze(1), torch.from_numpy(ya).float())
    return torch.utils.data.DataLoader(ds, batch_size=HP["batch_size"], shuffle=shuffle,
                                       generator=g if shuffle else None, drop_last=shuffle)


def train_variant(Xtr, ytr, Xev, yev, broadcast_loss, pos_weight=None):
    """Original training loop; `broadcast_loss=True` reproduces recon [B,F] vs target [B,1,F]."""
    wg.set_seed(SEED)
    model = OriginalSAE(Xtr.shape[1], HP["hidden"], HP["latent"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(HP["epochs"]):
        model.train()
        for xb, yb in buggy_loader(Xtr, ytr, True):                  # xb: [B, 1, F]
            recon, logits, _ = model(xb)
            rec = torch.mean((recon - xb) ** 2) if broadcast_loss else torch.mean((recon - xb.squeeze(1)) ** 2)
            loss = rec + HP["lambda_clf"] * bce(logits, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    model.eval()
    return model


def probs(model, Xa):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(Xa).float().unsqueeze(1))[1]).numpy()


# ---- A. original pipeline, reproduced ----------------------------------------------------------
imp, sc = SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler()
X_all = sc.fit_transform(imp.fit_transform(X))                          # fitted on ALL rows (incl. future/test)
X_bal, y_bal = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(X_all, y)
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=SEED)
mA = train_variant(Xa_tr, ya_tr, Xa_te, ya_te, broadcast_loss=True)
pA = probs(mA, Xa_te)
A = wg.evaluate(ya_te, pA, 0.5)
n_syn = len(X_bal) - len(X)
out["A_original"] = dict(A, n_synthetic_total=int(n_syn), n_real=int(len(X)), test_prevalence=A["prevalence"])
print(f"[A] original pipeline: test n={A['n']} prevalence={A['prevalence']:.1%} "
      f"F1={A['f1']:.3f} P={A['precision']:.3f} R={A['recall']:.3f} AP={A['pr_auc']:.3f} AUC={A['roc_auc']:.3f}")
print(f"    SMOTE added {n_syn} synthetic rows to {len(X)} real ones; "
      f"~{n_syn/len(X_bal):.0%} of the balanced data (and of its random test split) is synthetic")

# ---- Defect 1: leakage evidence -----------------------------------------------------------------
tr, va, te = wg.chronological_split(ts)
mu_all, mu_tr = np.nanmean(X, axis=0), np.nanmean(X[tr], axis=0)
sd_all, sd_tr = np.nanstd(X, axis=0), np.nanstd(X[tr], axis=0)
ok = sd_tr > 0
shift = np.abs(mu_all[ok] - mu_tr[ok]) / sd_tr[ok]
out["leak_stats"] = dict(median_mean_shift_in_train_sd=float(np.median(shift)),
                         n_features_shift_gt_0_25_sd=int((shift > 0.25).sum()), n_features=int(ok.sum()))
print(f"[leak] all-data vs train-only mean differs by median {np.median(shift):.3f} sd; "
      f"{(shift > 0.25).sum()}/{ok.sum()} features move >0.25 sd")
# nearest-train-neighbour distance: random-split test rows vs chronological test rows (same scaled space)
from sklearn.neighbors import NearestNeighbors
nn_rand = NearestNeighbors(n_neighbors=1).fit(Xa_tr).kneighbors(Xa_te)[0].mean()
Xs_tr, Xs_te = X_all[tr], X_all[te]
nn_chrono = NearestNeighbors(n_neighbors=1).fit(Xs_tr).kneighbors(Xs_te)[0].mean()
out["leak_nn"] = dict(random_split_mean_nn_dist=float(nn_rand), chronological_mean_nn_dist=float(nn_chrono))
print(f"[leak] mean nearest-train-neighbour distance of test rows: random split+SMOTE {nn_rand:.1f} "
      f"vs chronological {nn_chrono:.1f}")

# ---- B/C/D. ablation ladder, all scored on the chronological test split at natural prevalence ---
imp_t, sc_t = SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler()
Xn_tr = sc_t.fit_transform(imp_t.fit_transform(X[tr]))                    # train-only fit, legacy NOT handled
Xn_va, Xn_te = (sc_t.transform(imp_t.transform(X[i])) for i in (va, te))
Xn_sm, yn_sm = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(Xn_tr, y[tr])   # SMOTE on train only


def ladder(name, model_probs_va, model_probs_te):
    thr = wg.select_threshold(y[va], model_probs_va)
    m = wg.evaluate(y[te], model_probs_te, thr)
    out[name] = m
    print(f"[{name}] test n={m['n']} prev={m['prevalence']:.1%} thr={thr:.3f} F1={m['f1']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} AP={m['pr_auc']:.3f} AUC={m['roc_auc']:.3f} "
          f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")


mB = train_variant(Xn_sm, yn_sm, Xn_te, y[te], broadcast_loss=True)
ladder("B_leakage_fixed_only", probs(mB, Xn_va), probs(mB, Xn_te))
mC = train_variant(Xn_sm, yn_sm, Xn_te, y[te], broadcast_loss=False)
ladder("C_plus_loss_fixed", probs(mC, Xn_va), probs(mC, Xn_te))
res = wg.run_experiment(CSV)
Z_va = res["prep"].transform(X[va])
ladder("D_full_repair", wg.predict_proba(res["model"], Z_va), res["p_test"])

# ---- Defect 2: loss evidence --------------------------------------------------------------------
B, Fdim = HP["batch_size"], X_all.shape[1]
xb = torch.from_numpy(Xa_tr[:B]).float().unsqueeze(1)                    # [B, 1, F] as in the original loader
perfect = xb.squeeze(1).clone().requires_grad_(True)                      # a PERFECT reconstruction, [B, F]
buggy = torch.mean((perfect - xb) ** 2)                                   # original expression
buggy.backward()
g_buggy = perfect.grad.clone()
perfect.grad = None
correct = torch.mean((perfect - xb.squeeze(1)) ** 2); correct.backward()
within_batch_var = float(xb.squeeze(1).var(dim=0, unbiased=False).mean() * 2)
out["loss_evidence"] = dict(
    broadcast_shape=list(torch.broadcast_shapes(perfect.shape, xb.shape)),
    loss_at_perfect_reconstruction_buggy=float(buggy), loss_at_perfect_reconstruction_correct=float(correct),
    grad_norm_at_perfect_reconstruction_buggy=float(g_buggy.norm()),
    grad_norm_at_perfect_reconstruction_correct=float(perfect.grad.norm()),
    expected_buggy_floor=within_batch_var)
print(f"[loss] recon {tuple(perfect.shape)} vs target {tuple(xb.shape)} broadcasts to "
      f"{tuple(torch.broadcast_shapes(perfect.shape, xb.shape))}")
print(f"[loss] at a PERFECT reconstruction: buggy loss={float(buggy):.4f} (grad norm {float(g_buggy.norm()):.4f}) "
      f"vs correct loss={float(correct):.4f} (grad norm {float(perfect.grad.norm()):.4f})")

# ---- Defect 3: invariant-blindness evidence -----------------------------------------------------
z = (X[:, :2] - np.nanmean(X[:, :2], axis=0)) / np.nanstd(X[:, :2], axis=0)       # original scaler stats (all rows)
codes = X[legacy, 1]
real_range = (np.nanmin(X[~legacy, 1]), np.nanmax(X[~legacy, 1]))
syn1 = X_bal[len(X):, :]                                                  # SMOTE rows live in z-space
raw_syn1 = sc.inverse_transform(syn1)[:, 1]
fab = (~np.isin(np.round(raw_syn1), [100, 200, 300])) & ((raw_syn1 < real_range[0]) | (raw_syn1 > real_range[1]))
out["invariant"] = dict(
    n_legacy=int(legacy.sum()), codes={str(c): int((codes == c).sum()) for c in wg.FIRMWARE_CODES},
    sensor_0_mean_all=float(np.nanmean(X[:, 0])), sensor_0_mean_nonlegacy=float(np.nanmean(X[~legacy, 0])),
    sensor_1_mean_all=float(np.nanmean(X[:, 1])), sensor_1_mean_nonlegacy=float(np.nanmean(X[~legacy, 1])),
    legacy_sensor_1_zscore_range=[float(z[legacy, 1].min()), float(z[legacy, 1].max())],
    legacy_sensor_0_zscore=float(z[legacy, 0].mean()),
    sensor_1_std_all=float(np.nanstd(X[:, 1])), sensor_1_std_nonlegacy=float(np.nanstd(X[~legacy, 1])),
    sensor_0_std_all=float(np.nanstd(X[:, 0])), sensor_0_std_nonlegacy=float(np.nanstd(X[~legacy, 0])),
    code_100_vs_300_zscore_gap=float(z[legacy & (X[:, 1] == 300), 1].mean() - z[legacy & (X[:, 1] == 100), 1].mean()),
    smote_rows_with_impossible_sensor_1=int(fab.sum()), smote_rows=int(len(syn1)),
    legacy_defect_rate=float(y[legacy].mean()), nonlegacy_defect_rate=float(y[~legacy].mean()))
inv = out["invariant"]
print(f"[invariant] {inv['n_legacy']} legacy wafers, codes {inv['codes']}; original scaler maps their sensor_1 to "
      f"z in [{inv['legacy_sensor_1_zscore_range'][0]:.1f}, {inv['legacy_sensor_1_zscore_range'][1]:.1f}] "
      f"and sensor_0 to z={inv['legacy_sensor_0_zscore']:.1f}")
print(f"[invariant] sensor_1 mean {inv['sensor_1_mean_all']:.1f} (all) vs {inv['sensor_1_mean_nonlegacy']:.1f} "
      f"(non-legacy); SMOTE produced {inv['smote_rows_with_impossible_sensor_1']}/{inv['smote_rows']} rows "
      f"with sensor_1 outside both the code set and the real range")
print(f"[invariant] std inflated by sentinel/codes: sensor_0 {inv['sensor_0_std_nonlegacy']:.1f} -> {inv['sensor_0_std_all']:.1f}, "
      f"sensor_1 {inv['sensor_1_std_nonlegacy']:.1f} -> {inv['sensor_1_std_all']:.1f}; "
      f"codes 100 vs 300 differ by only {inv['code_100_vs_300_zscore_gap']:.2f} z (an ordinal fake distance)")
print(f"[invariant] defect rate: legacy {inv['legacy_defect_rate']:.1%} vs non-legacy {inv['nonlegacy_defect_rate']:.1%}")

# ---- other defects: outliers, dead features, imputed sentinel, fixed 0.5 threshold -------------------
a_abs = np.abs(X_all)
mostly_missing = np.isnan(X).mean(0) > 0.5
constant = np.nanstd(X, axis=0) == 0
s0 = X[:, 0]
real0 = s0[~legacy & ~np.isnan(s0)]
pB = probs(mB, Xn_te)                                                     # original model, SMOTE-balanced training
out["extras"] = dict(
    max_abs_z=float(a_abs.max()), cols_max_z_gt20=int((a_abs.max(0) > 20).sum()),
    share_sq_mass_from_z_gt5=float((X_all[a_abs > 5] ** 2).sum() / (X_all ** 2).sum()), frac_entries_z_gt5=float((a_abs > 5).mean()),
    constant_cols=int(constant.sum()), mostly_missing_cols=int(mostly_missing.sum()), dead_cols=int((constant | mostly_missing).sum()),
    imputed_share_in_mostly_missing=float(np.isnan(X[:, mostly_missing]).mean()),
    sensor0_imputed_value=float(np.nanmean(s0)), sensor0_min_real=float(real0.min()), sensor0_rows_imputed=int(np.isnan(s0).sum()),
    thr05_flagged_share=float((pB >= 0.5).mean()), thr05_mean_prob=float(pB.mean()), test_prevalence=float(y[te].mean()))
e = out["extras"]
print(f"[extra] outliers: {e['share_sq_mass_from_z_gt5']:.1%} of the squared mass the reconstruction loss sees comes from "
      f"{e['frac_entries_z_gt5']:.2%} of entries (|z| > 5); {e['cols_max_z_gt20']} columns reach |z| > 20")
print(f"[extra] dead features: {e['dead_cols']} of {X.shape[1]} sensors are constant ({e['constant_cols']}) or over 50% missing "
      f"({e['mostly_missing_cols']}); {e['imputed_share_in_mostly_missing']:.0%} of cells in the latter are imputed")
print(f"[extra] sensor_0 mean-imputed with {e['sensor0_imputed_value']:.1f}, below every real reading (min {e['sensor0_min_real']:.1f}); {e['sensor0_rows_imputed']} rows")
print(f"[extra] fixed 0.5 threshold on a model trained at 50% prevalence flags {e['thr05_flagged_share']:.1%} of real test lots "
      f"(true rate {e['test_prevalence']:.1%}); mean predicted probability {e['thr05_mean_prob']:.3f}")

with open("audit_evidence.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("wrote audit_evidence.json")
