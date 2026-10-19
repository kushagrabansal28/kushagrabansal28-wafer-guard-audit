"""WaferGuard: causality-safe preprocessing, supervised autoencoder and evaluation.

Repair of the Fab-7 wafer excursion detector. Every learned quantity (feature
selection, clipping bounds, scaling statistics, class weight, decision threshold)
is fitted on the training split only; splits are chronological.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

LEGACY_SENTINEL = 0.0                     # sensor_0 == 0.0 marks Probe Station PS-L
FIRMWARE_CODES = (100.0, 200.0, 300.0)    # sensor_1 on legacy wafers is categorical
SENTINEL_COL, CODE_COL = 0, 1             # positions of sensor_0 / sensor_1

HP = dict(batch_size=64, epochs=40, lr=1e-3, weight_decay=1e-5,
          hidden=256, latent=32, lambda_clf=1.0, dropout=0.10)
SEED = 1337


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------- #
# Data loading and chronological splitting
# --------------------------------------------------------------------------- #
def load_telemetry(path):
    """Return (X [N,590] float64, y [N] int64 with 1 = defect, timestamps [N]) sorted by time."""
    df = pd.read_csv(path)
    for col in ("label", "timestamp"):
        if col not in df.columns:
            raise ValueError(f"missing required column {col!r}")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError("no sensor_* columns found")
    ts = pd.to_datetime(df["timestamp"], errors="raise")
    order = np.argsort(ts.to_numpy(), kind="stable")           # stable: ties keep file order
    df, ts = df.iloc[order].reset_index(drop=True), ts.iloc[order].reset_index(drop=True)
    X = df[sensor_cols].to_numpy(dtype=np.float64)
    y = (df["label"].to_numpy() == 1).astype(np.int64)
    return X, y, ts.to_numpy()


def chronological_split(timestamps, fractions=(0.6, 0.2, 0.2)):
    """Split time-sorted data into train / val / test index arrays.

    Cut points never fall inside a block of identical timestamps, so
    max(train time) < min(val time) < ... holds strictly.
    """
    ts = np.asarray(timestamps)
    if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
        raise ValueError("timestamps must be sorted ascending")
    if not np.isclose(sum(fractions), 1.0) or len(fractions) != 3:
        raise ValueError("fractions must be three numbers summing to 1")
    n = len(ts)
    cuts, acc = [], 0.0
    for f in fractions[:2]:
        acc += f
        c = int(round(n * acc))
        while 0 < c < n and ts[c] == ts[c - 1]:
            c += 1
        cuts.append(c)
    idx = np.arange(n)
    parts = (idx[:cuts[0]], idx[cuts[0]:cuts[1]], idx[cuts[1]:])
    if any(len(p) == 0 for p in parts):
        raise ValueError("a split is empty; adjust fractions")
    assert ts[parts[0]].max() < ts[parts[1]].min() and ts[parts[1]].max() < ts[parts[2]].min()
    return parts


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class WaferPreprocessor:
    """Legacy-aware, training-only-fitted feature pipeline.

    Output columns: [continuous sensors kept | is_legacy | code_100 | code_200 | code_300 | code_unknown].

    * Legacy wafers (sensor_0 == 0.0) are flagged; sensor_0 / sensor_1 on those rows
      are NOT continuous measurements. sensor_1 is one-hot encoded as a firmware code
      (unknown / missing code -> `code_unknown`), and the two continuous channels are
      masked (set to 0.0 in the standardised space) on those rows.
    * Statistics of sensor_0 / sensor_1 are fitted on non-legacy rows only.
    * Other sensors: inf -> NaN, winsorise to training [q_lo, q_hi], standardise with
      training mean/std, missing -> 0.0 (training mean). Features that are constant or
      missing in more than `max_missing` of training rows are dropped.
    """

    def __init__(self, max_missing: float = 0.5, clip_quantiles=(0.01, 0.99),
                 firmware_codes=FIRMWARE_CODES):
        self.max_missing = max_missing
        self.clip_quantiles = clip_quantiles
        self.firmware_codes = tuple(float(c) for c in firmware_codes)
        self.fitted_ = False

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D [N, n_sensors], got shape {X.shape}")
        return np.where(np.isfinite(X), X, np.nan)

    @staticmethod
    def legacy_mask(X) -> np.ndarray:
        """True where sensor_0 is exactly the 0.0 sentinel (NaN is missing, not legacy)."""
        X = np.asarray(X, dtype=np.float64)
        return X[:, SENTINEL_COL] == LEGACY_SENTINEL

    def _masked(self, X, legacy):
        X = X.copy()
        X[legacy, SENTINEL_COL] = np.nan       # sentinel / code are not continuous values
        X[legacy, CODE_COL] = np.nan
        return X

    def fit(self, X):
        X = self._clean(X)
        self.n_sensors_ = X.shape[1]
        if self.n_sensors_ <= max(SENTINEL_COL, CODE_COL):
            raise ValueError("X needs at least two sensor columns")
        Xc = self._masked(X, self.legacy_mask(X))
        n_obs = np.sum(~np.isnan(Xc), axis=0)
        n_valid = np.full(self.n_sensors_, X.shape[0], dtype=float)
        n_valid[[SENTINEL_COL, CODE_COL]] = np.sum(~self.legacy_mask(X))
        with np.errstate(invalid="ignore", divide="ignore"):
            missing = 1.0 - n_obs / n_valid
        lo = np.full(self.n_sensors_, np.nan)
        hi = np.full(self.n_sensors_, np.nan)
        mean = np.full(self.n_sensors_, np.nan)
        std = np.full(self.n_sensors_, np.nan)
        for j in range(self.n_sensors_):
            col = Xc[:, j]
            col = col[~np.isnan(col)]
            if col.size < 2:
                continue
            lo[j], hi[j] = np.quantile(col, self.clip_quantiles)
            clipped = np.clip(col, lo[j], hi[j])
            mean[j], std[j] = clipped.mean(), clipped.std()
        keep = (missing <= self.max_missing) & np.isfinite(std) & (std > 1e-12)
        self.keep_ = np.flatnonzero(keep)
        self.lo_, self.hi_, self.mean_, self.std_ = lo, hi, mean, std
        self.missing_rate_ = missing
        self.feature_names_ = ([f"sensor_{j}" for j in self.keep_] + ["is_legacy"]
                               + [f"fw_code_{int(c)}" for c in self.firmware_codes] + ["fw_code_unknown"])
        self.fitted_ = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("WaferPreprocessor must be fitted on the training split first")
        X = self._clean(X)
        if X.shape[1] != self.n_sensors_:
            raise ValueError(f"expected {self.n_sensors_} sensor columns, got {X.shape[1]}")
        legacy = self.legacy_mask(X)
        Xc = self._masked(X, legacy)[:, self.keep_]
        k = self.keep_
        Z = (np.clip(Xc, self.lo_[k], self.hi_[k]) - self.mean_[k]) / self.std_[k]
        Z = np.where(np.isnan(Z), 0.0, Z)                   # missing / masked -> training mean
        code = X[:, CODE_COL]
        onehot = np.stack([legacy & (code == c) for c in self.firmware_codes], axis=1)
        unknown = legacy & ~onehot.any(axis=1)              # NaN or unseen code on a legacy wafer
        out = np.concatenate([Z, legacy[:, None], onehot, unknown[:, None]], axis=1).astype(np.float32)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite values after preprocessing")
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Model and loss
# --------------------------------------------------------------------------- #
class WaferSAE(nn.Module):
    """Supervised autoencoder (architecture of the original notebook, with shape guards)."""

    def __init__(self, d_in: int, d_hid: int = 256, d_lat: int = 32, dropout: float = 0.10):
        super().__init__()
        self.d_in = d_in
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat))
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_in))
        self.head = nn.Linear(d_lat, 1)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"expected x of shape [B, {self.d_in}], got {tuple(x.shape)}")
        z = self.encoder(x)                                   # [B, d_lat]
        recon = self.decoder(z)                               # [B, d_in]
        logits = self.head(z).squeeze(-1)                     # [B]
        return recon, logits, z


def sae_loss(recon, logits, x, y, pos_weight=None, lambda_clf: float = 1.0):
    """Reconstruction MSE + weighted BCE with strict shape checks (no silent broadcasting)."""
    if x.ndim != 2:
        raise ValueError(f"x must be [B, F]; got {tuple(x.shape)} (a channel axis would broadcast)")
    if recon.shape != x.shape:
        raise ValueError(f"recon {tuple(recon.shape)} != x {tuple(x.shape)}")
    if logits.shape != (x.shape[0],) or y.shape != logits.shape:
        raise ValueError(f"logits {tuple(logits.shape)} / y {tuple(y.shape)} must both be [{x.shape[0]}]")
    recon_loss = (recon - x).pow(2).mean(dim=1).mean()        # per-sample MSE, then batch mean
    clf_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    total = recon_loss + lambda_clf * clf_loss
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite loss: recon={recon_loss.item()}, clf={clf_loss.item()}")
    return total, recon_loss, clf_loss


def assert_valid_gradients(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise RuntimeError(f"no gradient for parameter {name}")
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"non-finite gradient for parameter {name}")


def make_loader(X, y, batch_size, shuffle, seed=SEED):
    Xt, yt = torch.from_numpy(np.ascontiguousarray(X)).float(), torch.from_numpy(np.asarray(y)).float()
    if Xt.ndim != 2 or yt.ndim != 1 or len(Xt) != len(yt):
        raise ValueError(f"bad shapes X {tuple(Xt.shape)}, y {tuple(yt.shape)}")
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt), batch_size=batch_size,
                                       shuffle=shuffle, generator=g if shuffle else None)


@torch.no_grad()
def predict_proba(model, X, batch_size=512, device="cpu") -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in make_loader(X, np.zeros(len(X)), batch_size, shuffle=False):
        _, logits, _ = model(xb.to(device))
        out.append(torch.sigmoid(logits).cpu())
    p = torch.cat(out).numpy()
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("non-finite predictions")
    return p


def train_model(Xtr, ytr, Xva, yva, hp=HP, seed=SEED, device="cpu", verbose=True):
    """Train on the training split; keep the epoch with the best validation PR-AUC."""
    set_seed(seed)
    model = WaferSAE(Xtr.shape[1], hp["hidden"], hp["latent"], hp.get("dropout", 0.10)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    n_pos = float(np.sum(ytr))
    if n_pos == 0:
        raise ValueError("training split has no positive samples")
    pos_weight = torch.tensor((len(ytr) - n_pos) / n_pos, device=device)      # training prevalence only
    loader = make_loader(Xtr, ytr, hp["batch_size"], shuffle=True, seed=seed)
    history, best_ap, best_state = [], -1.0, None
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        acc = np.zeros(3)
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            recon, logits, _ = model(xb)
            loss, rec, clf = sae_loss(recon, logits, xb, yb, pos_weight, hp["lambda_clf"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1 and i == 0:
                assert_valid_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            acc += np.array([loss.item(), rec.item(), clf.item()]) * len(xb)
        sched.step()
        acc /= len(ytr)
        ap = average_precision_score(yva, predict_proba(model, Xva, device=device))
        history.append(dict(epoch=epoch, total=acc[0], recon=acc[1], clf=acc[2], val_ap=ap))
        if ap > best_ap:
            best_ap, best_state = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:02d}/{hp['epochs']} | loss {acc[0]:.4f} | recon {acc[1]:.4f} "
                  f"| clf {acc[2]:.4f} | val AP {ap:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Thresholding and metrics
# --------------------------------------------------------------------------- #
def select_threshold(y_val, p_val) -> float:
    """F1-maximising threshold chosen on validation data only (ties -> largest threshold)."""
    y_val = np.asarray(y_val)
    if y_val.sum() == 0:
        raise ValueError("validation split has no positives; cannot select a threshold")
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    both = len(np.unique(y)) == 2
    return dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()), prevalence=float(y.mean()),
                threshold=float(threshold), precision=float(prec), recall=float(rec), f1=float(f1),
                pr_auc=float(average_precision_score(y, p)) if y.sum() else float("nan"),
                roc_auc=float(roc_auc_score(y, p)) if both else float("nan"),
                accuracy=float((pred == y).mean()), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bootstrap_ci(y, p, threshold, n_boot=2000, seed=SEED, alpha=0.05) -> dict:
    """Percentile bootstrap (resampling test rows) for F1 and PR-AUC; skips positive-free draws."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    f1s, aps = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        m = evaluate(y[i], p[i], threshold)
        f1s.append(m["f1"]); aps.append(m["pr_auc"])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return dict(f1=tuple(np.percentile(f1s, q)), pr_auc=tuple(np.percentile(aps, q)), n_valid=len(f1s))


# --------------------------------------------------------------------------- #
# End-to-end run and persistence
# --------------------------------------------------------------------------- #
def run_experiment(csv_path, seed=SEED, hp=HP, fractions=(0.6, 0.2, 0.2), verbose=False, device="cpu"):
    X, y, ts = load_telemetry(csv_path)
    tr, va, te = chronological_split(ts, fractions)
    prep = WaferPreprocessor().fit(X[tr])                       # training rows only
    Ztr, Zva, Zte = prep.transform(X[tr]), prep.transform(X[va]), prep.transform(X[te])
    model, history = train_model(Ztr, y[tr], Zva, y[va], hp, seed, device, verbose)
    thr = select_threshold(y[va], predict_proba(model, Zva, device=device))   # validation only
    p_te = predict_proba(model, Zte, device=device)
    return dict(model=model, prep=prep, history=history, threshold=thr, p_test=p_te,
                splits=(tr, va, te), y=y, ts=ts, test=evaluate(y[te], p_te, thr))


def save_artifacts(directory, model, prep, threshold, hp=HP) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(directory, "wafer_sae.pt"))
    joblib.dump(prep, os.path.join(directory, "preprocessor.joblib"))
    meta = dict(threshold=float(threshold), d_in=int(model.d_in), hp=hp)
    Path(directory, "meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(directory):
    meta = json.loads(Path(directory, "meta.json").read_text())
    hp = meta["hp"]
    model = WaferSAE(meta["d_in"], hp["hidden"], hp["latent"], hp.get("dropout", 0.10))
    model.load_state_dict(torch.load(os.path.join(directory, "wafer_sae.pt"), map_location="cpu"))
    model.eval()
    return model, joblib.load(os.path.join(directory, "preprocessor.joblib")), meta["threshold"]
