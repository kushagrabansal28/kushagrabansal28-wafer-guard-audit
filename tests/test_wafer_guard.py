"""Targeted tests for wafer_guard. Run:  python -m pytest -q"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wafer_guard as wg  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "secom_telemetry_modified.csv")


def make_X(n=200, d=12, n_legacy=30, seed=0):
    """Small synthetic fixture (unit tests only): sensor_0 ~ 3000, sensor_1 ~ 2500, rest N(0,1)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 0] = rng.normal(3000, 50, n)
    X[:, 1] = rng.normal(2500, 80, n)
    legacy = np.zeros(n, bool)
    legacy[rng.choice(n, n_legacy, replace=False)] = True
    X[legacy, 0] = 0.0
    X[legacy, 1] = rng.choice(wg.FIRMWARE_CODES, legacy.sum())
    return X, legacy


# 1. legacy sentinel identification ------------------------------------------------------------------
def test_legacy_sentinel_identified():
    X, legacy = make_X()
    assert np.array_equal(wg.WaferPreprocessor.legacy_mask(X), legacy)


def test_nan_and_nonzero_sensor0_are_not_legacy():
    X, _ = make_X(n=10, n_legacy=0)
    X[0, 0] = np.nan
    X[1, 0] = 1e-9
    assert not wg.WaferPreprocessor.legacy_mask(X)[:2].any()


# 2. firmware codes 100 / 200 / 300 ------------------------------------------------------------------
def test_firmware_codes_one_hot_and_unknown():
    X, _ = make_X()
    prep = wg.WaferPreprocessor().fit(X)
    rows = np.tile(X[:1], (5, 1))
    rows[:, 0] = 0.0
    rows[:, 1] = [100.0, 200.0, 300.0, 250.0, np.nan]            # last two: unseen / missing code
    out = prep.transform(rows)
    tail = out[:, -5:]                                            # is_legacy, c100, c200, c300, unknown
    assert tail.tolist() == [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]]


def test_non_legacy_rows_have_no_code_flags():
    X, legacy = make_X()
    out = wg.WaferPreprocessor().fit_transform(X)
    assert out[~legacy, -5:].sum() == 0


# 3. codes are neither scaled nor imputed -------------------------------------------------------------
def test_codes_not_scaled_or_imputed_as_continuous():
    X, legacy = make_X()
    prep = wg.WaferPreprocessor().fit(X)
    assert np.isclose(prep.mean_[1], X[~legacy, 1].mean(), rtol=0.05)   # stats from non-legacy rows only
    assert prep.mean_[1] > 2000 and prep.mean_[0] > 2500                # not dragged toward 100-300 / 0
    out = prep.transform(X)
    k0, k1 = list(prep.keep_).index(0), list(prep.keep_).index(1)
    assert np.all(out[legacy][:, [k0, k1]] == 0.0)                      # masked, independent of the code value
    X2 = X.copy()
    X2[legacy, 1] = 300.0
    assert np.array_equal(prep.transform(X2)[:, :-4], out[:, :-4])      # continuous block ignores the code
    assert not np.array_equal(prep.transform(X2)[:, -4:], out[:, -4:])  # code lives only in the one-hot


# 4. training-only fit and reuse --------------------------------------------------------------------
def test_preprocessor_requires_fit_and_is_frozen():
    X, _ = make_X()
    with pytest.raises(RuntimeError):
        wg.WaferPreprocessor().transform(X)
    prep = wg.WaferPreprocessor().fit(X[:120])
    ref = wg.WaferPreprocessor().fit(X[:120])
    before = prep.mean_.copy()
    _ = prep.transform(X[120:] * 100 + 7)                              # wildly different eval data
    assert np.array_equal(prep.mean_, before) and np.array_equal(prep.std_, ref.std_)
    assert np.array_equal(prep.transform(X[120:]), ref.transform(X[120:]))


def test_dropped_features_use_training_data_only():
    X, _ = make_X()
    X[:, 5] = 3.0                                                       # constant in training...
    prep = wg.WaferPreprocessor().fit(X[:100])
    assert 5 not in prep.keep_
    X_eval = X[100:].copy()
    X_eval[:, 5] = np.random.default_rng(1).normal(size=len(X_eval))    # ...varies later: still dropped
    assert prep.transform(X_eval).shape[1] == len(prep.keep_) + 5


def test_robust_to_missing_inf_and_constant_columns():
    X, _ = make_X()
    X[::7, 3] = np.nan
    X[3, 4] = np.inf
    X[:, 6] = np.nan                                                    # all-missing column
    out = wg.WaferPreprocessor().fit_transform(X)
    assert np.isfinite(out).all() and 6 not in wg.WaferPreprocessor().fit(X).keep_


# 5. temporal split integrity / no future leakage ------------------------------------------------------
def test_chronological_split_is_strict_and_tie_safe():
    ts = np.repeat(np.arange(50), 2).astype("datetime64[m]")            # every timestamp duplicated
    tr, va, te = wg.chronological_split(ts, (0.6, 0.2, 0.2))
    assert ts[tr].max() < ts[va].min() and ts[va].max() < ts[te].min()
    assert sorted(np.concatenate([tr, va, te])) == list(range(len(ts)))
    with pytest.raises(ValueError):
        wg.chronological_split(ts[::-1])


def test_prediction_at_t_independent_of_future_rows():
    X, _ = make_X()
    prep = wg.WaferPreprocessor().fit(X[:120])
    full = prep.transform(X)
    X_future_changed = X.copy()
    X_future_changed[150:] = np.random.default_rng(2).normal(size=X[150:].shape) * 1e3
    assert np.array_equal(prep.transform(X_future_changed)[:150], full[:150])   # row-wise, no look-ahead
    assert np.array_equal(prep.transform(X[10:11]), full[10:11])


# 6. shapes and loss ---------------------------------------------------------------------------------
def _batch(B=8, F=20):
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, F, generator=g), (torch.arange(B) % 2).float()


def test_model_output_shapes_and_input_guard():
    model = wg.WaferSAE(20, 16, 4)
    x, _ = _batch()
    recon, logits, z = model(x)
    assert recon.shape == (8, 20) and logits.shape == (8,) and z.shape == (8, 4)
    with pytest.raises(ValueError):
        model(x.unsqueeze(1))                                           # [B,1,F] must be rejected


def test_loss_matches_manual_and_rejects_broadcasting():
    x, y = _batch()
    recon = x + 0.5
    logits = torch.randn(8)
    total, rec, clf = wg.sae_loss(recon, logits, x, y, None, 1.0)
    assert torch.isclose(rec, torch.tensor(0.25))                        # exactly the per-element MSE
    assert torch.isclose(total, rec + clf)
    assert wg.sae_loss(x.clone(), logits, x, y)[1].item() == 0.0         # perfect reconstruction -> 0 loss
    with pytest.raises(ValueError):
        wg.sae_loss(recon, logits, x.unsqueeze(1), y)                    # [B,1,F] target would broadcast to [B,B,F]
    with pytest.raises(ValueError):
        wg.sae_loss(recon, logits.unsqueeze(1), x, y)                    # [B,1] logits vs [B] targets


# 7. finite loss / valid gradients ------------------------------------------------------------------
def test_loss_finite_and_gradients_valid_for_all_parameters():
    torch.manual_seed(0)
    model = wg.WaferSAE(20, 16, 4)
    x, y = _batch()
    recon, logits, _ = model(x)
    loss, _, _ = wg.sae_loss(recon, logits, x, y, torch.tensor(3.0))
    loss.backward()
    wg.assert_valid_gradients(model)
    assert torch.isfinite(loss)
    assert all(p.grad.abs().sum() > 0 for p in model.parameters())      # encoder, decoder and head all learn


def test_nonfinite_loss_is_detected():
    x, y = _batch()
    with pytest.raises(FloatingPointError):
        wg.sae_loss(x * float("nan"), torch.zeros(8), x, y)


# 8. training / evaluation mode ------------------------------------------------------------------------
def test_train_eval_mode_separation():
    torch.manual_seed(0)
    model = wg.WaferSAE(20, 16, 4, dropout=0.5)
    x, _ = _batch()
    model.eval()
    assert torch.equal(model(x)[1], model(x)[1])                         # deterministic in eval
    model.train()
    assert not torch.equal(model(x)[1], model(x)[1])                     # dropout active in train
    p = wg.predict_proba(model, x.numpy())
    assert not model.training and p.shape == (8,) and ((p >= 0) & (p <= 1)).all()


# 9. natural prevalence in final evaluation (real dataset) ----------------------------------------------
@pytest.fixture(scope="module")
def real_run():
    if not os.path.exists(CSV):
        pytest.skip("secom_telemetry_modified.csv not found")
    return wg.run_experiment(CSV)


def test_final_evaluation_uses_natural_prevalence_and_time_order(real_run):
    r = real_run
    tr, va, te = r["splits"]
    y, ts = r["y"], r["ts"]
    assert r["test"]["n"] == len(te) and r["test"]["n_pos"] == int(y[te].sum())   # no resampling of the test set
    assert abs(r["test"]["prevalence"] - y[te].mean()) < 1e-12
    assert 0.02 < r["test"]["prevalence"] < 0.12 and abs(y.mean() - 0.066) < 0.005  # near the natural ~6.6 %
    assert ts[tr].max() < ts[va].min() and ts[va].max() < ts[te].min()
    assert r["test"]["tp"] + r["test"]["fn"] == r["test"]["n_pos"]
    assert r["test"]["tn"] + r["test"]["fp"] == r["test"]["n_neg"]


def test_real_data_legacy_rows_follow_business_rules(real_run):
    X, y, _ = wg.load_telemetry(CSV)
    legacy = wg.WaferPreprocessor.legacy_mask(X)
    assert legacy.sum() > 0 and set(np.unique(X[legacy, 1])) <= set(wg.FIRMWARE_CODES)
    assert not np.isin(X[~legacy, 1], wg.FIRMWARE_CODES).any()
    out = real_run["prep"].transform(X)
    assert np.all(out[legacy, -5] == 1) and np.all(out[~legacy, -5] == 0)       # is_legacy flag
    assert np.all(out[legacy, -4:].sum(axis=1) == 1)                            # exactly one code category
    assert out[legacy, -4:-1].sum(axis=0).tolist() == [61, 35, 50]              # codes 100 / 200 / 300 counts


# 10. repeatable inference and serialisation --------------------------------------------------------------
def test_same_seed_gives_identical_results(real_run):
    again = wg.run_experiment(CSV)
    assert np.array_equal(again["p_test"], real_run["p_test"]) and again["threshold"] == real_run["threshold"]


def test_artifacts_round_trip(real_run):
    r = real_run
    X, _, _ = wg.load_telemetry(CSV)
    te = r["splits"][2]
    with tempfile.TemporaryDirectory() as d:
        wg.save_artifacts(d, r["model"], r["prep"], r["threshold"])
        model, prep, thr = wg.load_artifacts(d)
    p = wg.predict_proba(model, prep.transform(X[te]))
    assert thr == r["threshold"] and np.array_equal(p, r["p_test"])
    assert np.array_equal(p, wg.predict_proba(model, prep.transform(X[te])))   # repeatable inference
