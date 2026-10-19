# Audit: WaferGuard wafer excursion detector

## Summary

**Recommendation: no-go.** The original F1 of 0.90+ came from three silent bugs. With them fixed, and the model scored on a time-ordered test split at its natural defect rate, **F1 is 0.105**. Flagging every lot as defective scores 0.103, so the model adds nothing. All numbers come from code in this repo; the original notebook and dataset were not modified.

| # | Bug | Evidence | Fix |
|---|---|---|---|
| 1 | **Data leakage.** Scaling and SMOTE used every row, and the split was random. | Reproduced F1 0.990 on a test set that was 50% defective and about 46% synthetic. | Chronological 60/20/20 split, preprocessing fitted on training rows only, no SMOTE. |
| 2 | **Corrupted loss.** `[B, F]` against `[B, 1, F]` broadcast to `[B, B, F]`. | A perfect reconstruction scored loss 1.138 instead of 0. | `[B, F]` tensors, per-sample loss, shape checks that raise. |
| 3 | **Legacy-wafer blindness.** The `sensor_0 == 0.0` sentinel and firmware codes 100/200/300 were scaled as measurements. | 146 lots (9.3%); `sensor_0` spread 74.3 to 880.4. | Explicit legacy flag, one-hot codes, continuous channels masked, training-only statistics. |

**Result** (313 untouched test lots, 17 defects, 5.43%): precision 0.095, recall 0.118, F1 0.105, PR-AUC 0.081. It catches 2 of 17 defects with 19 false alarms. Five seeds give F1 0.111 ± 0.020; a logistic regression gives 0.120.

**Also found:** five smaller defects in the original, measured below: an outlier-dominated loss, 144 unusable sensors, an impossible imputed `sensor_0` value, a fixed 0.5 threshold that flags none of the real test lots, and a deployment artifact built from all rows.

**Why not ship:** validation PR-AUC peaks in the first epoch while the network memorises the 76 training defects. In a five-window out-of-time backtest the repaired model's ROC-AUC averages 0.57 (95% interval 0.46 to 0.67, which includes chance). Simpler models reach 0.59 to 0.64, so a weak, unstable signal does exist, but none gets near usable precision. With 17 test defects every single-split number is fragile (F1 interval 0.000 to 0.255).

The rest of this file gives the evidence for each bug, the repair-by-repair scores, the full result tables and the limits.

## The three bugs

### 1. Data leakage

```python
X_scaled = scaler.fit_transform(imputer.fit_transform(X_raw))    # statistics from every row
X_bal, y_bal = SMOTE(...).fit_resample(X_scaled, y_raw)          # oversampling before the split
train_test_split(X_bal, y_bal, stratify=y_bal, ...)              # random split of time-ordered data
```

Imputation and scaling saw the evaluation rows, SMOTE made "test defects" out of blends of training defects, and the random split let the model train on later lots to predict earlier ones (`timestamp` was never used). The test set ended up 50% defective and about 46% synthetic (1,359 synthetic rows added to 1,567 real).

Reproduced original: **F1 0.990** on 586 test rows. With only the leakage fixed and scored on the real test split, F1 is 0.188.

Fix: 60/20/20 chronological split (cuts never fall inside a block of identical timestamps), preprocessing fitted on training rows only, no SMOTE (a training-only `pos_weight` handles imbalance), epoch and threshold chosen on validation, test split left at its natural prevalence. The transform is row-wise, so a prediction at time t cannot depend on later rows (unit-tested).

### 2. Corrupted loss gradients

```python
Xt = torch.from_numpy(X).float().unsqueeze(1)             # [B, 1, F]
recon_loss = torch.mean((reconstructed - batch_x) ** 2)    # [B, F] against [B, 1, F]
```

`[B, F]` against `[B, 1, F]` broadcasts to `[B, B, F]`, so each reconstruction was compared with every sample in the batch, not its own input. The minimiser of that loss is the batch mean. No error was raised because the model flattens its input. For a batch of 64, `(64, 590)` against `(64, 1, 590)` gives `(64, 64, 590)`; at a perfect reconstruction the original expression returns 1.138 instead of 0.

Fix: `[B, F]` tensors throughout, per-sample MSE plus weighted BCE, and shape checks that raise in the model and the loss. The first step asserts every parameter gets a finite gradient, and a non-finite loss stops training.

### 3. Legacy probe-station blindness

On PS-L wafers `sensor_0` is the sentinel `0.0` and `sensor_1` is a firmware code (100, 200 or 300). The original scaled both as measurements. 146 lots (9.3%) are legacy: 61 with code 100, 35 with 200, 50 with 300. The sentinel and codes inflate the two channels' spread (`sensor_0` std 74.3 to 880.4, `sensor_1` 79.8 to 675.1), squeeze the 1,421 real wafers into a narrow band, put codes 100 and 300 only 0.30 standard deviations apart, and never expose "legacy" as a feature. None of the 146 legacy lots is defective (7.3% of the rest are).

SMOTE did not invent impossible codes (0 of 1,359 rows) only because no legacy lot is a defect; that is a measurement, not a bug.

Fix (`WaferPreprocessor`): `is_legacy = (sensor_0 == 0.0)`, with NaN treated as missing. On legacy rows `sensor_1` becomes one-hot `fw_code_100/200/300` plus `fw_code_unknown`, and both continuous channels are masked to 0.0 after scaling, with their statistics taken from non-legacy rows only. Other sensors are clipped to the training 1st/99th percentile and standardised with training statistics. Features constant or over 50% missing in training are dropped (146 of 590).

## Other defects in the original

Smaller than the three above, each measured by `audit_evidence.py`:

- **Loss dominated by a few extreme readings.** With no outlier handling, 0.25% of the scaled values (|z| > 5) carry 37.8% of the squared mass the reconstruction loss sees; 108 columns reach |z| > 20 (maximum 39.6). Fix: clip to the training 1st/99th percentile.
- **144 of 590 sensors are unusable.** 116 are constant and 28 are over 50% missing (69% of their cells imputed). Fix: drop them using training statistics.
- **An impossible imputed value.** The 6 missing `sensor_0` cells were filled with 2732.1, below every real reading (minimum 2743.2), because the mean includes the sentinel zeros. Fix: statistics from non-legacy rows only.
- **A fixed 0.5 threshold.** The original-style model, trained on 50%-prevalence data, flags none of the 313 real test lots at 0.5 (mean predicted probability 0.00002, true defect rate 5.4%). Fix: threshold chosen on validation at natural prevalence.
- **Leaky deployment artifact.** `artifacts/preprocessor.joblib` is saved from the all-row fit, so the deployed scaler carries statistics from every lot (41 of 468 feature means move over 0.25 sd). The release gate also compared a single F1 point estimate with 0.90, with no interval or seeds.

## What each repair did

| Step | F1 | PR-AUC |
|---|---|---|
| A. Original (leaky, random split, 50% prevalence test) | 0.990 | 1.000 |
| B. Leakage fixed | 0.188 | 0.139 |
| C. Plus loss fixed | 0.112 | 0.083 |
| D. Plus legacy handling (final) | 0.105 | 0.081 |

B to D use the same test split (313 lots, 17 defects). Only A to B is a large change; B, C and D differ by less than the noise 17 positives produce, so I do not claim the later repairs raised the score.

## Final result

Splits: train 940 lots (76 defects, 8.09%), validation 314 (11, 3.50%), test 313 (17, 5.43%); whole dataset 1,567 (104, 6.64%). The defect rate falls from 22% in July to 3% in September, so the test split is reported as it is, not resampled to 6.6%. The model is the original supervised autoencoder (305,666 parameters) with its original hyperparameters; seed 1337, best epoch by validation PR-AUC, threshold 0.6162 (F1-maximising on validation). Nothing was tuned on the test split.

| Precision | Recall | F1 | PR-AUC | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| 0.095 | 0.118 | 0.105 | 0.081 | 0.422 | 0.891 |

Confusion matrix: TN 277, FP 19, FN 15, TP 2. Bootstrap 95% intervals: F1 0.000 to 0.255, PR-AUC 0.032 to 0.236. An all-pass rule has accuracy 0.946, so accuracy and ROC-AUC are supplementary only.

| Reference | F1 | PR-AUC |
|---|---|---|
| Flag every lot | 0.103 | 0.054 |
| Class-weighted logistic regression | 0.120 | 0.073 |
| Five seeds of this pipeline (mean, sd) | 0.111, 0.020 | 0.078, 0.007 |

A missed defect ships a bad lot; a false alarm costs an inspection. Here 19 false alarms come with 2 catches and 88% of defects are missed, and no threshold on this model gives a useful balance.

## Could a different model do better?

One test split holds 17 defects, so I also ran a rolling out-of-time backtest (`backtest.py`): five consecutive windows from 1 September to 17 October (188 lots each, 4 to 18 defects, 39 in total, mean prevalence 4.1%). In every window the preprocessor and each model are fitted on earlier lots only. Rank metrics are used because they need no threshold. Settings are fixed in advance; only the logistic regression's regularisation is chosen on a validation slice of the past, and nothing was tuned on the windows.

| Model | Mean ROC-AUC | 95% interval | Mean PR-AUC |
|---|---|---|---|
| Repaired autoencoder | 0.566 | 0.46 to 0.67 | 0.075 |
| Logistic regression | 0.638 | 0.56 to 0.72 | 0.080 |
| Random forest | 0.626 | 0.54 to 0.71 | 0.149 |
| Gradient boosting | 0.594 | 0.51 to 0.68 | 0.171 |
| Chance | 0.500 | | 0.041 (prevalence) |

The repaired autoencoder has not shown skill: its interval includes 0.5, and it is the weakest of the four. The other three have a weak but real ranking signal (intervals above 0.5), so the data is not empty and the supplied architecture is a poor fit for it. The tree models' higher PR-AUC comes almost entirely from one window with 7 defects (September 12 to 22); in the other windows they sit near the prevalence, and per-window ROC-AUC swings from 0.39 to 0.97. Precision stays low throughout, so none of these is ship-ready either.

What I would try next: a regularised linear or boosted-tree model with forward-chaining validation, more defect examples from the recent period, and drift monitoring. I did not change the repaired model to chase a better number, because the task is to repair the supplied detector.

## Limits

- 17 test and 11 validation defects make every number fragile; the chosen threshold ranges from 0.005 to 0.616 across seeds.
- Validation PR-AUC peaks at epoch 1 while training classification loss falls to 0.0008: this network memorises the 76 training defects, and what it memorises does not carry forward in time.
- The backtest windows cover 1 September to 17 October only; the early, high-defect weeks are used for training but never as a test window. Each window has few defects, so the model comparison is indicative, not conclusive.
- Defect rate and legacy share drift, so a deployment needs monitoring and recalibration. There are no defective legacy lots to learn from.
- Choices that are mine, not the brief's: the 60/20/20 split, masking plus flags for the legacy channels, 1%/99% clipping, the 50% missing-feature cutoff, `pos_weight` instead of SMOTE. All are fitted on training data only.
- Tested on Python 3.13, torch 2.14.0 (CPU), scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.6.

## Tests and reproduction

22 tests cover sentinel detection, codes 100/200/300 (including unknown and missing), no scaling or imputation of codes, training-only fit and frozen reuse, tie-safe chronological split and no look-ahead, tensor shapes and loss values, finite loss and gradients on every parameter, train/eval mode, natural prevalence on the real data, same-seed repeatability, artifact round trip, and strict time ordering of the backtest windows. Synthetic fixtures appear only in unit tests. The notebook runs from a fresh kernel without errors. As a one-off check outside the pipeline, the sentinel rule matches the supplied evaluator mask on all 146 rows; no code here reads that file.

```bash
pip install -r requirements.txt
python -m pytest -q
python audit_evidence.py
python backtest.py                             # rolling out-of-time comparison, about 2.5 minutes
jupyter nbconvert --to notebook --execute --inplace wafer_guard_fixed.ipynb
```
