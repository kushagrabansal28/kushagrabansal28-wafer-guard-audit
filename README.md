# WaferGuard: audit and repair of a wafer defect detector

GDG TECH AI/ML Wildcard entry. The task: a PyTorch notebook reported F1 of 0.90 or more on `secom_telemetry_modified.csv` and was due to ship. Audit it, repair it so it respects time causality, and report honest metrics at the natural defect rate.

The reported score was not real. It came from three silent bugs. With them fixed, the model scored on a time-ordered test split has **F1 0.105**, and flagging every lot as defective scores 0.103. **Recommendation: no-go.** The evidence is in [AUDIT.md](AUDIT.md), and the repaired notebook is [wafer_guard_fixed.ipynb](wafer_guard_fixed.ipynb).

## Result at a glance

| | Value |
|---|---|
| F1 reported by the original notebook | 0.990, on a test set that was 50% defective and about 46% synthetic |
| F1 after the repair | **0.105**, on 313 untouched lots with 17 defects (5.43%) |
| Flag every lot (trivial rule) | 0.103 |
| Repaired model | precision 0.095, recall 0.118, PR-AUC 0.081; catches 2 of 17 defects with 19 false alarms |
| Rolling out-of-time backtest | repaired model ROC-AUC 0.57 (interval 0.46 to 0.67, includes chance); simpler models 0.59 to 0.64 |

## What was wrong in the original

| # | Bug | Evidence | Fix |
|---|---|---|---|
| 1 | **Data leakage.** Scaling and SMOTE used every row, and the train/test split was random on time-ordered data. | Reproduced F1 0.990; fixing only the leakage drops it to 0.188. | Chronological 60/20/20 split, preprocessing fitted on training rows only, no SMOTE. |
| 2 | **Corrupted loss.** Reconstructions `[B, F]` were compared with targets `[B, 1, F]`, which broadcast to `[B, B, F]`. | A perfect reconstruction scored loss 1.138 instead of 0. | `[B, F]` tensors throughout, per-sample loss, shape checks that raise. |
| 3 | **Legacy-wafer blindness.** The `sensor_0 == 0.0` sentinel and firmware codes 100/200/300 on probe station PS-L were scaled as measurements. | 146 lots (9.3%); `sensor_0` spread inflated from 74.3 to 880.4. | Explicit legacy flag, one-hot codes, continuous channels masked, training-only statistics. |

Five smaller defects are also measured: a loss dominated by a few extreme readings, 144 of 590 unusable sensors, an impossible imputed `sensor_0` value, a fixed 0.5 threshold that flags none of the real test lots, and a deployment scaler fitted on every row. Details are in `AUDIT.md`.

## Repository layout

| Path | Purpose |
|---|---|
| `wafer_guard_fixed.ipynb` | Repaired notebook, executed end to end, with the fix source printed inline |
| `AUDIT.md` | One-page summary, then evidence, repairs, measured metrics and limits |
| `wafer_guard.py` | The pipeline: `WaferPreprocessor`, `WaferSAE`, `sae_loss`, training, thresholding, metrics, artifact I/O |
| `backtest.py`, `backtest_results.json` | Rolling out-of-time backtest comparing the repaired model with simpler models |
| `audit_evidence.py`, `audit_evidence.json` | Reproduces the original bugs and the repair-by-repair scores |
| `tests/` | 22 tests |
| `index.html`, `dashboard/` | Interactive report of the audit, built from this repository's own outputs |
| `secom_telemetry_modified.csv` | The supplied dataset, unmodified |
| `requirements.txt` | Python dependencies |

## Quick start

Requires Python 3.10 or newer. A CPU is enough. Run everything from the repository root.

```bash
pip install -r requirements.txt

python -m pytest -q                                                          # 22 tests, about 20 seconds
jupyter nbconvert --to notebook --execute --inplace wafer_guard_fixed.ipynb  # about 4 minutes, includes the backtest
python audit_evidence.py                                                     # reproduces the original bugs, about 2 minutes
python backtest.py                                                           # out-of-time comparison, about 2.5 minutes
python dashboard/build_dashboard.py                                          # rebuilds index.html
```

To look at the interactive report, open `index.html` in a browser. It needs no server or network connection.

Suggested reading order: `AUDIT.md` (summary first), then the notebook, then `index.html`.

## How the repaired pipeline works

1. **Split.** Lots are sorted by `timestamp` and split 60/20/20 into train, validation and test. Cut points never fall inside a block of identical timestamps, and strict time order is asserted.
2. **Preprocessing** (`WaferPreprocessor`, fitted on training rows only, then frozen):
   - `is_legacy` is `sensor_0 == 0.0`. A `NaN` is a missing value, not a legacy marker.
   - On legacy rows `sensor_1` becomes a one-hot firmware code (100, 200, 300 or unknown) and both continuous channels are masked. Their statistics come from non-legacy rows only.
   - Other sensors are clipped to the training 1st and 99th percentile and standardised. Missing values become the training mean. Features that are constant or over 50% missing in training are dropped.
3. **Model.** The original supervised autoencoder (encoder to a 32-dimensional latent vector, mirrored decoder, linear classification head), with strict `[B, F]` input and shape checks.
4. **Loss.** Per-sample reconstruction error plus class-weighted binary cross-entropy. The class weight comes from the training split, and SMOTE is not used.
5. **Training.** AdamW, cosine schedule, gradient clipping. The first step asserts that every parameter has a finite gradient, and a non-finite loss stops training. The epoch with the best validation PR-AUC is kept.
6. **Threshold.** Chosen on the validation split to maximise F1.
7. **Evaluation.** The test split is scored once, at its natural defect rate and without resampling. Precision, recall, F1, PR-AUC, the confusion matrix and a bootstrap interval are reported, with ROC-AUC and accuracy as supplementary figures.
8. **Reference points.** Flagging every lot, class-weighted logistic regression, five seeds, and a five-window rolling backtest with a random forest and gradient boosting.

## Measured results

Final model, seed 1337, threshold 0.6162:

| Lots | Defects | Prevalence | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 313 | 17 | 5.43% | 0.095 | 0.118 | 0.105 | 0.081 | 0.422 |

Confusion matrix: TN 277, FP 19, FN 15, TP 2. Bootstrap 95% intervals: F1 0.000 to 0.255, PR-AUC 0.032 to 0.236. Five seeds of the full pipeline: F1 0.111 ± 0.020.

What each repair did to the score (rows B to D use the same test split):

| Step | F1 | PR-AUC |
|---|---|---|
| A. Original notebook | 0.990 | 1.000 |
| B. Leakage fixed | 0.188 | 0.139 |
| C. Plus loss fixed | 0.112 | 0.083 |
| D. Plus legacy handling (final) | 0.105 | 0.081 |

Only A to B is a large change. B, C and D differ by less than the noise 17 defects produce.

Rolling out-of-time backtest (five windows, 39 defects in total):

| Model | Mean ROC-AUC | 95% interval |
|---|---|---|
| Repaired autoencoder | 0.566 | 0.46 to 0.67 |
| Logistic regression | 0.638 | 0.56 to 0.72 |
| Random forest | 0.626 | 0.54 to 0.71 |
| Gradient boosting | 0.594 | 0.51 to 0.68 |

## Tests

The 22 tests in `tests/` cover:
- legacy sentinel detection, including `NaN` and near-zero values
- firmware codes 100, 200 and 300, including unknown and missing codes
- no scaling or imputation of the codes
- training-only fit and frozen reuse
- tie-safe chronological splits and the guarantee that a prediction at time t cannot depend on later rows
- tensor shapes and loss values, including rejection of `[B, 1, F]` targets
- finite loss and valid gradients on every parameter
- train and eval mode, and natural prevalence on the real data
- same-seed repeatability and an artifact save/load round trip
- strict time ordering of the backtest windows

Synthetic fixtures appear only in unit tests. All reported numbers come from the real dataset.

## Reproducibility

- Fixed seed (1337), deterministic PyTorch algorithms, and a seeded data loader. Re-running gives identical probabilities.
- Tested on Python 3.13, torch 2.14.0 (CPU), scikit-learn 1.9.0, pandas 2.3.3 and numpy 2.4.6. Other platforms or versions may differ in the last digits.
- The notebook is self-contained: it imports `wafer_guard.py` and `backtest.py` from the same folder and reads the CSV from the working directory.

## Limitations

- 17 test defects (and 11 validation defects) make every single-split number fragile. The chosen threshold ranges from 0.005 to 0.616 across seeds.
- The network memorises the 76 training defects: validation PR-AUC peaks at epoch 1 while training loss falls to nearly zero.
- Defect rate and legacy share drift over time, so a real deployment would need monitoring and recalibration.
- No legacy lot is defective, so nothing can be learned or checked about failures on that tool.
- The backtest covers 1 September to 17 October only, and each window holds 4 to 18 defects, so the model comparison is indicative rather than conclusive.
- Design choices that are mine, not the brief's: the 60/20/20 split, masking plus flags for the legacy channels, 1%/99% clipping, the 50% missing-feature cutoff, and a class weight instead of SMOTE. All are fitted on training data only.

## Data

`secom_telemetry_modified.csv` is the supplied dataset, unmodified: 1,567 lots, 590 sensors, a `label` (-1 pass, +1 fail) and a `timestamp`. A separate evaluator-only legacy mask was also supplied. It is not part of this repository, and no code here reads it.
