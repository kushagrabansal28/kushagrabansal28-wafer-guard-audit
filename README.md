# WaferGuard: audit and repair of the Fab-7 wafer anomaly detector

GDG TECH AI/ML Wildcard entry. The task was to audit a PyTorch notebook that reported F1 of 0.90+ on `secom_telemetry_modified.csv` and repair it so it respects time causality.

Short version: the original score came from data leakage, a loss that broadcast against the wrong shape, and legacy probe-station columns scaled as ordinary numbers. After the repairs, the model scored on a time-ordered test split at its natural defect rate has F1 0.105 (flagging every lot scores 0.103). A five-window out-of-time backtest agrees: the repaired model shows no demonstrated skill, and simpler models show only a weak, unstable signal. Recommendation: no-go. Details and evidence are in [AUDIT.md](AUDIT.md).

## Files

| File | Purpose |
|---|---|
| [wafer_guard_fixed.ipynb](wafer_guard_fixed.ipynb) | Repaired notebook, executed end to end |
| [AUDIT.md](AUDIT.md) | Findings, repairs, measured metrics |
| `wafer_guard.py` | Pipeline code: preprocessing, model, loss, training, metrics |
| `tests/` | 22 tests |
| `backtest.py`, `backtest_results.json` | Rolling out-of-time backtest comparing the repaired model with simpler models |
| `audit_evidence.py`, `audit_evidence.json` | Reproduces the original defects and the repair-by-repair ladder |
| `index.html` | Interactive report of the audit, built from this repo's outputs by `dashboard/build_dashboard.py` |
| `secom_telemetry_modified.csv` | The supplied dataset, unmodified |

## Setup and run

Python 3.10 or newer; CPU is enough (the full pipeline trains in seconds). Run everything from the repository root.

```bash
pip install -r requirements.txt
python -m pytest -q                                                          # tests
jupyter nbconvert --to notebook --execute --inplace wafer_guard_fixed.ipynb  # notebook
python audit_evidence.py                                                     # evidence for the original bugs
python backtest.py                                                           # out-of-time backtest, about 2.5 minutes
python dashboard/build_dashboard.py                                          # rebuild index.html
```

Open `index.html` in a browser; it needs no server or network.

## Design notes

- Chronological 60/20/20 split on `timestamp`. Preprocessing, class weight, checkpoint and threshold are fitted on training or validation rows only. The test split is scored once and keeps its natural defect rate.
- Legacy wafers (`sensor_0 == 0.0`) get an explicit flag. `sensor_1` is one-hot encoded as firmware code 100, 200, 300 or unknown and is never scaled or imputed as a measurement.
- The model is the original supervised autoencoder, with strict `[batch, features]` tensors and a shape-checked loss.
- scikit-learn supplies metrics and the logistic-regression baseline. `imbalanced-learn` is used only in `audit_evidence.py` to reproduce the original SMOTE step.

Limitations are listed in [AUDIT.md](AUDIT.md#limits).
