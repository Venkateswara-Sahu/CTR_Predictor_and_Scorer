# CTR Predictor and Candidate Ranking

A bounded Criteo click-prediction study with one training-fitted transformer shared
by offline evaluation, Flask and Streamlit. Candidate scores are predicted click
probabilities; anonymous fields do not establish semantic ad quality.

The October 2026 v2 rebuild fixes pre-split target-label leakage, request-batch
fitting, process-dependent hashes and missing-value indicators. Historical v1.0
models/scores remain historical; v2 does not reproduce or validate their AUC.

## Measured result — 5 October 2026

Validation selected LightGBM with 63 leaves; the ensemble did not improve its
validation log loss. Retained rows after exact feature-duplicate filtering:
599,971 training, 149,994 validation, **249,987 final test**. The test window is
beyond the historical 10,000,001-row prefix; row order is not a verified timestamp.

| Final test metric | Selected LightGBM | 95% row-bootstrap interval |
|---|---:|---:|
| ROC AUC | 0.76052 | 0.75874–0.76244 |
| Log loss | 0.48573 | 0.48373–0.48764 |
| Brier score | 0.15811 | 0.15733–0.15879 |
| Average precision | 0.55162 | — |
| F1 at validation-selected threshold 0.28 | 0.54002 | — |
| Top-decile predictive enrichment | 2.574× | 2.557–2.595× |

Top decile: 16,777 clicked rows / 24,999 selected rows (67.11%), against a
26.07% test click rate. This is observed predictive enrichment within the sample,
not causal CTR/revenue improvement. AUC is not accuracy. Global-rate and logistic
baselines and all tree candidates are in [the full report](evidence/benchmark.json).
Baseline F1 uses the selected model's threshold; constant-baseline top-decile
ordering is arbitrary. Empty calibration bins are omitted.

Local warmed inference, including preprocessing and excluding model loading:
30 repetitions, median **183.8 ms for one row**, **186.9 ms for 100 rows**;
p95 192.8/197.9 ms. Recorded hardware and scope are in the report. These timings
do not establish concurrent-serving capacity or a production SLA.

## Run

Python 3.12; exact direct dependencies in `requirements.txt`, observed full
Windows environment in `requirements-lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

The selected native model, JSON transformer and manifest are checked into
`app/models`; no old-release download or arbitrary pickle fallback runs on app
startup. Missing, incompatible or checksum-mismatched assets fail closed.

Flask endpoints: `/health`, `/predict_ctr`, `/rank_ads`, `/predict_single`.
Raw input accepts I1–I13 numbers/null and C1–C26 strings/null. Omitted fields are
missing. Unknown fields and malformed values fail with 400. Requests accept at
most 1,000 rows. `/score_ad` returns 410: historical semantic quality proxies are
retired. `/rank_ads` returns input-row `ad_id`, `predicted_ctr`, `final_score`;
`final_score` equals `predicted_ctr`. This is a v2 API contract change.

```json
{"features": {"I1": 4, "I2": null, "C1": "category-hash", "C2": null}}
```

Partial/unseen inputs work mechanically but carry no evaluated performance
guarantee. The UI examples are synthetic, not real advertisements.

## Reproduce the study

Obtain the labelled Criteo Kaggle `train.txt` separately; raw data is not included.
Run from the repo root in the pinned environment:

```powershell
python -m training.train --source "PATH\TO\train.txt" --output artifacts/new-run
python -m training.evaluate --bundle artifacts/new-run
```

Read the [frozen protocol](docs/evaluation-protocol.md) first. Training extracts
fixed source windows and pins raw bytes. Only training rows fit medians,
frequencies and target maps; training target values use five-fold label exclusion.
The feature schema has **123 features from 39 fields**. Deterministic categorical
hashing has collisions; it is not a learned embedding. Model selection and F1
threshold selection use validation only. Final evaluation checks frozen hashes
and creates a once-only marker. Do not reuse the same test for later tuning.

Saved evidence: [manifest](evidence/manifest.json), [benchmark](evidence/benchmark.json),
[raw test predictions](evidence/test_predictions.csv.gz), [evaluation start record](evidence/evaluation_started.json).
Predictions contain original source row IDs, labels and each model's probabilities.
The local ignored bundle also preserves baseline assets and duplicate fingerprints;
rerun training to reconstruct them. The shipping bundle includes only the selected
model. Code hashes identify the executed training code; evaluation has its own hash.

## Limits

This is a bounded sample prototype, not a full Criteo result or industrial ranking
system. Independent-row intervals omit advertiser/time clustering. No live traffic,
causal intervention, revenue, fairness, or semantic quality evaluation is established.
The original larger academic run used a different, invalid preprocessing contract;
its historical score cannot be compared as a valid baseline.
