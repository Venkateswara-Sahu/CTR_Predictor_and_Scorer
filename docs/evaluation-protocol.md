# CTR repair protocol (frozen before test evaluation)

This replaces the invalid historical pipeline; the original notebooks and v1.0
release remain historical evidence. No historical AUC is a clean benchmark.

## Scope and data

Criteo labelled `train.txt`, 13 anonymous integer fields and 26 anonymous
categorical fields. Zero-based source row windows: training [0, 600000),
validation [1000000, 1150000), test [10000001, 10250001). The final test window
is beyond the historical 10,000,001-row prefix. Row order is not a verified
timestamp. This is a bounded sample study, not full-dataset or production proof.
Keep the first occurrence of each exact raw feature tuple, within each split;
exclude validation duplicates of training and test duplicates of train/validation.
Report removals, class balance, missingness and raw subset SHA256 hashes.

## Training and model choice

Fixed 123-feature schema: each numeric field has train-median imputation,
signed log1p and an original-null flag; each category has a deterministic
BLAKE2b bucket in 0..65535, training frequency, and missing flag. Six categories
(C1,C2,C14,C17,C19,C21) additionally have target means smoothed with 100
training-prior observations. Training target features use five-fold shuffled
KFold(seed=42), with each fold's labels excluded; inference uses full training
maps. No preprocessing or feature selection uses validation/test rows.

Baselines: training global click rate and standardized logistic regression
(C=1, max_iter=300). Tree candidates: LightGBM 31 and 63 leaves, 300 rounds,
learning_rate=.05, min_child_samples=100, colsample=.9; XGBoost 300 rounds,
max_depth=6, learning_rate=.05, min_child_weight=20, colsample=.9. Four CPU
threads, seed 42. Early stop after 40 rounds on validation log loss. Validate
LightGBM/XGBoost mixtures with weights 0,.1,...,1. Select the lowest validation
log loss among candidates and mixtures. Choose binary F1 threshold on validation
from .01,.02,...,.99. Freeze models, transformer, selection and checksums before
opening final test labels. No test-driven retuning.

## Evidence and serving

Report test ROC AUC, average precision, log loss, Brier score, F1 at the frozen
threshold, ten calibration bins, and top-decile click precision/lift with counts.
Use 200 seeded paired row bootstrap resamples for 95% percentile intervals on
AUC, log loss, Brier and lift, and selected-minus-global log loss. Intervals
assume independent rows and do not capture advertiser/time clustering.
Save row IDs, labels, predictions, source/code/environment hashes and timings.
Latency excludes model loading and measures warmed local batches (1 and 100).

Serve only bundles with the v2 schema and matching file hashes; no old pickle
fallback. Rank by predicted CTR. Anonymous fields provide no verified semantic
ad-quality score or causal ranking/revenue effect. Reject malformed inputs;
preserve explicitly missing fields as null. A partial row is allowed but carries
no performance guarantee. The dashboard reads saved evidence, never hardcoded
"accuracy", conversion or business-lift claims.

## Implementation sequence

- [ ] Regression tests: null handling, batch/process/order invariance, unseen
  categories, cross-fit isolation, artifact schema and input validation.
- [ ] Shared transformer and bounded streaming data loader; run red/green tests.
- [ ] Train/validate, freeze compatible bundle; evaluate test once and save evidence.
- [ ] Integrate strict predictor, CTR ranking, Flask factory and Streamlit UI.
- [ ] Run regression/API/dashboard checks, independent review, update public copy
  with actual results and prepare an isolated reviewable branch.
