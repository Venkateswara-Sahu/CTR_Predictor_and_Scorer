# V2 model bundle

Checked-in `manifest.json`, `transformer.json` and `lightgbm_63.txt` are the
validation-selected compatible bundle. `CTRPredictor` checks required hashes and
exact feature order before inference. The training manifest also records hashes
of candidate/baseline assets retained locally; those unselected assets are not
needed to serve this bundle. See ../../evidence/benchmark.json for measurements.

V1.0 model pickles and historical headline values are unsupported by this contract.
Do not mix v1 and v2 files. Retrain with ../../training/train.py rather than applying
new transforms to the old model.
