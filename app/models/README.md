# Models Directory

This directory contains the trained machine learning models for the CTR prediction system.

## Files

### Required Models:
- **`lightgbm_model.txt`** - LightGBM model for CTR prediction
- **`xgboost_model.json`** - XGBoost model for CTR prediction
- **`ad_ranking_system.pkl`** - Complete ad ranking system (optional, for quick loading)

## Model Details

### LightGBM Model
- **Format:** Text file (LightGBM native format)
- **Size:** ~5-10 MB
- **Training AUC:** 0.74+
- **Features:** 150 engineered features

### XGBoost Model
- **Format:** JSON (XGBoost native format)
- **Size:** ~3-8 MB
- **Training AUC:** 0.75+
- **Features:** 150 engineered features

### Ensemble
- **Weights:** 50% XGBoost + 50% LightGBM
- **Combined AUC:** 0.76+
- **With Ad Scoring:** 0.8888 AUC

## Loading Models

The models are automatically loaded by the `CTRPredictor` class in `../ctr_model.py`:

```python
from app.ctr_model import CTRPredictor

# Load models
predictor = CTRPredictor()
predictor.load_models()

# Make predictions
predictions = predictor.predict(features)
```

## Model Training

Models were trained using:
- **Dataset:** Criteo Display Advertising Challenge (10M samples)
- **Training samples:** 7M (subsampled to 1M for memory efficiency)
- **Test samples:** 2M
- **Validation samples:** 1M
- **Optimization:** Optuna with 100 trials
- **Training time:** ~20 minutes per model

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8888 |
| Log Loss | 0.1089 |
| Precision@10 | 1.0000 |
| Precision@100 | 1.0000 |
| CTR Lift (Top 10%) | +265.6% |

## Updating Models

To update the models:

1. Train new models using the notebooks
2. Save models to `../../models/` directory
3. Copy to this directory:
   ```bash
   cp ../../models/lightgbm_model.txt .
   cp ../../models/xgboost_model.json .
   ```
4. Restart the Flask application

## Notes

- Models are loaded once at startup for efficiency
- Model files are NOT included in version control (too large)
- Download models from training artifacts or retrain using notebooks
- Ensure model files are present before deploying to production
