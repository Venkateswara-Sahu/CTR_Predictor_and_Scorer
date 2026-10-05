"""Strict inference for checksum-verified v2 bundles."""
import hashlib
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from .feature_engineer import FeatureEngineer, FEATURE_NAMES

MODEL_FILES = {'lightgbm_31': 'lightgbm_31.txt', 'lightgbm_63': 'lightgbm_63.txt',
               'xgboost': 'xgboost.json', 'logistic': 'logistic.joblib'}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class CTRPredictor:
    def __init__(self, model_path=None):
        self.models = {}
        self.load_model(model_path or Path(__file__).parent / 'models')

    def load_model(self, model_path):
        root = Path(model_path)
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 2 or manifest.get('feature_names') != FEATURE_NAMES:
            raise ValueError('Incompatible model schema; a complete v2 bundle is required')
        weights = manifest.get('weights', {})
        if not weights or set(weights) - set(MODEL_FILES) - {'global_rate'}:
            raise ValueError('Invalid model selection')
        if not all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1):
            raise ValueError('Invalid ensemble weights')
        required = {'transformer.json'} | {MODEL_FILES[name] for name, weight in weights.items()
                                         if weight > 0 and name != 'global_rate'}
        for name in required:
            path = root / name
            if not path.is_file() or file_hash(path) != manifest.get('files', {}).get(name):
                raise ValueError(f'Bundle checksum mismatch or missing file: {name}')
        self.feature_engineer = FeatureEngineer.load(root / 'transformer.json')
        models = {}
        for name, weight in weights.items():
            if weight == 0 or name == 'global_rate':
                continue
            path = root / MODEL_FILES[name]
            if name.startswith('lightgbm'):
                model = lgb.Booster(model_file=str(path))
                if model.feature_name() != FEATURE_NAMES:
                    raise ValueError('LightGBM feature order mismatch')
            elif name == 'xgboost':
                model = xgb.XGBClassifier(n_jobs=4)
                model.load_model(path)
                if model.get_booster().feature_names != FEATURE_NAMES:
                    raise ValueError('XGBoost feature order mismatch')
            else:
                model = joblib.load(path)  # Locally trained, checksum-pinned bundle only.
                if list(model.feature_names_in_) != FEATURE_NAMES:
                    raise ValueError('Logistic feature order mismatch')
            models[name] = model
        if not 0 < manifest.get('threshold', 0) < 1 or not 0 <= manifest.get('prior', -1) <= 1:
            raise ValueError('Invalid probability metadata')
        self.models, self.manifest, self.ensemble_weights = models, manifest, weights

    def predict(self, X, return_probability=True):
        engineered = self.feature_engineer.transform(X)
        prediction = np.zeros(len(X), dtype=float)
        for name, weight in self.ensemble_weights.items():
            if weight == 0:
                continue
            if name == 'global_rate':
                values = np.full(len(X), self.manifest['prior'])
            elif name.startswith('lightgbm'):
                values = self.models[name].predict(engineered, num_threads=4)
            else:
                values = self.models[name].predict_proba(engineered)[:, 1]
            prediction += weight * values
        if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
            raise ValueError('Invalid model probabilities')
        return prediction if return_probability else (prediction >= self.manifest['threshold']).astype(int)

    def predict_batch(self, X_list):
        return self.predict(pd.DataFrame(X_list))
