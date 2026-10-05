"""Training-fitted v2 Criteo transformer; no request-time fitting or pickle."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

NUMERIC = [f'I{i}' for i in range(1, 14)]
CATEGORICAL = [f'C{i}' for i in range(1, 27)]
RAW_FEATURES = NUMERIC + CATEGORICAL
TARGET_COLUMNS = ['C1', 'C2', 'C14', 'C17', 'C19', 'C21']
FEATURE_NAMES = ([name for c in NUMERIC for name in (c, f'{c}_log', f'{c}_missing')]
                 + [name for c in CATEGORICAL for name in (f'{c}_hash', f'{c}_frequency', f'{c}_missing')]
                 + [f'{c}_target' for c in TARGET_COLUMNS])


def category_values(series):
    return series.map(lambda v: 'M:' if pd.isna(v) else 'V:' + str(v))


def bucket(value):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=4).digest(), 'little') % 65536


class FeatureEngineer:
    def __init__(self, artifacts_path=None):
        self.state = None
        if artifacts_path is not None:
            self.state = self.load(artifacts_path).state

    @staticmethod
    def validate(frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise ValueError('Provide a nonempty frame with unique raw feature columns')
        extra = set(frame.columns) - set(RAW_FEATURES)
        if extra:
            raise ValueError(f'Unknown fields: {sorted(extra)}')
        frame = frame.reindex(columns=RAW_FEATURES).copy()
        for c in NUMERIC:
            values = pd.to_numeric(frame[c], errors='coerce')
            if (frame[c].notna() & values.isna()).any() or np.isinf(values.to_numpy(dtype=float)).any():
                raise ValueError(f'{c} must be a finite number or null')
            frame[c] = values
        for c in CATEGORICAL:
            if frame[c].map(lambda x: not isinstance(x, str) and not pd.isna(x) if not isinstance(x, (dict, list, tuple)) else True).any():
                raise ValueError(f'{c} must be a string category or null')
        return frame

    @staticmethod
    def target_map(values, y):
        prior = float(np.mean(y))
        grouped = pd.DataFrame({'v': values.to_numpy(), 'y': y}).groupby('v')['y'].agg(['sum', 'count'])
        return ((grouped['sum'] + 100 * prior) / (grouped['count'] + 100)).to_dict(), prior

    def fit(self, frame, y):
        frame = self.validate(frame)
        y = np.asarray(y)
        if y.shape != (len(frame),) or not np.isin(y, [0, 1]).all():
            raise ValueError('Labels must be a matching binary vector')
        medians = {c: float(frame[c].median()) if frame[c].notna().any() else 0.0 for c in NUMERIC}
        frequencies, targets = {}, {}
        for c in CATEGORICAL:
            values = category_values(frame[c])
            frequencies[c] = (values.value_counts() / len(frame)).to_dict()
            if c in TARGET_COLUMNS:
                targets[c], _ = self.target_map(values, y)
        self.state = {'schema_version': 2, 'feature_names': FEATURE_NAMES, 'medians': medians,
                      'frequencies': frequencies, 'targets': targets, 'prior': float(y.mean())}
        return self

    def transform(self, frame):
        if self.state is None:
            raise ValueError('Transformer is not fitted')
        frame = self.validate(frame)
        out = {}
        for c in NUMERIC:
            x = frame[c].fillna(self.state['medians'][c]).to_numpy(dtype=float)
            out[c] = x
            out[f'{c}_log'] = np.sign(x) * np.log1p(np.abs(x))
            out[f'{c}_missing'] = frame[c].isna().to_numpy(dtype=float)
        for c in CATEGORICAL:
            values = category_values(frame[c])
            mapping = {value: bucket(value) for value in values.unique()}
            out[f'{c}_hash'] = values.map(mapping).to_numpy()
            out[f'{c}_frequency'] = values.map(self.state['frequencies'][c]).fillna(0).to_numpy()
            out[f'{c}_missing'] = frame[c].isna().to_numpy(dtype=float)
            if c in TARGET_COLUMNS:
                out[f'{c}_target'] = values.map(self.state['targets'][c]).fillna(self.state['prior']).to_numpy()
        result = pd.DataFrame(out, index=frame.index).reindex(columns=FEATURE_NAMES).astype(np.float32)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError('Features exceed supported float32 range')
        return result

    def fit_transform(self, frame, y):
        self.fit(frame, y)
        result = self.transform(frame)
        if len(frame) < 5:
            raise ValueError('Cross fitting requires at least five training rows')
        frame = self.validate(frame)
        y = np.asarray(y)
        for train, holdout in KFold(n_splits=5, shuffle=True, random_state=42).split(frame):
            for c in TARGET_COLUMNS:
                mapping, prior = self.target_map(category_values(frame.iloc[train][c]), y[train])
                encoded = category_values(frame.iloc[holdout][c]).map(mapping).fillna(prior)
                result.iloc[holdout, result.columns.get_loc(f'{c}_target')] = encoded.to_numpy(dtype=np.float32)
        return result

    def save(self, path):
        if self.state is None:
            raise ValueError('Transformer is not fitted')
        Path(path).write_text(json.dumps(self.state, sort_keys=True, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if state.get('schema_version') != 2 or state.get('feature_names') != FEATURE_NAMES:
            raise ValueError('Incompatible transformer schema; use a complete v2 bundle')
        if set(state.get('medians', {})) != set(NUMERIC) or set(state.get('frequencies', {})) != set(CATEGORICAL) or set(state.get('targets', {})) != set(TARGET_COLUMNS):
            raise ValueError('Incomplete transformer')
        obj = cls()
        obj.state = state
        return obj
