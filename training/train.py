"""Train and freeze before test evaluation. Run as python -m training.train."""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.feature_engineer import FeatureEngineer, FEATURE_NAMES, RAW_FEATURES
from training.data import WINDOWS, extract_windows, load_split, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/v2'))
    args = parser.parse_args()
    if (args.output / 'manifest.json').exists():
        raise SystemExit('Frozen bundle already exists; choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    executable_files = [Path('app/feature_engineer.py'), *sorted(Path('training').glob('*.py'))]
    code_hashes = {str(p).replace('\\', '/'): sha256(p) for p in executable_files}
    protocol_hash = sha256('docs/evaluation-protocol.md')
    started = time.perf_counter()
    paths = extract_windows(args.source, args.output / 'data')
    test_source_hash = sha256(paths['test'])  # Bytes only; final labels remain unused.
    seen = set()
    train, train_meta = load_split(paths['train'], seen, WINDOWS['train'][0])
    val, val_meta = load_split(paths['validation'], seen, WINDOWS['validation'][0])
    joblib.dump(seen, args.output / 'seen_features.joblib')
    y, vy = train.click.to_numpy(), val.click.to_numpy()
    engineer = FeatureEngineer()
    print('Cross-fitting training features', flush=True)
    features = engineer.fit_transform(train[RAW_FEATURES], y)
    vf = engineer.transform(val[RAW_FEATURES])
    engineer.save(args.output / 'transformer.json')
    candidates, models, elapsed = {}, {}, {}
    candidates['global_rate'] = np.full(len(val), y.mean())
    with threadpool_limits(limits=4):
        t = time.perf_counter()
        linear = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=300, random_state=42))
        print('Training logistic baseline', flush=True)
        linear.fit(features, y)
        candidates['logistic'] = linear.predict_proba(vf)[:, 1]
        elapsed['logistic'] = time.perf_counter() - t
        joblib.dump(linear, args.output / 'logistic.joblib')
        for leaves in (31, 63):
            name = f'lightgbm_{leaves}'
            t = time.perf_counter()
            print(f'Training {name}', flush=True)
            model = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=leaves,
                min_child_samples=100, colsample_bytree=.9, n_jobs=4, random_state=42,
                deterministic=True, force_col_wise=True, verbosity=-1)
            model.fit(features, y, eval_set=[(vf, vy)], eval_metric='binary_logloss',
                      callbacks=[lgb.early_stopping(40, verbose=False)])
            candidates[name] = model.predict_proba(vf)[:, 1]
            models[name] = model
            elapsed[name] = time.perf_counter() - t
            model.booster_.save_model(str(args.output / f'{name}.txt'))
        t = time.perf_counter()
        print('Training xgboost', flush=True)
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05,
            min_child_weight=20, colsample_bytree=.9, n_jobs=4, random_state=42,
            tree_method='hist', eval_metric='logloss', early_stopping_rounds=40)
        model.fit(features, y, eval_set=[(vf, vy)], verbose=False)
        candidates['xgboost'] = model.predict_proba(vf)[:, 1]
        models['xgboost'] = model
        model.save_model(args.output / 'xgboost.json')
        elapsed['xgboost'] = time.perf_counter() - t
    mixtures = {}
    for name in ('lightgbm_31', 'lightgbm_63'):
        for alpha in np.linspace(0, 1, 11):
            key = f'{name}+xgboost@{alpha:.1f}'
            candidates[key] = alpha * candidates[name] + (1-alpha) * candidates['xgboost']
            mixtures[key] = {name: float(alpha), 'xgboost': float(1-alpha)}
    losses = {name: float(log_loss(vy, pred)) for name, pred in candidates.items()}
    selected = min(losses, key=losses.get)
    weights = mixtures.get(selected, {selected: 1.0})
    thresholds = np.arange(.01, 1, .01)
    threshold = float(thresholds[np.argmax([f1_score(vy, candidates[selected] >= v) for v in thresholds])])
    if any(sha256(path) != code_hashes[str(path).replace('\\', '/')] for path in executable_files) or sha256('docs/evaluation-protocol.md') != protocol_hash:
        raise RuntimeError('Executable code/protocol changed during fitting; bundle cannot be frozen')
    files = ['transformer.json', 'lightgbm_31.txt', 'lightgbm_63.txt', 'xgboost.json',
             'logistic.joblib', 'seen_features.joblib']
    manifest = {'schema_version': 2, 'feature_names': FEATURE_NAMES, 'selection': selected,
        'weights': weights, 'threshold': threshold, 'prior': float(y.mean()),
        'validation_log_loss': losses, 'windows': WINDOWS, 'splits': {'train': train_meta, 'validation': val_meta},
        'best_iterations': {name: int(model.best_iteration_ if name.startswith('lightgbm') else model.best_iteration)
                            for name, model in models.items()},
        'files': {file: sha256(args.output / file) for file in files},
        'training_seconds': elapsed, 'total_seconds': time.perf_counter()-started,
        'source_file_bytes': args.source.stat().st_size, 'protocol_sha256': protocol_hash,
        'test_source_sha256': test_source_hash,
        'code_sha256': code_hashes, 'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'joblib')}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'selected': selected, 'validation_log_loss': losses[selected], 'threshold': threshold,
                      'train_rows': len(train), 'validation_rows': len(val)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
