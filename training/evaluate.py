"""One final evaluation of an already-frozen bundle; no tuning here."""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss,
                            f1_score, log_loss, roc_auc_score)

from app.ctr_model import CTRPredictor
from app.feature_engineer import RAW_FEATURES
from training.data import load_split, sha256


def lift(y, probability):
    k = math.ceil(len(y) / 10)
    indices = np.argsort(-probability, kind='stable')[:k]
    positives = int(y[indices].sum())
    precision = positives / k
    return {'selected_rows': k, 'clicked_rows': positives, 'precision': precision,
            'population_click_rate': float(y.mean()), 'lift': float(precision / y.mean())}


def metrics(y, probability, threshold):
    return {'roc_auc': float(roc_auc_score(y, probability)),
            'average_precision': float(average_precision_score(y, probability)),
            'log_loss': float(log_loss(y, probability)),
            'brier': float(brier_score_loss(y, probability)),
            'f1': float(f1_score(y, probability >= threshold)),
            'threshold': threshold, 'top_decile': lift(y, probability)}


def bootstrap(y, prediction, global_prediction):
    rng = np.random.default_rng(42)
    samples = {name: [] for name in ('roc_auc', 'log_loss', 'brier', 'top_decile_lift', 'log_loss_minus_global')}
    for i in range(200):
        ix = rng.integers(0, len(y), len(y))
        by, bp = y[ix], prediction[ix]
        loss = log_loss(by, bp)
        samples['roc_auc'].append(roc_auc_score(by, bp))
        samples['log_loss'].append(loss)
        samples['brier'].append(brier_score_loss(by, bp))
        samples['top_decile_lift'].append(lift(by, bp)['lift'])
        samples['log_loss_minus_global'].append(loss - log_loss(by, global_prediction[ix]))
        if (i+1) % 50 == 0:
            print(f'Bootstrap {i+1}/200', flush=True)
    return {key: np.quantile(value, [.025, .975]).tolist() for key, value in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path('artifacts/v2-final'))
    args = parser.parse_args()
    root = args.bundle
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for file, digest in manifest['files'].items():
        if sha256(root / file) != digest:
            raise ValueError(f'Frozen asset changed: {file}')
    if sha256(root / 'data/test.tsv') != manifest['test_source_sha256']:
        raise ValueError('Frozen test source changed')
    if sha256('docs/evaluation-protocol.md') != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    # Preprocessing and training implementation must be the frozen source.
    for file, digest in manifest['code_sha256'].items():
        if sha256(file) != digest:
            raise ValueError(f'Frozen code changed: {file}')
    # Atomic marker prevents silently repeating final evaluation after seeing results.
    with (root / 'evaluation_started.json').open('x', encoding='utf-8') as stream:
        json.dump({'manifest_sha256': sha256(manifest_path), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, stream)
    started = time.perf_counter()
    predictor = CTRPredictor(root)
    seen = joblib.load(root / 'seen_features.joblib')
    test, test_meta = load_split(root / 'data/test.tsv', seen, manifest['windows']['test'][0])
    y = test.click.to_numpy()
    print(f'Final test: {len(test):,} retained rows', flush=True)
    raw = test[RAW_FEATURES]
    features = predictor.feature_engineer.transform(raw)
    probabilities = {'global_rate': np.full(len(test), manifest['prior'])}
    probabilities['logistic'] = joblib.load(root / 'logistic.joblib').predict_proba(features)[:, 1]
    for name in ('lightgbm_31', 'lightgbm_63'):
        model = lgb.Booster(model_file=str(root / f'{name}.txt'))
        probabilities[name] = model.predict(features, num_threads=4)
    model = xgb.XGBClassifier(n_jobs=4)
    model.load_model(root / 'xgboost.json')
    probabilities['xgboost'] = model.predict_proba(features)[:, 1]
    probabilities['selected'] = predictor.predict(raw)
    expected = sum(weight * probabilities[name] for name, weight in manifest['weights'].items())
    np.testing.assert_allclose(expected, probabilities['selected'], atol=1e-12, rtol=0)
    # Check actual trained serving path against single/batch/permutation inputs.
    for index in (0, 17, 99):
        np.testing.assert_allclose(predictor.predict(raw.iloc[[index]])[0], probabilities['selected'][index], atol=1e-12, rtol=0)
    np.testing.assert_allclose(predictor.predict(raw.iloc[:100].iloc[::-1])[::-1], probabilities['selected'][:100], atol=1e-12, rtol=0)
    result = {name: metrics(y, pred, manifest['threshold']) for name, pred in probabilities.items()}
    calibration = []
    bin_ids = np.minimum((probabilities['selected'] * 10).astype(int), 9)
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            calibration.append({'lower': bin_id/10, 'upper': (bin_id+1)/10, 'rows': int(mask.sum()),
                'mean_prediction': float(probabilities['selected'][mask].mean()), 'observed_click_rate': float(y[mask].mean())})
    ci = bootstrap(y, probabilities['selected'], probabilities['global_rate'])
    latency = {}
    for size in (1, 100):
        batch = raw.iloc[:size]
        for _ in range(5):
            predictor.predict(batch)
        durations = []
        for _ in range(30):
            t = time.perf_counter()
            predictor.predict(batch)
            durations.append((time.perf_counter()-t)*1000)
        latency[str(size)] = {'batch_rows': size, 'repetitions': 30, 'warmups': 5,
                             'median_ms': float(np.median(durations)), 'p95_ms': float(np.quantile(durations, .95))}
    frame = pd.DataFrame({'source_row': test.source_row, 'label': y, **probabilities})
    frame.to_csv(root / 'test_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    report = {'schema_version': 2, 'selection': manifest['selection'], 'manifest_sha256': sha256(manifest_path),
        'splits': {**manifest['splits'], 'test': test_meta}, 'metrics': result, 'selected_ci95': ci,
        'bootstrap': {'repetitions': 200, 'seed': 42, 'unit': 'row', 'method': 'paired percentile'},
        'calibration_bins': calibration, 'local_warm_latency': latency,
        'hardware': {'processor': platform.processor(), 'logical_cpus': psutil.cpu_count(),
                     'ram_bytes': psutil.virtual_memory().total},
        'measurement_process_rss_bytes': psutil.Process().memory_info().rss,
        'evaluation_seconds': time.perf_counter()-started,
        'predictions_sha256': sha256(root / 'test_predictions.csv.gz'),
        'evaluation_code_sha256': sha256(__file__), 'serving_code_sha256': sha256('app/ctr_model.py'),
        'limitations': ['Bounded sample; no full-dataset benchmark.',
            'Row windows are not verified chronological splits.',
            'Independent-row confidence intervals omit advertiser/time clustering.',
            'Top-decile lift is predictive enrichment, not measured causal CTR or revenue improvement.',
            'Local warm timings exclude loading/network/concurrency; no production SLA.',
            'Categorical hashing has collisions; unseen category target/frequency values use training prior/zero.',
            'No representative live traffic, fairness, or field semantics have been established.']}
    (root / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'selected_metrics': result['selected'], 'ci95': ci, 'latency': latency}, indent=2), flush=True)


if __name__ == '__main__':
    main()
