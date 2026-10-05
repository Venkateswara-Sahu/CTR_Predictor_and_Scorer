import json
import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from app.feature_engineer import FeatureEngineer, FEATURE_NAMES
from app.ctr_model import CTRPredictor
from app.ad_ranking_system import AdRankingSystem


@pytest.fixture
def bundle(tmp_path):
    from training.data import sha256
    raw = pd.DataFrame({'I1': np.arange(40), 'C1': ['a', 'b'] * 20})
    y = np.arange(40) % 2
    engineer = FeatureEngineer()
    x = engineer.fit_transform(raw, y)
    engineer.save(tmp_path / 'transformer.json')
    model = lgb.LGBMClassifier(n_estimators=3, min_child_samples=2, verbosity=-1, n_jobs=1).fit(x, y)
    model.booster_.save_model(str(tmp_path / 'lightgbm_31.txt'))
    manifest = {'schema_version': 2, 'feature_names': FEATURE_NAMES, 'selection': 'lightgbm_31',
                'weights': {'lightgbm_31': 1.0}, 'threshold': .4, 'prior': .5,
                'files': {name: sha256(tmp_path / name) for name in ('transformer.json', 'lightgbm_31.txt')}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_predictions_and_ranking_are_batch_independent(bundle):
    predictor = CTRPredictor(bundle)
    raw = pd.DataFrame([{'I1': 3, 'C1': 'a'}, {'I1': 99, 'C1': 'unseen'}])
    whole = predictor.predict(raw)
    np.testing.assert_allclose(whole[0], predictor.predict(raw.iloc[[0]])[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(whole, predictor.predict(raw.iloc[::-1])[::-1], rtol=0, atol=1e-12)
    ranked = AdRankingSystem(bundle).rank_ads(raw, return_details=True)
    np.testing.assert_allclose(ranked.final_score, ranked.predicted_ctr)
    assert 'quality_score' not in ranked.columns
    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame([{'click': 0}]))


def test_corruption_and_old_assets_fail_closed(bundle, tmp_path):
    (bundle / 'transformer.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        CTRPredictor(bundle)
    with pytest.raises((ValueError, FileNotFoundError)):
        CTRPredictor(tmp_path / 'missing')


def test_api_validation_and_health(bundle):
    from app.api import create_app
    client = create_app(bundle).test_client()
    assert client.get('/health').get_json()['schema_version'] == 2
    assert client.post('/predict_ctr', json={'features': {'I1': 3}}).status_code == 200
    for data in (None, {}, {'features': []}, {'features': [{'I1': 'bad'}]}, {'features': 1}):
        assert client.post('/predict_ctr', json=data).status_code == 400
    assert client.post('/rank_ads', json={'ads': [{'I1': 3}], 'top_k': -1}).status_code == 400
    assert client.post('/score_ad', json={'features': {'I1': 1}}).status_code == 410
    assert client.post('/predict_ctr', json={'features': {'I1': 10**400}}).status_code == 400
