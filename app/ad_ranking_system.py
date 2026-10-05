"""Candidate ranking by predicted click probability, without semantic quality proxies."""
import pandas as pd
from .ctr_model import CTRPredictor


class AdRankingSystem:
    def __init__(self, model_path=None, weights=None):
        if weights is not None and weights != {'ctr_prediction': 1.0}:
            raise ValueError('v2 ranks by CTR only; anonymous fields do not establish ad quality')
        self.ctr_predictor = CTRPredictor(model_path)

    def rank_ads(self, ad_data, return_details=False):
        frame = pd.DataFrame(ad_data).copy()
        probabilities = self.ctr_predictor.predict(frame)
        results = frame.copy() if return_details else pd.DataFrame(index=frame.index)
        results['ad_id'] = range(len(frame))
        results['predicted_ctr'] = probabilities
        results['final_score'] = probabilities
        return results.sort_values('final_score', ascending=False, kind='stable')

    def predict_single_ad(self, ad_features):
        probability = float(self.ctr_predictor.predict(pd.DataFrame([ad_features]))[0])
        return {'predicted_ctr': probability, 'final_score': probability,
                'score_definition': 'predicted click probability'}
