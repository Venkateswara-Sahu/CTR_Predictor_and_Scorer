"""
Ad Ranking System - Combines CTR Prediction and Quality Scoring
"""

import numpy as np
import pandas as pd
import logging
from .ctr_model import CTRPredictor
from .ad_scorer import AdScorer

# Setup logger
logger = logging.getLogger(__name__)


class AdRankingSystem:
    """
    Complete ad ranking system integrating CTR prediction and ad quality scoring
    """
    
    def __init__(self, model_path=None, weights=None):
        logger.info("Initializing AdRankingSystem")
        self.ctr_predictor = CTRPredictor(model_path)
        self.ad_scorer = AdScorer()
        
        self.weights = weights or {
            'quality_score': 0.4,
            'ctr_prediction': 0.6
        }
        logger.info("AdRankingSystem initialized successfully")
    
    def rank_ads(self, ad_data, return_details=False):
        """
        Rank ads based on combined CTR prediction and quality score
        
        Args:
            ad_data: DataFrame or list of dicts containing ad features
            return_details: If True, return all intermediate scores
        
        Returns:
            Ranked DataFrame or list of ranked ad IDs with scores
        """
        # Convert to DataFrame if needed
        if isinstance(ad_data, list):
            df = pd.DataFrame(ad_data)
        else:
            df = ad_data.copy()
        
        # Get CTR predictions
        ctr_predictions = self.ctr_predictor.predict(df)
        
        # Get quality scores
        quality_scores = self.ad_scorer.compute_ad_score(df)
        
        # Compute final scores
        final_scores = (
            self.weights['quality_score'] * quality_scores +
            self.weights['ctr_prediction'] * ctr_predictions
        )
        
        # Create results
        if return_details:
            results = df.copy()
            results['predicted_ctr'] = ctr_predictions
            results['quality_score'] = quality_scores
            results['final_score'] = final_scores
            results = results.sort_values('final_score', ascending=False)
        else:
            results = pd.DataFrame({
                'ad_id': range(len(df)),
                'final_score': final_scores,
                'predicted_ctr': ctr_predictions,
                'quality_score': quality_scores
            }).sort_values('final_score', ascending=False)
        
        return results
    
    def predict_single_ad(self, ad_features):
        """
        Predict CTR and quality score for a single ad
        
        Args:
            ad_features: Dict containing ad features
        
        Returns:
            Dict with predicted_ctr, quality_score, and final_score
        """
        df = pd.DataFrame([ad_features])
        
        ctr = float(self.ctr_predictor.predict(df)[0])
        quality = float(self.ad_scorer.compute_ad_score(df).iloc[0])
        final_score = (
            self.weights['quality_score'] * quality +
            self.weights['ctr_prediction'] * ctr
        )
        
        return {
            'predicted_ctr': ctr,
            'quality_score': quality,
            'final_score': final_score
        }
