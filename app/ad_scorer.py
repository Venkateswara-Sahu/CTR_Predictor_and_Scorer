"""
Ad Scoring System - Production Ready
Extracted from ad_scoring_and_ctr.ipynb for deployment
"""

import numpy as np
import pandas as pd
import logging

# Setup logger
logger = logging.getLogger(__name__)


class AdScorer:
    """
    Computes ad quality scores based on multiple factors
    """
    
    def __init__(self, weights=None):
        logger.info("AdScorer initialized")
        self.weights = weights or {
            'historical_ctr': 0.3,
            'placement_quality': 0.2,
            'device_compatibility': 0.15,
            'time_relevance': 0.15,
            'engagement': 0.2
        }
        self.historical_stats = {}
    
    def compute_historical_ctr(self, df, group_cols):
        """Compute historical CTR for different ad attributes"""
        if 'click' not in df.columns:
            return
            
        for col in group_cols:
            if col in df.columns:
                self.historical_stats[f'{col}_ctr'] = (
                    df.groupby(col)['click']
                    .agg(['count', 'mean'])
                    .rename(columns={'count': 'impressions', 'mean': 'ctr'})
                )
    
    def calculate_placement_score(self, df):
        """Calculate ad placement quality score"""
        placement_score = pd.Series(0.5, index=df.index)
        
        if 'I1' in df.columns:
            i1_values = df['I1'].fillna(-1)
            if i1_values.max() > i1_values.min():
                placement_score = (i1_values - i1_values.min()) / (i1_values.max() - i1_values.min())
        
        return placement_score
    
    def calculate_device_score(self, df):
        """Calculate device compatibility score"""
        device_score = pd.Series(0.5, index=df.index)
        
        if 'C1' in df.columns and 'C1_ctr' in self.historical_stats:
            device_score = df['C1'].map(
                self.historical_stats['C1_ctr']['ctr']
            ).fillna(0.5)
        
        return device_score
    
    def calculate_time_score(self, df):
        """Calculate time relevance score"""
        time_score = pd.Series(0.5, index=df.index)
        
        if 'I2' in df.columns:
            i2_values = df['I2'].fillna(-1)
            if i2_values.max() > i2_values.min():
                time_score = (i2_values - i2_values.min()) / (i2_values.max() - i2_values.min())
        
        return time_score
    
    def calculate_engagement_score(self, df):
        """Calculate engagement score based on historical metrics"""
        engagement_score = pd.Series(0.5, index=df.index)
        
        for col, stats in self.historical_stats.items():
            feature = col.replace('_ctr', '')
            if feature in df.columns:
                ctr = df[feature].map(stats['ctr']).fillna(stats['ctr'].mean())
                impressions = df[feature].map(stats['impressions']).fillna(stats['impressions'].mean())
                
                norm_impressions = (impressions - impressions.min()) / (impressions.max() - impressions.min() + 1e-8)
                engagement_score = 0.7 * ctr + 0.3 * norm_impressions
        
        return engagement_score
    
    def compute_ad_score(self, df):
        """Compute final ad quality score"""
        placement_score = self.calculate_placement_score(df)
        device_score = self.calculate_device_score(df)
        time_score = self.calculate_time_score(df)
        engagement_score = self.calculate_engagement_score(df)
        
        final_score = (
            self.weights['placement_quality'] * placement_score +
            self.weights['device_compatibility'] * device_score +
            self.weights['time_relevance'] * time_score +
            self.weights['engagement'] * engagement_score
        )
        
        # Normalize to 0-1 range
        if final_score.max() > final_score.min():
            final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min())
        
        return final_score
    
    def score_single_ad(self, ad_features):
        """Score a single ad"""
        df = pd.DataFrame([ad_features])
        score = self.compute_ad_score(df)
        return float(score.iloc[0])
