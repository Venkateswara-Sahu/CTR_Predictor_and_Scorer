"""
CTR Prediction Model - Production Ready
Extracted from modelling.ipynb for deployment
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
import logging
from sklearn.preprocessing import LabelEncoder
from .feature_engineer import FeatureEngineer

# Setup logger
logger = logging.getLogger(__name__)


class CTRPredictor:
    """Production-ready CTR Prediction Model"""
    
    def __init__(self, model_path=None):
        self.lgb_model = None
        self.xgb_model = None
        self.ensemble_weights = {'lightgbm': 0.6, 'xgboost': 0.4}
        self.feature_names = None
        self.encoders = {}
        self.feature_engineer = None
        logger.info("CTRPredictor initialized")
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained models"""
        logger.info(f"Loading models from: {model_path}")
        try:
            # Load LightGBM model
            self.lgb_model = lgb.Booster(model_file=f'{model_path}/lightgbm_model.txt')
            print("✓ LightGBM model loaded")
            logger.info("LightGBM model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load LightGBM model: {e}")
            logger.warning(f"Could not load LightGBM model: {e}")
        
        try:
            # Load XGBoost model
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(f'{model_path}/xgboost_model.json')
            print("✓ XGBoost model loaded")
            logger.info("XGBoost model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load XGBoost model: {e}")
            logger.warning(f"Could not load XGBoost model: {e}")
        
        # Load feature engineering artifacts
        artifacts_path = os.path.join(model_path, 'feature_engineering_artifacts.pkl')
        if os.path.exists(artifacts_path):
            try:
                self.feature_engineer = FeatureEngineer(artifacts_path)
                print("✓ Feature engineering pipeline loaded")
                logger.info("Feature engineering pipeline loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load feature engineering: {e}")
                print("Will use basic preprocessing")
                logger.warning(f"Could not load feature engineering: {e}")
        
        if not self.lgb_model and not self.xgb_model:
            logger.error("No models could be loaded!")
            raise ValueError("No models could be loaded!")
    
    def preprocess_features(self, df):
        """Preprocess features for prediction"""
        df_processed = df.copy()
        
        # Convert any object columns to numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                if col not in self.encoders:
                    unique_vals = df_processed[col].unique()
                    self.encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
                    self.encoders[col]['__UNKNOWN__'] = len(unique_vals)
                
                df_processed[col] = df_processed[col].map(
                    lambda x: self.encoders[col].get(x, self.encoders[col]['__UNKNOWN__'])
                ).astype(int)
        
        # Ensure all columns are numeric
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
        return df_processed
    
    def predict(self, X, return_probability=True):
        """
        Make CTR predictions
        
        Args:
            X: DataFrame with features (raw or engineered)
            return_probability: If True, return probabilities; else binary predictions
        
        Returns:
            Array of predictions
        """
        # If we have feature engineer and X has raw features, transform them
        if self.feature_engineer is not None and len(X.columns) < 100:
            # Looks like raw features (39 columns), engineer them
            X_processed = self.feature_engineer.transform(X)
        else:
            # Already engineered or no feature engineer available
            X_processed = self.preprocess_features(X)
        
        predictions = []
        
        # LightGBM prediction
        if self.lgb_model:
            lgb_pred = self.lgb_model.predict(X_processed)
            predictions.append(lgb_pred)
        
        # XGBoost prediction
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict_proba(X_processed)[:, 1]
            predictions.append(xgb_pred)
        
        # Ensemble
        if len(predictions) > 1:
            weights = [self.ensemble_weights['lightgbm'], self.ensemble_weights['xgboost']]
            ensemble_pred = np.average(predictions, axis=0, weights=weights[:len(predictions)])
        else:
            ensemble_pred = predictions[0]
        
        if return_probability:
            return ensemble_pred
        else:
            return (ensemble_pred >= 0.5).astype(int)
    
    def predict_batch(self, X_list):
        """Predict for multiple ad instances"""
        df = pd.DataFrame(X_list)
        return self.predict(df)
