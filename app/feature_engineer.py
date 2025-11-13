"""
Feature Engineering Pipeline for CTR Prediction
Transforms raw 39 features into 150 engineered features
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import logging

# Setup logger
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline that matches the training process
    Transforms raw Criteo features (13 integer + 26 categorical) into 150 engineered features
    """
    
    def __init__(self, artifacts_path=None):
        logger.info("FeatureEngineer initialized")
        self.artifacts = None
        self.feature_mappings = {}
        self.ctr_encodings = {}
        self.frequency_encodings = {}
        self.training_features = None  # The exact 150 features used in training
        
        if artifacts_path and os.path.exists(artifacts_path):
            self.load_artifacts(artifacts_path)
        
        # Load the training feature list
        self.load_training_features()
    
    def load_artifacts(self, artifacts_path):
        """Load pre-computed feature engineering artifacts"""
        try:
            self.artifacts = joblib.load(artifacts_path)
            print(f"✓ Feature engineering artifacts loaded from {artifacts_path}")
            
            # Extract encodings if available
            if 'ctr_encodings' in self.artifacts:
                self.ctr_encodings = self.artifacts['ctr_encodings']
            if 'frequency_encodings' in self.artifacts:
                self.frequency_encodings = self.artifacts['frequency_encodings']
            if 'feature_mappings' in self.artifacts:
                self.feature_mappings = self.artifacts['feature_mappings']
                
        except Exception as e:
            print(f"Warning: Could not load artifacts: {e}")
            print("Will use fallback feature engineering")
    
    def load_training_features(self):
        """Load the exact list of 150 features used during training"""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'models', 'training_features.json'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'training_features.json'),
            'deployment/app/models/training_features.json',
            'app/models/training_features.json',
        ]
        
        for features_file in possible_paths:
            if os.path.exists(features_file):
                try:
                    with open(features_file, 'r') as f:
                        self.training_features = json.load(f)
                    print(f"✓ Loaded {len(self.training_features)} training features from {features_file}")
                    return
                except Exception as e:
                    print(f"Warning: Could not load from {features_file}: {e}")
        
        print("ERROR: Could not find training_features.json in any expected location!")
        print("Will use all engineered features (this will cause feature count mismatch!)")
    
    def engineer_numerical_features(self, df):
        """
        Engineer numerical features (I1-I13)
        Creates: sqrt transforms, binned features, missing indicators, extreme value indicators
        """
        df_eng = df.copy()
        
        # Integer columns
        int_cols = [f'I{i}' for i in range(1, 14)]
        
        for col in int_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0
            
            # Fill missing values with 0
            df_eng[col] = df_eng[col].fillna(0)
            
            # Sqrt transform (for non-negative values)
            df_eng[f'{col}_sqrt'] = np.sqrt(np.maximum(df_eng[col], 0))
            
            # Is zero indicator
            df_eng[f'{col}_is_zero'] = (df_eng[col] == 0).astype(int)
            
            # Missing indicator (if original had nulls)
            df_eng[f'{col}_missing'] = (df_eng[col].isna()).astype(int)
            
            # Is extreme indicator (values beyond 3 standard deviations)
            # Using conservative thresholds for deployment
            mean_val = df_eng[col].mean()
            std_val = df_eng[col].std()
            if std_val > 0:
                df_eng[f'{col}_is_extreme'] = (
                    np.abs(df_eng[col] - mean_val) > 3 * std_val
                ).astype(int)
            else:
                df_eng[f'{col}_is_extreme'] = 0
            
            # Binned features (quartiles)
            try:
                df_eng[f'{col}_binned'] = pd.qcut(
                    df_eng[col], 
                    q=4, 
                    labels=False, 
                    duplicates='drop'
                ).fillna(0).astype(int)
            except:
                df_eng[f'{col}_binned'] = 0
        
        return df_eng
    
    def engineer_categorical_features(self, df):
        """
        Engineer categorical features (C1-C26)
        Creates: CTR encoding, frequency encoding, label encoding, grouped, encoded, combined
        """
        df_eng = df.copy()
        
        # Categorical columns
        cat_cols = [f'C{i}' for i in range(1, 27)]
        
        for col in cat_cols:
            if col not in df_eng.columns:
                df_eng[col] = 'missing'
            
            # Fill missing with 'missing' string
            df_eng[col] = df_eng[col].fillna('missing').astype(str)
            
            # CTR encoding (if available from artifacts)
            if col in self.ctr_encodings:
                df_eng[f'{col}_ctr_encoded'] = df_eng[col].map(
                    self.ctr_encodings[col]
                ).fillna(self.ctr_encodings[col].get('__UNKNOWN__', 0.25))
            else:
                # Fallback: use global CTR
                df_eng[f'{col}_ctr_encoded'] = 0.25
            
            # Frequency encoding (if available)
            if col in self.frequency_encodings:
                df_eng[f'{col}_freq'] = df_eng[col].map(
                    self.frequency_encodings[col]
                ).fillna(0)
            else:
                # Fallback: compute frequency on current batch
                freq = df_eng[col].value_counts(normalize=True)
                df_eng[f'{col}_freq'] = df_eng[col].map(freq).fillna(0)
            
            # Grouped encoding (simple hash to group rare categories)
            df_eng[f'{col}_grouped'] = df_eng[col].apply(hash).abs() % 100
            
            # Encoded version (label encoding)
            unique_vals = df_eng[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_vals)}
            df_eng[f'{col}_encoded'] = df_eng[col].map(label_map).fillna(0).astype(int)
            
            # Keep original as integer too
            df_eng[col] = df_eng[f'{col}_encoded']
        
        # Create combined features
        if 'C1' in df_eng.columns and 'C15' in df_eng.columns:
            df_eng['C1_C15_combined'] = df_eng['C1'].astype(str) + '_' + df_eng['C15'].astype(str)
            df_eng['C1_C15_combined'] = df_eng['C1_C15_combined'].apply(hash).abs() % 1000
            
            # Frequency for combined
            freq = df_eng['C1_C15_combined'].value_counts(normalize=True)
            df_eng['C1_C15_freq'] = df_eng['C1_C15_combined'].map(freq).fillna(0)
        
        if 'C14' in df_eng.columns and 'C17' in df_eng.columns:
            df_eng['C14_C17_combined'] = df_eng['C14'].astype(str) + '_' + df_eng['C17'].astype(str)
            df_eng['C14_C17_combined'] = df_eng['C14_C17_combined'].apply(hash).abs() % 1000
            
            freq = df_eng['C14_C17_combined'].value_counts(normalize=True)
            df_eng['C14_C17_freq'] = df_eng['C14_C17_combined'].map(freq).fillna(0)
        
        if 'C19' in df_eng.columns and 'C20' in df_eng.columns:
            df_eng['C19_C20_combined'] = df_eng['C19'].astype(str) + '_' + df_eng['C20'].astype(str)
            df_eng['C19_C20_combined'] = df_eng['C19_C20_combined'].apply(hash).abs() % 1000
            
            freq = df_eng['C19_C20_combined'].value_counts(normalize=True)
            df_eng['C19_C20_freq'] = df_eng['C19_C20_combined'].map(freq).fillna(0)
        
        return df_eng
    
    def create_interaction_features(self, df):
        """Create interaction features between important columns"""
        df_eng = df.copy()
        
        # Ratio features
        if 'I10' in df_eng.columns and 'I11' in df_eng.columns:
            df_eng['I10_I11_ratio'] = np.where(
                df_eng['I11'] != 0,
                df_eng['I10'] / (df_eng['I11'] + 1e-5),
                0
            )
        
        if 'I6' in df_eng.columns and 'I7' in df_eng.columns:
            df_eng['I6_I7_ratio'] = np.where(
                df_eng['I7'] != 0,
                df_eng['I6'] / (df_eng['I7'] + 1e-5),
                0
            )
        
        if 'I7' in df_eng.columns and 'I8' in df_eng.columns:
            df_eng['I7_I8_ratio'] = np.where(
                df_eng['I8'] != 0,
                df_eng['I7'] / (df_eng['I8'] + 1e-5),
                0
            )
        
        if 'I3' in df_eng.columns and 'I5' in df_eng.columns:
            df_eng['I3_I5_ratio'] = np.where(
                df_eng['I5'] != 0,
                df_eng['I3'] / (df_eng['I5'] + 1e-5),
                0
            )
        
        if 'I1' in df_eng.columns and 'I2' in df_eng.columns:
            df_eng['I1_I2_ratio'] = np.where(
                df_eng['I2'] != 0,
                df_eng['I1'] / (df_eng['I2'] + 1e-5),
                0
            )
        
        return df_eng
    
    def transform(self, df_raw):
        """
        Main transformation method
        Converts raw 39 features to 150 engineered features
        
        Args:
            df_raw: DataFrame with raw features (I1-I13, C1-C26)
        
        Returns:
            DataFrame with 150 engineered features
        """
        logger.info(f"Transforming {len(df_raw)} samples with {len(df_raw.columns)} raw features")
        # Start with raw data
        df = df_raw.copy()
        
        # Apply numerical feature engineering
        df = self.engineer_numerical_features(df)
        
        # Apply categorical feature engineering
        df = self.engineer_categorical_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Select only the 150 features used in training
        if self.training_features is not None:
            print(f"DEBUG: Created {len(df.columns)} features")
            print(f"DEBUG: Need {len(self.training_features)} features")
            
            # Add any missing features with zeros
            missing_features = []
            for feat in self.training_features:
                if feat not in df.columns:
                    df[feat] = 0
                    missing_features.append(feat)
            
            if missing_features:
                print(f"DEBUG: Added {len(missing_features)} missing features")
            
            # Select only training features in the correct order
            df = df[self.training_features]
            print(f"DEBUG: Final feature count: {len(df.columns)}")
        
        return df
    
    def transform_single(self, features_dict):
        """
        Transform a single ad's features
        
        Args:
            features_dict: Dictionary with raw features
        
        Returns:
            DataFrame with engineered features (single row)
        """
        df = pd.DataFrame([features_dict])
        return self.transform(df)
