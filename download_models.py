"""
Download model files from cloud storage on app startup
"""
import os
import urllib.request
import streamlit as st

# Model file URLs from GitHub Release v1.0
MODEL_URLS = {
    'lightgbm_model.txt': 'https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer/releases/download/v1.0/lightgbm_model.txt',
    'xgboost_model.json': 'https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer/releases/download/v1.0/xgboost_model.json',
    'feature_engineering_artifacts.pkl': 'https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer/releases/download/v1.0/feature_engineering_artifacts.pkl',
    'training_features.json': 'https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer/releases/download/v1.0/training_features.json',
}

def download_models():
    """Download models if they don't exist locally"""
    models_dir = os.path.join(os.path.dirname(__file__), 'app', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath) and url != 'YOUR_GOOGLE_DRIVE_OR_DROPBOX_LINK_HERE':
            try:
                st.info(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                st.success(f"✓ Downloaded {filename}")
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
                return False
    
    return True

if __name__ == "__main__":
    download_models()
