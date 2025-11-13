"""
Flask REST API for CTR Prediction and Ad Scoring System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from app.ad_ranking_system import AdRankingSystem

app = Flask(__name__)
CORS(app)

# Initialize the ad ranking system
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
ranking_system = AdRankingSystem(model_path=MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'CTR Prediction & Ad Scoring API',
        'version': '1.0',
        'endpoints': {
            '/health': 'Health check',
            '/predict_ctr': 'Predict CTR for ad features (POST)',
            '/score_ad': 'Score ad quality (POST)',
            '/rank_ads': 'Rank multiple ads (POST)',
            '/predict_single': 'Complete prediction for single ad (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'ctr_model': ranking_system.ctr_predictor.lgb_model is not None,
            'ad_scorer': True
        }
    })

@app.route('/predict_ctr', methods=['POST'])
def predict_ctr():
    """
    Predict CTR for given ad features
    
    Expected JSON format:
    {
        "features": {
            "I1": 1, "I2": 2, ..., "C1": "value1", "C2": "value2", ...
        }
    }
    or
    {
        "features": [
            {"I1": 1, "I2": 2, ...},
            {"I1": 3, "I2": 4, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        features = data['features']
        
        # Handle single or multiple predictions
        if isinstance(features, dict):
            features = [features]
        
        df = pd.DataFrame(features)
        predictions = ranking_system.ctr_predictor.predict(df)
        
        result = {
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else [float(predictions)],
            'count': len(predictions)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/score_ad', methods=['POST'])
def score_ad():
    """
    Score ad quality
    
    Expected JSON format:
    {
        "features": {
            "I1": 1, "I2": 2, ..., "C1": "value1", ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        features = data['features']
        
        if isinstance(features, list):
            df = pd.DataFrame(features)
            scores = ranking_system.ad_scorer.compute_ad_score(df)
            result = {
                'quality_scores': scores.tolist(),
                'count': len(scores)
            }
        else:
            score = ranking_system.ad_scorer.score_single_ad(features)
            result = {
                'quality_score': score
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rank_ads', methods=['POST'])
def rank_ads():
    """
    Rank multiple ads
    
    Expected JSON format:
    {
        "ads": [
            {"I1": 1, "I2": 2, "C1": "value1", ...},
            {"I1": 3, "I2": 4, "C1": "value2", ...}
        ],
        "top_k": 10  # optional, return top K ads
    }
    """
    try:
        data = request.get_json()
        
        if 'ads' not in data:
            return jsonify({'error': 'Missing "ads" in request'}), 400
        
        ads = data['ads']
        top_k = data.get('top_k', None)
        
        # Rank ads
        ranked_df = ranking_system.rank_ads(ads, return_details=True)
        
        if top_k:
            ranked_df = ranked_df.head(top_k)
        
        result = {
            'ranked_ads': ranked_df.to_dict(orient='records'),
            'total_ads': len(ads),
            'returned_ads': len(ranked_df)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """
    Complete prediction for a single ad (CTR + Quality Score + Final Rank)
    
    Expected JSON format:
    {
        "features": {
            "I1": 1, "I2": 2, ..., "C1": "value1", ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        features = data['features']
        result = ranking_system.predict_single_ad(features)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
