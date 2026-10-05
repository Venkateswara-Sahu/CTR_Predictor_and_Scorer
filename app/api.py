"""Application factory; startup fails closed if the compatible bundle is absent."""
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, UnsupportedMediaType
from .ad_ranking_system import AdRankingSystem


def create_app(model_path=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
    CORS(app)
    ranking = AdRankingSystem(model_path or Path(__file__).parent / 'models')
    app.extensions['ranking_system'] = ranking

    def payload_rows(key):
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Provide a JSON object')
        rows = data.get(key)
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 1000 or not all(isinstance(row, dict) and row for row in rows):
            raise ValueError(f'{key} must contain 1..1000 nonempty raw-feature objects')
        try:
            return data, pd.DataFrame(rows)
        except OverflowError as error:
            raise ValueError('Numeric input exceeds the supported range') from error

    @app.errorhandler(ValueError)
    @app.errorhandler(BadRequest)
    @app.errorhandler(UnsupportedMediaType)
    def bad_input(error):
        return jsonify(error=str(error)), 400

    @app.get('/')
    def home():
        return jsonify(version='2.0', score_definition='predicted click probability',
                       endpoints=['/health', '/predict_ctr', '/rank_ads', '/predict_single'])

    @app.get('/health')
    def health():
        return jsonify(status='healthy', schema_version=2,
                       selection=ranking.ctr_predictor.manifest['selection'])

    @app.post('/predict_ctr')
    def predict_ctr():
        _, frame = payload_rows('features')
        values = ranking.ctr_predictor.predict(frame)
        return jsonify(predictions=values.tolist(), count=len(values))

    @app.post('/rank_ads')
    def rank_ads():
        data, frame = payload_rows('ads')
        top_k = data.get('top_k', len(frame))
        if type(top_k) is not int or not 1 <= top_k <= len(frame):
            raise ValueError('top_k must be an integer between 1 and the number of ads')
        results = ranking.rank_ads(frame, return_details=False).head(top_k)
        return jsonify(ranked_ads=results.to_dict(orient='records'), total_ads=len(frame), returned_ads=len(results))

    @app.post('/predict_single')
    def single():
        _, frame = payload_rows('features')
        if len(frame) != 1:
            raise ValueError('predict_single requires one row')
        return jsonify(ranking.predict_single_ad(frame.iloc[0].to_dict()))

    @app.post('/score_ad')
    def retired():
        return jsonify(error='Semantic ad quality is not established. Use /predict_ctr or /rank_ads.'), 410

    return app
