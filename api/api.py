"""
CPX Prediction API

This module provides a Flask-based REST API for predicting campaign metrics.
It uses Redis for model storage and caching, and provides endpoints for
predicting various campaign metrics based on input parameters.
"""

import datetime
import io
import logging
import math
import os
from typing import Dict, Any, Union, List

import joblib
import pandas as pd
import redis
import torch
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', ''),
    db=int(os.getenv('REDIS_DB', 0)),
    decode_responses=False  # We need bytes for model storage
)

# Initialize Flask app
app = Flask(__name__)

# Constants
MODEL_KEYS = {
    'impressions': 'impressions_model',
    'clicks': 'clicks_model',
    'purchases': 'purchases_model',
    'cost_per_impression': 'cost_per_impression_model',
    'cost_per_click': 'cost_per_click_model',
    'cost_per_purchase': 'cost_per_purchase_model'
}
COLUMNS_KEY = 'model_columns'


app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping() -> str:
    """Health check endpoint to verify the API is running.
    
    Returns:
        str: Confirmation message indicating the server is running.
    """
    try:
        # Verify Redis connection
        redis_client.ping()
        return jsonify({
            'status': 'healthy',
            'message': 'Server is running and connected to Redis',
            'timestamp': datetime.datetime.utcnow().isoformat()
        }), 200
    except redis.RedisError as e:
        logger.error(f"Redis connection error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Redis connection failed',
            'error': str(e)
        }), 500


@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
        data = {}
        data['cost'] = int(request.form['budget'])
        start_month, start_day, start_year = \
            [int(i) for i in request.form['start_date'].split('/')]
        end_month, end_day, end_year = \
            [int(i) for i in request.form['end_date'].split('/')]
        data['start_month'] = start_month
        data['end_month'] = end_month
        start_date = datetime.date(start_year, start_month, start_day)
        end_date = datetime.date(end_year, end_month, end_day)
        data['start_week'] = start_date.isocalendar()[1]
        data['end_week'] = end_date.isocalendar()[1]
        data['days'] = (end_date - start_date).days
        data['ticket_capacity'] = int(request.form['capacity'])
        data['average_ticket_price'] = int(request.form['price'])
        for channel in ['facebook', 'instagram',
                        'google_search', 'google_display']:
            try:
                if request.form[channel] == 'on':
                    data[channel] = 1
            except KeyError:
                data[channel] = 0
        data['facebook_likes'] = int(request.form['likes'])
        data['region_' + request.form['region'].lower()] = 1
        # try:
        #     if request.form['targets'] == 'on':
        #         data['locality_single'] = 0
        # except KeyError:
        #     data['locality_single'] = 1
        data['category_' + request.form['category'].lower()] = 1
        data['shop_' + request.form['shop'].lower()] = 1
        predictions = predict_metrics(data)
        impressions_low = int(round_down(predictions['impressions'] * 0.8, -4))
        impressions_high = int(round_up(predictions['impressions'] * 1.2, -4))
        clicks_low = int(round_down(predictions['clicks'] * 0.8, -2))
        clicks_high = int(round_up(predictions['clicks'] * 1.2, -2))
        purchases_low = int(round_down(predictions['purchases'] * 0.8, -1))
        purchases_high = int(round_up(predictions['purchases'] * 1.2, -1))
        return render_template('index.html', scroll='results',
                               impressions_low=f'{impressions_low:,}',
                               impressions_high=f'{impressions_high:,}',
                               clicks_low=f'{clicks_low:,}',
                               clicks_high=f'{clicks_high:,}',
                               purchases_low=f'{purchases_low:,}',
                               purchases_high=f'{purchases_high:,}')

    return render_template('index.html')


@app.route('/<metric>', methods=['POST'])
def metric_prediction(metric):
    if metric not in ['impressions', 'clicks', 'purchases',
                      'cost_per_impression', 'cost_per_click',
                      'cost_per_purchase']:
        return 'Metric "' + metric + '" not supported.'
    data = request.json
    data = format_categoricals(data)
    prediction = int(predict([data], metric))
    return jsonify({metric: prediction})


@app.route('/campaign', methods=['POST'])
def campaign_prediction():
    data = request.json
    data = format_categoricals(data)
    predictions = predict_metrics(data)
    return jsonify(predictions)


def round_up(x, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(x * multiplier) / multiplier


def round_down(x, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(x * multiplier) / multiplier


def format_categoricals(data):
    categoricals = ['category', 'region', 'shop', 'locality']
    for cat in categoricals:
        if cat in data:
            data[cat + '_' + data[cat].lower()] = 1
            del data[cat]
    return data


def predict_metrics(data):
    predictions = {}
    for metric in ['impressions', 'clicks', 'purchases']:
        direct = int(predict([data], metric))
        cpx = int(data['cost'] / predict([data], 'cost_per_' + metric[0:-1]))
        trans = int(predict([{'direct': direct, 'cpx': cpx}],
                            metric + '_transfer'))
        predictions[metric] = trans
    return predictions


def predict(data: Union[Dict, List[Dict]], output: str) -> float:
    """Make predictions using the specified model.
    
    Args:
        data: Input data for prediction (single dict or list of dicts)
        output: The type of prediction to make (e.g., 'impressions', 'clicks')
        
    Returns:
        float: The predicted value
        
    Raises:
        ValueError: If the output type is invalid or model loading fails
    """
    try:
        # Get model and columns from Redis
        model_key = MODEL_KEYS.get(output)
        if not model_key:
            raise ValueError(f"Invalid output type: {output}")
            
        columns_key = f"{output}_columns"
        
        # Load model and columns
        model = load_model_from_redis(model_key)
        columns = load_model_from_redis(columns_key)
        
        # Prepare input data
        if isinstance(data, dict):
            data = [data]
            
        df = pd.DataFrame(data).reindex(columns=columns, fill_value=0)
        
        # Make prediction - handle PyTorch models differently
        if hasattr(model, 'predict'):
            # Standard scikit-learn interface
            prediction = model.predict(df)[0]
        elif isinstance(model, torch.nn.Module):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                df_tensor = torch.FloatTensor(df.values)
                prediction = model(df_tensor).item()
        else:
            # Try standard predict method
            prediction = model.predict(df)[0]
        
        return float(prediction)
        
    except Exception as e:
        logger.error(f"Prediction error for {output}: {str(e)}")
        raise


def load_model_from_redis(model_key: str):
    """Load a model from Redis cache.
    
    Args:
        model_key (str): The key under which the model is stored in Redis
        
    Returns:
        The deserialized model object
        
    Raises:
        redis.RedisError: If there's an issue connecting to Redis
        Exception: If model loading fails
    """
    try:
        # Check if model exists in Redis
        if not redis_client.exists(model_key):
            logger.error(f"Model not found in Redis: {model_key}")
            raise ValueError(f"Model {model_key} not found in Redis")
            
        # Load model from Redis
        model_data = redis_client.get(model_key)
        if not model_data:
            raise ValueError(f"Empty model data for key: {model_key}")
            
        # Deserialize the model
        # Try joblib first (for scikit-learn, xgboost, etc.)
        try:
            model = joblib.load(io.BytesIO(model_data))
            return model
        except Exception as e:
            # If joblib fails, try PyTorch
            try:
                model = torch.load(io.BytesIO(model_data), map_location='cpu')
                model.eval()  # Set to evaluation mode
                return model
            except Exception as e2:
                logger.error(f"Failed to load model with both joblib and PyTorch: {str(e)}, {str(e2)}")
                raise
        
    except Exception as e:
        logger.error(f"Error loading model {model_key} from Redis: {str(e)}")
        raise


if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.getenv('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')
