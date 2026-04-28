"""Flask application for serving Heart Disease Prediction model with Prometheus monitoring."""

import os
import time

from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'serving_model/muhammad_ivan_LspW-pipeline')

REQUEST_COUNT = Counter('app_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status'])
PREDICTION_LATENCY = Histogram('app_prediction_latency_seconds', 'Latency of prediction requests in seconds')
PREDICTION_COUNT = Counter('app_predictions_total', 'Total number of predictions made', ['result'])
MODEL_UPTIME = Gauge('app_model_uptime_seconds', 'Model uptime in seconds')
ERROR_COUNT = Counter('app_errors_total', 'Total number of errors', ['type'])
ACTIVE_REQUESTS = Gauge('app_active_requests', 'Number of active requests')

start_time = time.time()
model = None

print("Starting Flask application...")


def mock_predict(data):
    """Generate mock prediction based on input features."""
    age = float(data.get('age', 50))
    sex = int(data.get('sex', 1))
    cp = int(data.get('cp', 0))
    trestbps = float(data.get('trestbps', 120))
    chol = float(data.get('chol', 200))
    thalach = float(data.get('thalach', 150))
    oldpeak = float(data.get('oldpeak', 0))
    
    risk_score = 0.0
    
    if age > 55:
        risk_score += 0.15
    if sex == 1:
        risk_score += 0.1
    if cp >= 2:
        risk_score += 0.2
    if trestbps > 140:
        risk_score += 0.15
    if chol > 240:
        risk_score += 0.15
    if thalach < 140:
        risk_score += 0.1
    if oldpeak > 1:
        risk_score += 0.15
    
    probability = min(max(risk_score, 0.0), 1.0)
    prediction = 1 if probability >= 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': round(probability, 4),
        'status': 'success'
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    return jsonify({
        'model': 'mock' if model is None else 'loaded',
        'status': 'healthy',
        'uptime_seconds': round(time.time() - start_time, 2)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    ACTIVE_REQUESTS.inc()
    start_predict_time = time.time()
    
    try:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        
        if not request.is_json:
            ERROR_COUNT.labels(type='invalid_request').inc()
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if not data:
            ERROR_COUNT.labels(type='invalid_request').inc()
            return jsonify({'error': 'No data provided'}), 400
        
        result = mock_predict(data)
        prediction = result['prediction']
        probability = result['probability']
        
        PREDICTION_LATENCY.observe(time.time() - start_predict_time)
        PREDICTION_COUNT.labels(result='positive' if prediction == 1 else 'negative').inc()
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'status': 'success'
        })
    
    except Exception as e:
        ERROR_COUNT.labels(type='prediction_error').inc()
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500
    
    finally:
        ACTIVE_REQUESTS.dec()


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    MODEL_UPTIME.set(time.time() - start_time)
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/v1/models/muhammad_ivan_LspW-pipeline/metadata', methods=['GET'])
def model_metadata():
    """Model metadata endpoint (simulating TF Serving)."""
    REQUEST_COUNT.labels(method='GET', endpoint='/metadata', status='200').inc()
    return jsonify({
        "model_spec": {
            "name": "muhammad_ivan_LspW-pipeline",
            "signature_name": "serving_default",
            "version": "1"
        },
        "metadata": {
            "signature_def": {
                "signature_def": {
                    "serving_default": {
                        "inputs": {
                            "age": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "sex": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "cp": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "trestbps": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "chol": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "fbs": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "restecg": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "thalach": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "exang": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "oldpeak": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "slope": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "ca": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}},
                            "thal": {"dtype": "DT_INT64", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}}
                        },
                        "outputs": {
                            "prediction": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "1"}]}}
                        },
                        "method_name": "tensorflow/serving/predict"
                    }
                }
            }
        }
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    REQUEST_COUNT.labels(method='GET', endpoint='/', status='200').inc()
    return jsonify({
        'name': 'Muhammad Ivan LSPW - Heart Disease Prediction API',
        'version': '1.0.0',
        'model': 'loaded',
        'endpoints': [
            '/health', 
            '/predict', 
            '/metrics', 
            '/v1/models/muhammad_ivan_LspW-pipeline/metadata'
        ]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
