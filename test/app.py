import logging
import pandas as pd
from flask import Flask, request, jsonify
import utils_python as utils

app = Flask(__name__)

# Basic logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Prediction function
def get_predictions(data, model_name):
    return utils.predict_value(data, model_name)

# Function to get similar samples
def get_similar_samples(n, data, cut, color, clarity, weight):
    filtered_data = data[(data['cut'] == cut) & (data['color'] == color) & (data['clarity'] == clarity)]
    filtered_data['weight_diff'] = abs(filtered_data['carat'] - weight)
    sorted_data = filtered_data.sort_values('weight_diff')
    return sorted_data.head(n)

# API endpoint to get predictions
@app.route('/predict', methods=['GET'])
def predict():
    try:
        model_name = 'xgboost_diamonds'
        # Extracting query parameters
        color = request.args.get('color')
        cut = request.args.get('cut')
        carat = request.args.get('carat', type=float)
        depth = request.args.get('depth', type=float)
        table = request.args.get('table', type=float)
        clarity = request.args.get('clarity')
        x = request.args.get('x', type=float)
        y = request.args.get('y', type=float)
        z = request.args.get('z', type=float)

        # Check if all parameters are provided
        if not all([color, cut, carat, depth, table, clarity, x, y, z]):
            return jsonify({"error": "Missing parameters"}), 400

        # Create DataFrame from parameters
        data = pd.DataFrame({
            'carat': [carat],'cut': [cut],'color': [color],'clarity': [clarity],'depth': [depth],'table': [table],'x': [x],'y': [y],'z': [z]
        })

        # Get predictions
        predictions = get_predictions(data, model_name)
        return jsonify(predictions.tolist())

    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

# API endpoint to get similar samples
@app.route('/similar_samples', methods=['GET'])
def similar_samples():
    try:

        # Extract data from request
        cut = request.args.get('cut')
        color = request.args.get('color')
        clarity = request.args.get('clarity')
        weight = request.args.get('weight', type=float)
        n = request.args.get('n', type=int)

        # Read dataset
        diamonds_data = pd.read_csv('./data/diamonds_1.csv')

        # Get similar samples
        samples = get_similar_samples(n, diamonds_data, cut, color, clarity, weight)
        return jsonify(samples.to_dict(orient='records'))

    except Exception as e:
        logger.error(f"Error in /similar_samples: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)