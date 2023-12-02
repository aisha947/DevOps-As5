# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load pre-trained models
max_t_model = joblib.load('max_t_model.joblib')
min_t_model = joblib.load('min_t_model.joblib')

# Define the transformer for one-hot encoding 'Weather_Condition'
preprocessor = ColumnTransformer(
    transformers=[
        ('weather', OneHotEncoder(), ['Weather_Condition'])
    ],
    remainder='passthrough'
)

# Combine the preprocessor with the model in a pipeline
max_t_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', max_t_model)
])

min_t_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', min_t_model)
])

@app.route('/predict/max_t', methods=['POST'])
def predict_max_t():
    data = request.get_json()
    features = pd.DataFrame(data['features'])
    
    # Use the pre-trained max_t_model for prediction
    prediction = max_t_model.predict(features)
    
    return jsonify({"prediction": prediction.tolist()})

@app.route('/predict/min_t', methods=['POST'])
def predict_min_t():
    data = request.get_json()
    features = pd.DataFrame(data['features'])
    
    # Use the pre-trained min_t_model for prediction
    prediction = min_t_model.predict(features)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
