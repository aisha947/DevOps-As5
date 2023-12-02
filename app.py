from flask import Flask, jsonify, request
import joblib
import datetime

app = Flask(__name__)

# Load pre-trained models
model_max = joblib.load('model_max.joblib')
model_min = joblib.load('model_min.joblib')

def generate_dates(start_date, num_days):
    date_list = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
    return [date.strftime('%Y-%m-%d') for date in date_list]

@app.route('/predict/max_t', methods=['POST'])
def predict_max_temperature():
    data = request.get_json()
    num_days = data['num_days']

    # Example: Extract features for prediction based on 'num_days'
    X_input_max = [[data['Humidity'], data['Wind_Speed']] for _ in range(num_days)]

    # Make predictions using the pre-trained model
    predictions_max = model_max.predict(X_input_max)

    # Generate date list
    start_date = datetime.datetime.now().date()
    date_list = generate_dates(start_date, num_days)

    response = {
        "predictions": [{"date": date, "max_temperature": float(temp)} for date, temp in zip(date_list, predictions_max)]
    }

    return jsonify(response)

@app.route('/predict/min_t', methods=['POST'])
def predict_min_temperature():
    data = request.get_json()
    num_days = data['num_days']

    # Example: Extract features for prediction based on 'num_days'
    X_input_min = [[data['Humidity'], data['Wind_Speed']] for _ in range(num_days)]

    # Make predictions using the pre-trained model
    predictions_min = model_min.predict(X_input_min)

    # Generate date list
    start_date = datetime.datetime.now().date()
    date_list = generate_dates(start_date, num_days)

    response = {
        "predictions": [{"date": date, "min_temperature": float(temp)} for date, temp in zip(date_list, predictions_min)]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
