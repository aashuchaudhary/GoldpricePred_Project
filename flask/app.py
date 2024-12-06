from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler
model_path = "gold_price_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file:
    regressor = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    # Historical data for initial graph
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    prices = [1500, 1600, 1550, 1650, 1700, 1680]  # Replace with actual data if available
    return render_template('index.html', months=months, prices=prices, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the form
        input_data = [float(x) for x in request.form.values()]
        input_data = np.array(input_data).reshape(1, -1)

        # Scale the input using the pre-loaded scaler
        scaled_data = scaler.transform(input_data)

        # Predict using the pre-loaded model
        prediction = regressor.predict(scaled_data)[0]

        # Historical data (add predicted value for demonstration)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Predicted']
        prices = [1500, 1600, 1550, 1650, 1700, 1680, prediction]

        return render_template(
            'index.html',
            months=months,
            prices=prices,
            prediction=f'Predicted Gold Price: ${prediction:.2f}'
        )
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
