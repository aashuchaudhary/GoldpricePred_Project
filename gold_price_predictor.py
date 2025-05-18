import streamlit as st
import pickle
import numpy as np
import os

# Check if model and scaler exist
model_path = "gold_price_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error(
        "Model and scaler files are missing! Please train and save the model again."
    )
else:
    # Load the trained model and scaler
    with open(model_path, "rb") as model_file:
        regressor = pickle.load(model_file)

    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    st.success("Model loaded successfully!")

    # Streamlit UI
    st.title("Gold Price Predictor")
    st.write("Enter the required financial indicators to predict the gold price.")

    # User input fields
    feature_1 = st.number_input(
        "Feature 1 (e.g., Gold Price Previous Day)", value=1500.0
    )
    feature_2 = st.number_input("Feature 2 (e.g., Stock Market Index)", value=80.0)
    feature_3 = st.number_input("Feature 3 (e.g., Crude Oil Price)", value=20.0)
    feature_4 = st.number_input("Feature 4 (e.g., Inflation Rate)", value=1.5)

    # Prediction function
    def predict_gold_price(features):
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = regressor.predict(scaled_features)
        return prediction[0]

    # Button to predict gold price
    if st.button("Predict Gold Price"):
        user_input = [feature_1, feature_2, feature_3, feature_4]
        predicted_price = predict_gold_price(user_input)
        st.success(f"Predicted Gold Price: ${predicted_price:.2f}")
