import streamlit as st
import numpy as np
import pickle

# Load model and encoders
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("brand_encoder.pkl", "rb") as f:
    brand_encoder = pickle.load(f)

with open("fuel_encoder.pkl", "rb") as f:
    fuel_encoder = pickle.load(f)

# UI
st.title("🚗 Car Price Predictor (with Pickle Model)")
st.markdown("Upload your car details to predict its selling price.")

brand = st.selectbox("Select Brand", brand_encoder.classes_)
fuel = st.selectbox("Select Fuel Type", fuel_encoder.classes_)
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

if st.button("Predict Price"):
    # Transform inputs
    brand_encoded = brand_encoder.transform([brand])[0]
    fuel_encoded = fuel_encoder.transform([fuel])[0]

    input_data = np.array([[brand_encoded, year, km_driven, fuel_encoded]])

    # Predict
    prediction = car_price_model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹{int(prediction):,}")
