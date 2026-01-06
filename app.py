import streamlit as st
import pickle
import numpy as np

# Files load karein
model = pickle.load(open('delivery_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍔 Food Delivery Time Predictor")
st.write("Enter your delivery details to get an accurate time estimate.")

# User Inputs
dist = st.number_input("Distance (km)", min_value=0.5)
prep = st.number_input("Preparation Time (min)", min_value=5)
exp = st.number_input("Courier Experience (yrs)", min_value=0)

if st.button("Predict Time"):
    # Input ko format karein
    # Note: The original code used fixed values (1, 1, 1, 1) for encoded categorical features.
    # In a real Streamlit app, these would come from user selections (e.g., dropdowns).
    # For this demonstration, we maintain consistency with the notebook's prediction function.
    features = np.array([[dist, 1, 1, 1, 1, prep, exp]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    st.success(f"Estimated Time: {prediction[0]:.2f} Minutes")


