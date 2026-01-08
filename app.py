import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('delivery_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍔 Food Delivery Time Predictor")
st.write("Enter your delivery details to get an accurate time estimate.")

dist = st.number_input("Distance (km)", min_value=0.5)
prep = st.number_input("Preparation Time (min)", min_value=5)
exp = st.number_input("Rider Experience (yrs)", min_value=0)

if st.button("Predict Time"):
    features = np.array([[dist, 1, 1, 1, 1, prep, exp]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]  # Total minutes extract karein

    # Calculation logic
    hours = int(prediction // 60)
    minutes = int(prediction % 60)
    seconds = int((prediction * 60) % 60)

    # Display Result
    if hours > 0:
        time_text = f"{hours} hrs {minutes} min {seconds} sec"
    else:
        time_text = f"{minutes} min {seconds} sec"
        
    st.success(f"⏳ Estimated Delivery Time: {time_text}")
