import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('delivery_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍔 Food Delivery Time Predictor")
st.write("Enter your delivery details for a precise estimate.")

dist = st.number_input("Distance (km)", min_value=0.1, step=0.1)
prep = st.number_input("Preparation Time (min)", min_value=5, step=1)
exp = st.number_input("Rider Experience (yrs)", min_value=0, step=1)
rating = st.slider("Rider Rating", 1.0, 5.0, 4.5)

if st.button("Predict Time"):
    features = np.array([[dist, rating, 1, 1, 1, prep, exp]]) 
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0] 
    if prediction < 0:
        prediction = prep + (dist * 3)
    hours = int(prediction // 60)
    minutes = int(prediction % 60)
    seconds = int((prediction * 60) % 60)
    
    if hours > 0:
        time_text = f"{hours} Hours {minutes} Minutes {seconds}Seconds"
    else:
        time_text = f"{minutes}Minutes {seconds}Seconds"

    st.success(f" Distance: {dist} km")
    st.info(f" Estimated Delivery Time: {time_text}")
