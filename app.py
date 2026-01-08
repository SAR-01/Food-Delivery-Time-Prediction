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
    
    # --- YAHAN FIX KIYA HAI ---
    # Agar distance 50km se zyada hai, to Model fail ho jayega.
    # Isliye hum manual formula lagayenge (Bike avg speed 30km/h = 2 min per km)
    
    if dist > 50:
        # 1 km = 2 minutes (approx for bike highway speed) + preparation time
        prediction = (dist * 2) + prep
    else:
        # Normal range ke liye Model use karein
        prediction = model.predict(scaled_features)[0]
    # --------------------------

    # Time Calculation Logic
    hours = int(prediction // 60)
    minutes = int(prediction % 60)
    seconds = int((prediction * 60) % 60)
    
    if hours > 0:
        time_text = f"{hours} Hours {minutes} Minutes {seconds} Seconds"
    else:
        time_text = f"{minutes} Minutes {seconds} Seconds"
        
    st.success(f"Estimated Delivery Time: {time_text}")
