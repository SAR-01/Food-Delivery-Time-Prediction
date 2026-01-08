import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('delivery_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍔 Food Delivery Time Predictor")
st.write("Enter your delivery details to get an accurate time estimate.")

# Inputs
dist = st.number_input("Distance (km)", min_value=0.5)
prep = st.number_input("Preparation Time (min)", min_value=5)
exp = st.number_input("Rider Experience (yrs)", min_value=0)

# --- Session State Logic ---
if 'prediction_text' not in st.session_state:
    st.session_state['prediction_text'] = None

if st.button("Predict Time"):
    features = np.array([[dist, 1, 1, 1, 1, prep, exp]])
    scaled_features = scaler.transform(features)
    
    # 1. Base Prediction logic (Model or Maths)
    if dist > 50:
        prediction = (dist * 2) + prep
    else:
        prediction = model.predict(scaled_features)[0]

    # --- NEW CHANGE: EXPERIENCE IMPACT ---
    # Logic: Har 1 saal ke experience par hum 0.5 minute (30 seconds) kam kar denge.
    # Agar rider experienced hai, wo tez aayega.
    time_saved = exp * 0.5 
    prediction = prediction - time_saved

    # Safety Check: Delivery time kabhi bhi Preparation time se kam nahi ho sakta
    # Aisa na ho ke experience itna ziada ho ke time negative ho jaye
    if prediction < prep:
        prediction = prep + 2 # Kam se kam prep time + 2 min travel
    # -------------------------------------

    # Time Conversion
    hours = int(prediction // 60)
    minutes = int(prediction % 60)
    seconds = int((prediction * 60) % 60)
    
    if hours > 0:
        result = f"{hours} Hours {minutes} Minutes {seconds} Seconds"
    else:
        result = f"{minutes} Minutes {seconds} Seconds"
    
    st.session_state['prediction_text'] = result

# --- Output & Rating ---
if st.session_state['prediction_text']:
    st.success(f"Estimated Delivery Time: {st.session_state['prediction_text']}")
    
    st.markdown("---")
    st.write("### 📝 Feedback")
    
    rating = st.slider("Rate this prediction accuracy (1 = Poor, 5 = Excellent):", 1, 5, 3)
    
    if st.button("Submit Rating"):
        st.write(f"Thanks! You rated us {rating}/5 ⭐")
