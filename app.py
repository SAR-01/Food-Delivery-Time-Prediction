import streamlit as st
import pickle
import numpy as np

# Model aur Scaler load karna
model = pickle.load(open('delivery_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍔 Food Delivery Time Predictor")
st.write("Enter your delivery details to get an accurate time estimate.")

# Inputs
dist = st.number_input("Distance (km)", min_value=0.5)
prep = st.number_input("Preparation Time (min)", min_value=5)
exp = st.number_input("Rider Experience (yrs)", min_value=0)

# --- Session State Logic (Taake result ghaib na ho) ---
if 'prediction_text' not in st.session_state:
    st.session_state['prediction_text'] = None

if st.button("Predict Time"):
    features = np.array([[dist, 1, 1, 1, 1, prep, exp]])
    scaled_features = scaler.transform(features)
    
    # Wohi Logic: 50km se zyada ho to Maths, kam ho to Model
    if dist > 50:
        prediction = (dist * 2) + prep
    else:
        prediction = model.predict(scaled_features)[0]

    # Time Calculation
    hours = int(prediction // 60)
    minutes = int(prediction % 60)
    seconds = int((prediction * 60) % 60)
    
    if hours > 0:
        result = f"{hours} Hours {minutes} Minutes {seconds} Seconds"
    else:
        result = f"{minutes} Minutes {seconds} Seconds"
    
    # Result ko session state me save kar lia
    st.session_state['prediction_text'] = result

# --- Result Display aur Rating Section ---
if st.session_state['prediction_text']:
    st.success(f"Estimated Delivery Time: {st.session_state['prediction_text']}")
    
    st.markdown("---") # Line lagane ke liye
    st.write("### 📝 Feedback")
    
    # Rating Option
    rating = st.slider("Rate this prediction accuracy (1 = Poor, 5 = Excellent):", 1, 5, 3)
    
    if st.button("Submit Rating"):
        st.write(f"Thanks! You rated us {rating}/5 ⭐")
        # Yahan aap chahain to is rating ko database me save karwa sakte hain
