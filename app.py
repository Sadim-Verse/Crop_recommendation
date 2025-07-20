import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# with open(r"logreg_crop_classifier", "rb") as model_file:
#     model = joblib.load(model_file)
model = joblib.load(r"logreg_crop_classifier")

# with open(r"label_encoder", "rb") as f:
#     le: LabelEncoder = joblib.load(f)
le = joblib.load(r"label_encoder")


st.title("A Basic Crop Recommender")
st.write("Input the features to specify the crop to plant.")

N = st.slider("Nitrogen Ratio: ", min_value=1, max_value=100)
P = st.slider("Phosphorus Ratio: ", min_value=1, max_value=100)
K = st.slider("Potassium Ratio: ", min_value=1, max_value=100)

temperature = st.number_input("Temperature: ")
humidity = st.number_input("Humidity: ")
ph = st.number_input("PH Value: ")
rainfall = st.number_input("Rainfall: ")



if st.button("Predict"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_num = model.predict(features)
    predicted_name = le.inverse_transform(predicted_num)
    st.write(f"Predicted Crop to Plant: {predicted_name[0]}")

    st.ballons()
