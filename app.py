import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

with open(r"C:\Users\HP\Projects\Python_for_ML\ML_projects\Supervised\Classification\Crop_recommendation\logreg_crop_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(r"C:\Users\HP\Projects\Python_for_ML\ML_projects\Supervised\Classification\Crop_recommendation\label_encoder.pkl", "rb") as f:
    le: LabelEncoder = pickle.load(f)


st.title("Crop Recommender")
st.write("Enter The features to specify the crop.")

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

    st.balloons()
