import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import load

# Loading trained model
def load_model():
    model_path = '/Users/anusreemohanan/Downloads/optimized_heart_disease_model.joblib'
    model = load(model_path)
    return model

model = load_model()

# Streamlit webpage
def main():
    st.title('Heart Disease Prediction App')

    # Input bars
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', options=cp_options)
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120)
    chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=400, value=150)
    fbs_options = ['Less than 120 mg/dl', 'More than 120 mg/dl']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=fbs_options)
    restecg_options = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', options=restecg_options)
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=140)
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', options=exang_options)
    oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=slope_options)
    ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, max_value=4, value=0)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', options=thal_options)

    # Mapping back to numerical codes for model prediction
    sex = 1 if sex == 'Male' else 0
    cp = cp_options.index(cp)
    fbs = 1 if fbs == 'More than 120 mg/dl' else 0
    restecg = restecg_options.index(restecg)
    exang = 1 if exang == 'Yes' else 0
    slope = slope_options.index(slope)
    thal = thal_options.index(thal)

    # Predict button
    if st.button('Predict'):
        # Create an array with the input values
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Standardize the input data if it was used during model training
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        
        # Prediction
        prediction = model.predict(input_data)
        result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        st.success(f'The model predicts: {result}')

if __name__ == '__main__':
    main()
