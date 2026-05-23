import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

try:
    penguin_model = joblib.load('opt_SVM_model')
except Exception:
    penguin_model = None

try:
    iris_model = joblib.load('iris_rf_model.joblib')
except Exception:
    iris_model = None

try:
    diabetes_model = joblib.load('diabetes_gb_model.joblib')
except Exception:
    diabetes_model = None

st.set_page_config(
    page_title="Machine Learning Prediction App",
    layout="centered",
    page_icon="🤖"
)

with st.sidebar:
    selected = option_menu(
        "Prediction Menu",
        [
            'Penguin species prediction',
            'Iris species prediction',
            'Diabetes prediction'
        ],
        menu_icon='cpu-fill',
        icons=['emoji-grin', 'flower1', 'activity'],
        default_index=0
    )

# ----------------- Penguin Species Prediction -----------------
if selected == 'Penguin species prediction':
    st.title("Penguin Species Prediction using SVM")
    st.write("This page uses a pre-trained optimized Support Vector Machine (SVM) pipeline to classify penguin species based on their physical features.")

    col1, col2 = st.columns(2)
    with col1:
        island = st.selectbox("Island selection", ["Biscoe", "Dream", "Torgersen"])
        bill_length_mm = st.number_input("Enter bill length (mm)", format="%.1f", value=43.0, min_value=30.0, max_value=60.0, step=0.1)
        bill_depth_mm = st.number_input("Enter bill depth (mm)", format="%.1f", value=17.0, min_value=12.0, max_value=22.0, step=0.1)
        
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        flipper_length_mm = st.number_input("Enter flipper length (mm)", format="%d", value=200, min_value=150, max_value=300, step=1)
        body_mass = st.number_input("Enter body mass (g)", format="%d", value=4200, min_value=3000, max_value=6500, step=1)

    prediction_btn = st.button("Predict Penguin Species")

    if prediction_btn:
        if penguin_model is not None:
            keys = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
            user_input = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass, sex]
            input_dict = dict(zip(keys, user_input))
            
            input_df = pd.DataFrame(input_dict, index=[0])
            prediction = penguin_model.predict(input_df).tolist()
            
            st.success(f"Predicted Penguin Species: **{prediction[0]}**")
        else:
            st.error("Error: Penguin model file (`opt_SVM_model`) not found.")

# ------------------- Iris Species Prediction -------------------
elif selected == 'Iris species prediction':
    st.title("Iris Species Prediction using Random Forest")
    st.write("This page uses a freshly trained Random Forest classifier to identify the species of an Iris flower based on its sepal and petal measurements.")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", format="%.1f", value=5.1, min_value=4.0, max_value=8.0, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", format="%.1f", value=3.5, min_value=2.0, max_value=5.0, step=0.1)
        
    with col2:
        petal_length = st.number_input("Petal Length (cm)", format="%.1f", value=1.4, min_value=1.0, max_value=7.0, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", format="%.1f", value=0.2, min_value=0.1, max_value=3.0, step=0.1)

    prediction_btn = st.button("Predict Iris Species")

    if prediction_btn:
        if iris_model is not None:
            features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
            user_input = [sepal_length, sepal_width, petal_length, petal_width]
            input_df = pd.DataFrame([user_input], columns=features)
            
            prediction = iris_model.predict(input_df).tolist()
            
            st.success(f"Predicted Iris Species: **{prediction[0]}**")
        else:
            st.error("Error: Iris model file (`iris_rf_model.joblib`) not found. Please run the training script first.")

# ------------------ Diabetes Prediction Page -------------------
elif selected == 'Diabetes prediction':
    st.title("Diabetes Prediction using Gradient Boosting")
    st.write("This page uses a trained Gradient Boosting Classifier to predict the likelihood of diabetes based on medical records.")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", value=0, min_value=0, step=1)
        glucose = st.number_input("Glucose level (mg/dL)", value=120, min_value=0, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", value=70, min_value=0, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", value=20, min_value=0, step=1)
    with col2:
        insulin = st.number_input("Insulin level (mu U/ml)", value=80, min_value=0, step=1)
        bmi = st.number_input("BMI (Weight in kg/(height in m)^2)", format="%.1f", value=25.0, min_value=0.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", format="%.3f", value=0.500, min_value=0.0, step=0.001)
        age = st.number_input("Age (years)", value=25, min_value=0, step=1)
        
    prediction_btn = st.button("Predict Diabetes")
    
    if prediction_btn:
        if diabetes_model is not None:

            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            input_df = pd.DataFrame([user_input], columns=features)
            

            prediction = diabetes_model.predict(input_df).tolist()
            
            if prediction[0] == 1:
                st.warning("Prediction: **The person is diabetic.**")
            else:
                st.success("Prediction: **The person is not diabetic.**")
        else:
            st.error("Error: Diabetes model file (`diabetes_gb_model.joblib`) not found. Please run the training script first.")
