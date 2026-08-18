
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Set Streamlit page configuration
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

st.title("Heart Disease Prediction Application")
st.write("This application allows you to predict heart disease and evaluate models based on patient data.")

# --- Load Trained Models and Scaler ---
@st.cache_resource
def load_artifacts():
    model_dir = 'trained_models'
    
    # Load StandardScaler
    scaler_filename = os.path.join(model_dir, 'scaler.joblib')
    if not os.path.exists(scaler_filename):
        st.error(f"Error: Scaler file not found at {scaler_filename}. Please ensure models are saved.")
        st.stop()
    scaler = joblib.load(scaler_filename)

    # Load models
    loaded_models = {}
    # List of models we trained, ensure these filenames match what was saved
    model_names_map = {
        'Logistic Regression': 'logistic_regression_model.joblib',
        'K-Nearest Neighbors': 'k-nearest_neighbors_model.joblib',
        'Support Vector Machine': 'support_vector_machine_model.joblib',
        'Random Forest': 'random_forest_model.joblib'
    }

    for display_name, filename in model_names_map.items():
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            st.error(f"Error: Model file {filename} not found at {model_path}. Please ensure all models are saved.")
            st.stop()
        loaded_models[display_name] = joblib.load(model_path)
    
    return scaler, loaded_models

scaler, tunned_models = load_artifacts()

# --- Function to preprocess input data (for both user input and uploaded data) ---
def preprocess_data(df_input):
    df_processed = df_input.copy()

    # Recreate preprocessing steps: imputation, one-hot encoding
    # Impute 0 values in 'RestingBP' and 'Cholesterol' with NaN, then with the median
    df_processed['RestingBP'] = df_processed['RestingBP'].replace(0, pd.NA)
    df_processed['Cholesterol'] = df_processed['Cholesterol'].replace(0, pd.NA)

    # Using the medians calculated from the *original* training data (hardcoded for simplicity here)
    # In a real app, these would also be saved/loaded.
    median_resting_bp_ref = 130.0 # From notebook context
    median_cholesterol_ref = 237.0 # From notebook context

    df_processed['RestingBP'] = df_processed['RestingBP'].fillna(median_resting_bp_ref)
    df_processed['Cholesterol'] = df_processed['Cholesterol'].fillna(median_cholesterol_ref)

    categorical_cols_for_encoding = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols_for_encoding, drop_first=False)

    # Ensure consistent column order and presence
    # This requires knowing the full set of columns from the original training data.
    # For this example, we'll recreate the expected columns based on the original data's structure.
    # In a production setting, you'd save a list of these columns.
    original_training_columns_example = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
       'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA',
       'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH',
       'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N',
       'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat',
       'ST_Slope_Up'] # This list should match X.columns from the notebook

    # Add missing columns with 0 and reorder
    for col in original_training_columns_example:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[original_training_columns_example]

    # Scale numerical features
    numerical_cols_for_scaling = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    df_processed[numerical_cols_for_scaling] = scaler.transform(df_processed[numerical_cols_for_scaling])

    return df_processed

# --- Sidebar for Navigation/Sections ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose your mode:", ("Predict Heart Disease", "Evaluate Models"))

# --- Predict Heart Disease Mode ---
if app_mode == "Predict Heart Disease":
    st.header("Predict Heart Disease for New Patient")
    st.write("Adjust the parameters below to get a prediction for heart disease.")

    # User Input Widgets
    with st.sidebar.expander("Patient Attributes Input", expanded=True):
        age = st.slider("Age", 18, 100, 50)
        sex = st.selectbox("Sex", ['M', 'F'])
        chest_pain_type = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
        resting_bp = st.slider("Resting Blood Pressure (mm/Hg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol (mm/dl)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        resting_ecg = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])
        max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ['N', 'Y'])
        oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.2, 1.0)
        st_slope = st.selectbox("ST_Slope of the peak exercise ST segment", ['Up', 'Flat', 'Down'])

        input_data = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain_type,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }
        input_df_raw = pd.DataFrame(input_data, index=[0])
    
    st.subheader("User Input Features:")
    st.write(input_df_raw)

    processed_input = preprocess_data(input_df_raw)

    # Model Selection for Interactive Prediction
    model_choice_interactive = st.selectbox(
        "Select Model for Interactive Prediction", 
        list(tunned_models.keys()), 
        index=list(tunned_models.keys()).index('K-Nearest Neighbors') # Default to best model
    )
    selected_model_interactive = tunned_models[model_choice_interactive]

    # Prediction
    prediction = selected_model_interactive.predict(processed_input)
    prediction_proba = selected_model_interactive.predict_proba(processed_input)

    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.success("The model predicts: No Heart Disease")
    else:
        st.error("The model predicts: Heart Disease")

    st.subheader("Prediction Probability:")
    st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")

# --- Evaluate Models Mode ---
elif app_mode == "Evaluate Models":
    st.header("Evaluate Models with Test Data")
    st.write("Upload a CSV file containing test data to evaluate the performance of the trained models.")

    uploaded_file = st.file_uploader("Upload heart.csv for evaluation", type=["csv"])

    if uploaded_file is not None:
        test_df_raw = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview:")
        st.write(test_df_raw.head())

        # Assuming the uploaded file has a 'HeartDisease' column for evaluation
        if 'HeartDisease' not in test_df_raw.columns:
            st.error("Uploaded CSV must contain a 'HeartDisease' column for evaluation.")
            st.stop()

        X_test_uploaded_raw = test_df_raw.drop('HeartDisease', axis=1)
        y_test_uploaded = test_df_raw['HeartDisease']

        # Preprocess the uploaded test data
        X_test_uploaded_processed = preprocess_data(X_test_uploaded_raw)

        st.subheader("Model Evaluation Results on Uploaded Data:")
        results_eval = {}

        for name, model in tunned_models.items():
            y_pred = model.predict(X_test_uploaded_processed)
            
            accuracy = accuracy_score(y_test_uploaded, y_pred)
            precision = precision_score(y_test_uploaded, y_pred, zero_division=0)
            recall = recall_score(y_test_uploaded, y_pred, zero_division=0)
            f1 = f1_score(y_test_uploaded, y_pred, zero_division=0)

            roc_auc = 'N/A'
            if hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(X_test_uploaded_processed)[:, 1]
                    roc_auc = roc_auc_score(y_test_uploaded, y_pred_proba)
                except ValueError as e:
                    roc_auc = f"Error: {e}"

            results_eval[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC AUC': roc_auc
            }
        
        results_df_eval = pd.DataFrame(results_eval).T.sort_values(by='ROC AUC', ascending=False)
        st.dataframe(results_df_eval)

        st.markdown("### Detailed Classification Report")
        model_choice_report = st.selectbox("Select Model for Detailed Report", list(tunned_models.keys()))
        selected_model_report = tunned_models[model_choice_report]

        y_pred_report = selected_model_report.predict(X_test_uploaded_processed)
        report = classification_report(y_test_uploaded, y_pred_report, output_dict=True)
        st.json(report) # Display as JSON for better readability in Streamlit

st.markdown("---")
st.write("Disclaimer: This is a predictive model for educational purposes and should not be used for medical diagnosis.")
