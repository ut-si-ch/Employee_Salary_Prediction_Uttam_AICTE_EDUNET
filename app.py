import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

# ──────────────────────────────────────────────
# Safe Model Loading
# ──────────────────────────────────────────────
model = None
try:
    if not os.path.exists("Champion_model.pkl"):
        with zipfile.ZipFile("Champion_model.zip", "r") as zip_ref:
            zip_ref.extractall()
    model = joblib.load("Champion_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Classification")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on their details.")

# ──────────────────────────────────────────────
# Mapping dictionaries for categorical → numeric
# ──────────────────────────────────────────────
education_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, 
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Doctorate/PhD": 15, "Prof-school": 16
}

marital_map = {
    "Never-married": 1,
    "Married-civ-spouse": 2,
    "Married-spouse-absent": 3,
    "Separated": 4,
    "Divorced": 5,
    "Widowed": 6
}

relationship_map = {
    "Husband": 1,
    "Wife": 2,
    "Not-in-family": 3,
    "Own-child": 4,
    "Unmarried": 5,
    "Other-relative": 6
}

gender_map = {
    "Male": 1,
    "Female": 0
}

# ──────────────────────────────────────────────
# Utility: Apply mapping safely
# ──────────────────────────────────────────────
def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map human-readable categorical columns to numeric codes safely."""
    mapping_cols = {
        "educational-num": education_map,
        "marital-status": marital_map,
        "relationship": relationship_map,
        "gender": gender_map
    }

    for col, mapping in mapping_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])  # fallback in case of unexpected values
    return df

# ──────────────────────────────────────────────
# Input Form (Single Prediction)
# ──────────────────────────────────────────────
with st.expander("📝 Single Prediction: Enter Employee Details", expanded=True):
    with st.form("prediction_form"):
        age = st.slider("Age", 18, 65, 30)
        education = st.selectbox("Education Level", list(education_map.keys()))
        marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
        relationship = st.selectbox("Relationship", list(relationship_map.keys()))
        gender = st.radio("Gender", list(gender_map.keys()))
        capital_gain = st.number_input("Capital Gain", min_value=0, step=100, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, step=100, value=0)
        hours_per_week = st.slider("Hours per Week", 1, 80, 40)

        submit_btn = st.form_submit_button("🔮 Predict Salary Class")

    if submit_btn:
        if model:
            try:
                # Convert to numeric according to mapping
                input_df = pd.DataFrame({
                    "age": [age],
                    "educational-num": [education_map[education]],
                    "marital-status": [marital_map[marital_status]],
                    "relationship": [relationship_map[relationship]],
                    "gender": [gender_map[gender]],
                    "capital-gain": [capital_gain],
                    "capital-loss": [capital_loss],
                    "hours-per-week": [hours_per_week]
                })

                st.subheader("🔍 Encoded Input Data")
                st.dataframe(input_df, use_container_width=True)

                prediction = model.predict(input_df)
                if prediction[0] == 1 or prediction[0] == ">50K":
                    st.success("💰 Prediction: Employee earns **>50K** ✅")
                else:
                    st.warning("📉 Prediction: Employee earns **≤50K** ⚠️")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        else:
            st.error("⚠️ Model not available. Please check deployment setup.")

# ──────────────────────────────────────────────
# Batch Prediction (CSV Upload)
# ──────────────────────────────────────────────
with st.expander("📂 Batch Prediction: Upload CSV File", expanded=False):
    uploaded_file = st.file_uploader("Upload CSV file with employee details", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded Data:")
            st.dataframe(df.head(), use_container_width=True)

            if model:
                # Encode categorical values
                df_encoded = encode_dataframe(df.copy())

                # Keep only expected features in correct order
                required_cols = [
                    "age", "educational-num", "marital-status", "relationship",
                    "gender", "capital-gain", "capital-loss", "hours-per-week"
                ]
                df_encoded = df_encoded[required_cols]

                st.subheader("🔍 Encoded Data for Prediction")
                st.dataframe(df_encoded.head(), use_container_width=True)

                predictions = model.predict(df_encoded)

                df["Predicted Salary Class"] = [
                    ">50K" if (p == 1 or p == ">50K") else "≤50K" for p in predictions
                ]

                st.success("✅ Predictions generated successfully!")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions as CSV",
                    data=csv,
                    file_name="salary_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("⚠️ Model not available. Please check deployment setup.")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
