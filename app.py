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

# ──────────────────────────────────────────────
# Encoding Maps (must match training preprocessing)
# ──────────────────────────────────────────────
education_map = {
    "HS-grad": 0,
    "Some-college": 1,
    "Assoc": 2,
    "Bachelors": 3,
    "Masters": 4,
    "PhD": 5
}

occupation_map = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
    "Armed-Forces": 13
}

def preprocess_input(df: pd.DataFrame):
    """Convert categorical features to numeric codes"""
    df = df.copy()
    try:
        df["education"] = df["education"].map(education_map)
        df["occupation"] = df["occupation"].map(occupation_map)
    except Exception as e:
        st.error(f"⚠️ Error in preprocessing: {e}")
    return df

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("💼 Employee Salary Classification")
st.markdown(
    "This app predicts whether an employee earns **>50K** or **≤50K** "
    "based on their details. You can either enter a single record manually or upload a CSV file."
)
st.info("💡 Tip: Use the single prediction form for testing, or upload a CSV for bulk predictions.")

# ──────────────────────────────────────────────
# Input Form (Single Prediction)
# ──────────────────────────────────────────────
with st.expander("📝 Single Prediction: Enter Employee Details", expanded=True):
    with st.form("prediction_form"):
        age = st.slider("Age", 18, 65, 30)

        education = st.selectbox("Education Level", list(education_map.keys()))

        occupation = st.selectbox("Occupation", list(occupation_map.keys()))

        hours_per_week = st.slider("Hours per Week", 1, 80, 40)
        experience = st.slider("Years of Experience", 0, 40, 5)

        submit_btn = st.form_submit_button("🔮 Predict Salary Class")

    if submit_btn:
        if model:
            try:
                input_df = pd.DataFrame({
                    "age": [age],
                    "education": [education],
                    "occupation": [occupation],
                    "hours-per-week": [hours_per_week],
                    "experience": [experience]
                })
                st.subheader("🔍 Preview of Input Data (Before Encoding)")
                st.dataframe(input_df, use_container_width=True)

                # Preprocess before prediction
                encoded_df = preprocess_input(input_df)

                prediction = model.predict(encoded_df)
                if prediction[0] == ">50K":
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
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")

st.info("📌 Upload a CSV file with employee data to get predictions for multiple records at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)

        if batch_data.empty:
            st.warning("⚠️ The uploaded CSV file is empty. Please check your data.")
        else:
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(batch_data.head(), use_container_width=True)

            if model:
                # Preprocess before prediction
                encoded_batch = preprocess_input(batch_data)

                batch_preds = model.predict(encoded_batch)
                batch_data["PredictedClass"] = batch_preds

                st.success("✅ Batch predictions complete!")
                st.write(batch_data.head())

                # Download option
                csv = batch_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predicted_classes.csv",
                    mime="text/csv"
                )
            else:
                st.error("⚠️ Model not available for batch predictions.")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📂 Awaiting CSV upload. Please select a file above.")
