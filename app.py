import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

# Unzip only if it's not already extracted
if not os.path.exists("Champion_model.pkl"):
    with zipfile.ZipFile("Champion_model.zip", "r") as zip_ref:
        zip_ref.extractall()  # Extracts in the current directory
        

# 🔹 Load the trained model
model = joblib.load("Champion_model.pkl")

# 🔹 Page configuration
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)

# 🔹 App Title
st.title("💼 Employee Salary Classification")
st.markdown("Use this app to predict whether an employee earns >50K or ≤50K based on various inputs.")

# ─────────────────────────────────────────────────────────────
# Sidebar: User Inputs
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📝 Enter Employee Details")

# 🌟 Input fields (should match the training data's feature names)
age = st.sidebar.slider("Age", 18, 65, 30)

education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])

occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])

hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# 🔹 Create input DataFrame (column names must match model's expectation)
input_df = pd.DataFrame({
    "age": [age],
    "education": [education],
    "occupation": [occupation],
    "hours-per-week": [hours_per_week],
    "experience": [experience]
})

# 🔹 Display input
st.subheader("🔍 Preview of Input Data")
st.dataframe(input_df)

# ─────────────────────────────────────────────────────────────
# Single Prediction
# ─────────────────────────────────────────────────────────────
if st.button("🔮 Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# ─────────────────────────────────────────────────────────────
# Batch Prediction Section
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with employee data", type="csv")

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(batch_data.head())

    # 🔹 Make predictions
    batch_preds = model.predict(batch_data)
    batch_data["PredictedClass"] = batch_preds

    # 🔹 Show results
    st.success("✅ Batch predictions complete!")
    st.write(batch_data.head())

    # 🔹 Download button
    csv = batch_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="predicted_classes.csv",
        mime="text/csv"
    )
