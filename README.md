
# 🧑‍💼 Employee Salary Prediction

This project predicts whether an employee’s income exceeds **50K per year** based on demographic and work-related attributes. The model is built using **Random Forest Classifier** and deployed with an interactive **Streamlit app**.

---

## 📌 Project Overview

The goal of this project is to build a machine learning pipeline that:

1. Cleans and preprocesses employee data.
2. Encodes categorical features into numeric form.
3. Trains and evaluates classification models.
4. Deploys an interactive web app where users can input details and get predictions.

---

## ⚙️ Features Used

The following features are used for prediction:

* **Age**
* **Educational Number** (mapped from education level)
* **Marital Status**
* **Relationship**
* **Gender**
* **Capital Gain**
* **Capital Loss**
* **Hours per Week**

The target variable is **Income** (`<=50K` or `>50K`).

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** (data preprocessing)
* **Scikit-learn** (RandomForestClassifier, model evaluation)
* **Streamlit** (web app deployment)
* **Joblib/Pickle** (model persistence)

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/employee-salary-prediction.git
   cd employee-salary-prediction
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Performance

* **Model Used:** Random Forest Classifier
* **Accuracy Achieved:** \~85% (on test set)
* Handles categorical variables through encoding and numeric mapping.

---

## 🌐 Demo

Users can input details such as age, education, marital status, and work hours to get an instant salary prediction via the web app.

---

## 📂 Repository Structure

```
employee-salary-prediction/
│── app.py                     # Streamlit app
│── employee_salary_prediction_updated.ipynb   # Notebook with data exploration & training
│── model.pkl                  # Trained Random Forest model
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
│── data/                      # Dataset (if included)
```

---

## 🔮 Future Work

* Improve feature engineering with additional socio-economic features.
* Experiment with **XGBoost, LightGBM** for better accuracy.
* Deploy on **Streamlit Cloud / AWS / Heroku** for public access.

---

## 👨‍💻 Author

Developed by **Uttam Singh Chaudhary**
📧 Feel free to connect on [LinkedIn]([https://linkedin.com/](https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b)

