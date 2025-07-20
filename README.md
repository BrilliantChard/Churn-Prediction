# 💡 Customer Churn Prediction App

This project is a **Streamlit web application** that uses a **deep learning model (TensorFlow)** to predict whether a customer is likely to churn or stay based on various features from the Telco dataset.

![Streamlit App Screenshot](screenshot.png) <!-- You can update this with a real screenshot later -->

---

## 📦 Features

✅ Upload single or multiple customer records  
✅ Real-time churn probability prediction using a trained deep learning model  
✅ Interactive visualizations using Matplotlib and Seaborn  
✅ Sidebar with author info and external links  
✅ Download predictions as CSV  

---

## 📁 Dataset

The app uses the popular **Telco Customer Churn** dataset, which includes the following features:

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `PhoneService`, `MultipleLines`, `InternetService`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`
- `TechSupport`, `StreamingTV`, `StreamingMovies`
- `Contract`, `PaperlessBilling`, `PaymentMethod`
- `MonthlyCharges`, `TotalCharges`

**Target variable:** `Churn`

---

## 🧠 Model

- Framework: TensorFlow / Keras
- Model: Deep Neural Network (Sequential)
- Metrics: Accuracy, Binary Crossentropy Loss
- Preprocessing:
  - One-hot encoding for categorical variables
  - StandardScaler for numerical variables
- Validation using Train/Test split

---

## 🚀 How to Run the App

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/churn-prediction-app.git
   cd churn-prediction-app

