import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Sidebar
st.sidebar.title("Customer Churn Predictor")
st.sidebar.markdown("Built by **Engineer Chard Omolo**")
st.sidebar.markdown("[GitHub](https://github.com/BrilliantChard) | [LinkedIn](https://www.linkedin.com/in/chardodhiambo)")

# --------------------------
# Load and preprocess data
# --------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('Churn-Dataset.csv')
    data.drop('customerID', axis=1, inplace=True)
    data = data[data['TotalCharges'] != ' ']
    data['TotalCharges'] = data['TotalCharges'].astype(float)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    data[binary_cols] = data[binary_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

    data = pd.get_dummies(data, drop_first=True)
    return data

df = load_data()

# Feature-label separation
X = df.drop('Churn', axis=1)
y = df['Churn']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------------
# Train Logistic Regression Model
# --------------------------
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model()

# --------------------------
# Main Area
# --------------------------
st.title("📊 Predict Customer Churn")
st.markdown("Use this interactive app to predict whether a customer is likely to churn based on their service details.")

# --------------------------
# Model Evaluation Section
# --------------------------
st.subheader("📈 Model Evaluation on Test Set")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ **Accuracy:** {accuracy:.4f}")
st.text("📋 Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🔷 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ROC Curve
st.subheader("🔵 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Feature Importance
st.subheader("🧠 Feature Importance (Logistic Coefficients)")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importance.head(15), x="Coefficient", y="Feature", palette="coolwarm", ax=ax_feat)
ax_feat.set_title("Top 15 Important Features")
st.pyplot(fig_feat)

# --------------------------
# Batch Prediction from File
# --------------------------
st.subheader("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with customer data (same format as training)", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(batch_df.head())

    try:
        full_batch_df = pd.concat([df.drop('Churn', axis=1), batch_df], ignore_index=True)
        full_batch_df = pd.get_dummies(full_batch_df, drop_first=True)

        for col in X.columns:
            if col not in full_batch_df.columns:
                full_batch_df[col] = 0
        full_batch_df = full_batch_df[X.columns]

        batch_scaled = scaler.transform(full_batch_df.tail(len(batch_df)))
        predictions = model.predict_proba(batch_scaled)[:, 1]

        results_df = batch_df.copy()
        results_df['Churn Probability'] = predictions
        results_df['Churn Prediction'] = ['Likely to Churn' if p > 0.5 else 'Not Likely to Churn' for p in predictions]

        st.success("✅ Batch predictions completed!")
        st.write(results_df)

        csv_output = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_output, file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during batch prediction: {e}")

# --------------------------
# Single Customer Prediction
# --------------------------
st.subheader("🔍 Enter Customer Information")
with st.form("predict_form"):
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
    partner = st.selectbox("Has Partner", ['Yes', 'No'])
    dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_sec = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    tech = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    movie = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=350.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == 'Yes' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone == 'Yes' else 0,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': backup,
            'DeviceProtection': device,
            'TechSupport': tech,
            'StreamingTV': tv,
            'StreamingMovies': movie,
            'Contract': contract,
            'PaperlessBilling': 1 if paperless == 'Yes' else 0,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }

        input_df = pd.DataFrame([input_dict])

        full_df = pd.concat([df.drop('Churn', axis=1), input_df], ignore_index=True)
        full_df = pd.get_dummies(full_df, drop_first=True)

        for col in X.columns:
            if col not in full_df.columns:
                full_df[col] = 0
        full_df = full_df[X.columns]

        input_scaled = scaler.transform(full_df.tail(1))
        prediction = model.predict_proba(input_scaled)[0][1]

        st.success(f"🔮 Churn Probability: **{prediction:.2f}**")
        st.write("🟠 Likely to churn" if prediction > 0.5 else "🟢 Not likely to churn")
