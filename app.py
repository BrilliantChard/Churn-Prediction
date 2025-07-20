import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

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
    data = pd.read_csv('Telco-Customer-Churn.csv')
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
# Train Model
# --------------------------
@st.cache_resource
def train_model():
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    return model, history

model, history = train_model()

# --------------------------
# Main Area
# --------------------------
st.title("📊 Predict Customer Churn")
st.markdown("Use this interactive app to predict whether a customer is likely to churn based on their service details.")

# Show training accuracy graph
st.subheader("Model Training Performance")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[0].set_title('Accuracy')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].set_title('Loss')
ax[1].legend()
st.pyplot(fig)

# --------------------------
# Upload CSV File for Batch Prediction
# --------------------------

st.subheader("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with customer data: same format as training data", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(batch_df.head())

    try:
        # Combine with training data to match structure
        full_batch_df = pd.concat([df.drop('Churn', axis=1), batch_df], ignore_index=True)
        full_batch_df = pd.get_dummies(full_batch_df, drop_first=True)

        for col in X.columns:
            if col not in full_batch_df.columns:
                full_batch_df[col] = 0
        full_batch_df = full_batch_df[X.columns]

        batch_scaled = scaler.transform(full_batch_df.tail(len(batch_df)))
        predictions = model.predict(batch_scaled).flatten()

        results_df = batch_df.copy()
        results_df['Churn Probability'] = predictions
        results_df['Churn Prediction'] = ['Likely to Churn' if p > 0.5 else 'Not Likely to Churn' for p in predictions]

        st.success("✅ Batch predictions completed!")
        st.write(results_df)

        # Download link
        csv_output = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_output, file_name="churn_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during batch prediction: {e}")

# --------------------------
# Input Form
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
        # Form to dataframe
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

        # Combine and encode
        full_df = pd.concat([df.drop('Churn', axis=1), input_df], ignore_index=True)
        full_df = pd.get_dummies(full_df, drop_first=True)

        for col in X.columns:
            if col not in full_df.columns:
                full_df[col] = 0
        full_df = full_df[X.columns]

        input_scaled = scaler.transform(full_df.tail(1))
        prediction = model.predict(input_scaled)[0][0]

        st.success(f"🔮 Churn Probability: **{prediction:.2f}**")
        st.write("🟠 Likely to churn" if prediction > 0.5 else "🟢 Not likely to churn")
