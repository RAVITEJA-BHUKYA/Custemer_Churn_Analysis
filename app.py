import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import base64

# Load the Logistic Regression model
model = joblib.load(r"D:\infotact\xgboost_model.joblib")

def predict(model, input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    return prediction[0], probability[0]

def preprocess_data(batch_df):
    relevant_columns = ['tenure', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract', 'MonthlyCharges', 'TotalCharges']
    batch_df = batch_df[relevant_columns]
    batch_df['OnlineSecurity'] = batch_df['OnlineSecurity'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    batch_df['OnlineBackup'] = batch_df['OnlineBackup'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    batch_df['TechSupport'] = batch_df['TechSupport'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    batch_df = batch_df.apply(pd.to_numeric, errors='coerce')
    batch_df.fillna(0, inplace=True)
    return batch_df

def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    
    prediction_mode = st.sidebar.selectbox("Prediction Mode", ['Online', 'Batch'])

    if prediction_mode == 'Online':
        st.title("Online Customer Churn Prediction")

        st.write("[0 = No, 1 = No Internet Service, 2 = Yes]")

        tenure = st.number_input('Tenure (Months):', min_value=1, max_value=72, value=1)
        OnlineSecurity = st.selectbox('Online Security', [2, 1, 0])
        OnlineBackup = st.selectbox('Online Backup', [2, 1, 0])
        TechSupport = st.selectbox('Tech Support', [2, 1, 0])
        Contract = st.selectbox('Contract (0 = Month-to-Month, 1 = 1 Year, 2 = 2 Year)', [0, 1, 2])
        MonthlyCharges = st.number_input('Monthly Charges', min_value=18, max_value=120, value=18)
        TotalCharges = st.number_input('Total Charges', min_value=18, max_value=9000, value=18)

        input_dict = {
            'tenure': [tenure],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'TechSupport': [TechSupport],
            'Contract': [Contract],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        }

        input_df = pd.DataFrame(input_dict)

        if st.button("Predict"):
            prediction, probability = predict(model, input_df)
            result = "Customer will churn" if prediction == 1 else "Customer will not churn"
            st.success(f"Prediction: {result}")
            st.info(f"Churn Probability: {probability:.2f}")

    elif prediction_mode == 'Batch':
        st.title("Batch Prediction from CSV File")
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            processed_df = preprocess_data(batch_df)
            predictions = model.predict(processed_df)
            probabilities = model.predict_proba(processed_df)[:, 1]

            batch_df['Predictions'] = predictions
            batch_df['Probability'] = probabilities

            st.write(batch_df)

            st.write("Churn Probability Stats:")
            st.write(f"Avg. Probability for Churn: {batch_df[batch_df['Predictions'] == 1]['Probability'].mean():.2f}")
            st.write(f"Avg. Probability for No Churn: {batch_df[batch_df['Predictions'] == 0]['Probability'].mean():.2f}")

            fig = px.histogram(batch_df, x='Predictions', color='Predictions', title='Churn Prediction Distribution')
            st.plotly_chart(fig)

            csv = batch_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv">Download CSV</a>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
