# Customer Churn Analysis & Prediction
## 📌 Project Overview
This project helps telecom businesses predict customer churn — identifying customers likely to discontinue their services. Early churn prediction enables businesses to implement proactive retention strategies, improve satisfaction, and reduce revenue loss.

This solution leverages machine learning models including Random Forest, XGBoost, and Logistic Regression, and is deployed using a Streamlit web application. A Power BI dashboard complements the solution with rich visual insights.

## 📁 Dataset Overview
The dataset contains 7043 rows and 21 columns of customer data, covering:

**Demographics** (e.g., gender, SeniorCitizen)

Services (e.g., InternetService, StreamingTV)

Account info (e.g., Contract, PaymentMethod)

Usage patterns and churn label

Preprocessing Steps:

No null values or duplicates found

TotalCharges: 17 missing values imputed with median

Categorical variables were encoded using Label Encoding

Univariate and bivariate analysis conducted to identify churn correlations

## 🧪 Data Balancing
The original dataset was imbalanced. To address this, SMOTE+ENN (a hybrid oversampling technique) was applied to balance the class distribution, followed by train-test split.

## 🤖 Model Building and Evaluation
Three models were trained and evaluated:

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	93%	93%	92%	92%
Random Forest	96%	96%	94%	95%
XGBoost Classifier	96%	97%	95%	96%

✅ XGBoost was selected as the final model due to superior performance.

## 📊 Power BI Dashboard
The Power BI dashboard includes:

Churn Overview: Visualizes churn rate and patterns

Customer Risk Analysis: Identifies high-risk customer segments

Insights & Strategies: Provides data-driven recommendations for reducing churn

## 🌐 Streamlit Web Application
Real-time churn prediction via:

Online Mode: Enter individual customer data

Batch Mode: Upload CSV file with multiple customer records

## 🖥️ Try the live app: Streamlit Deployment

https://telecom-customer-churn-analysis-prediction.streamlit.app/

📂 Customer Churn Analysis GitHub Repo

## 💡 Key Benefits
✅ Improved Retention: Identify high-risk customers proactively

📈 Better Decision-Making: Understand churn drivers and patterns

💰 Cost Efficiency: Reduce acquisition costs through improved retention

🏆 Competitive Edge: Deliver personalized retention strategies using data

## ⚙️ How to Run the App Locally
# Step 1: Clone the repository
git clone https://github.com/RAVITEJA-BHUKYA/Custemer_Churn_Analysis.git
cd Custemer_Churn_Analysis

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch the Streamlit app
streamlit run app.py

# Step 4: Access the app at
http://localhost:8501
## 📂 Project Structure
telecom-churn-analysis/
│
├── data/
│   ├── Telco-Customer-Churn.csv         ← Raw dataset
│   └── tel_churn_clean.csv             ← Cleaned and preprocessed dataset

├──  churn_analysis_model_training.ipynb  ← Jupyter notebook with analysis and model training
│
├─ app.py                         ← Core Streamlit application
│
├── xgboost_model.joblib  ← Trained model
├── Customer Churn Dashboard.pbix  ← PowerBI file
│
├── retrain.py                         ← Model retraining script
├── requirements.txt                   ← Python dependencies
├── README.md                          ← Project overview and setup instructions
└── .gitignore                         ← Git ignore file
## 📦 Requirements
streamlit
pandas
numpy
scikit-learn==1.2.2
imbalanced-learn
xgboost
pycaret
matplotlib
seaborn
plotly
joblib
## ⚠️ Challenges & Limitations
🔍 Model Bias: Despite SMOTE+ENN, data imbalance may affect predictions

📉 Interpretability: Gradient boosting models are less interpretable

🔁 Evolving Data: Regular model updates required

🧹 Data Quality: Prediction accuracy depends on the input data's completeness
