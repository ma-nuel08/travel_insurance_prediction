✈️ Travel Insurance Prediction — Machine Learning Classification Project

SQI College of ICT | Machine Learning Course | Phase 1–6


📌 Project Overview
This project builds a machine learning classification system to predict whether a customer will purchase travel insurance based on demographic and travel history data.
The dataset contains 1,987 records with features like age, income, employment type, and travel history. The target variable is binary — 1 (bought insurance) or 0 (did not buy).

🗂️ Project Phases
PhaseDescriptionPhase 1Data Loading & Exploratory Data Analysis (EDA)Phase 2Class Imbalance AnalysisPhase 3Data Preprocessing & Feature EncodingPhase 4Handling Imbalance with SMOTE & Class WeightsPhase 5Model Training & Evaluation (RF, GB, LR)Phase 6Model Deployment with Streamlit ✅

🤖 Models Used

Random Forest Classifier
Gradient Boosting Classifier
Logistic Regression

All models were trained with:

SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance
Class Weights for additional balance correction
80/20 Train-Test Split with stratification


📊 Phase 6 — Streamlit Dashboard Features
The deployed app (app.py) includes:

KPI Summary Row — Best model's Accuracy, AUC, Precision, Recall, F1
Model Comparison Tab — Table + bar charts across all 3 models
Confusion Matrices Tab — Side-by-side CM with TP/TN/FP/FN breakdown
ROC Curves Tab — All 3 models overlaid on one chart
Feature Importance Tab — Random Forest feature rankings
Live Predictor Tab — Real-time prediction from user inputs


📁 Project Structure
travel-insurance-prediction/
│
├── app.py                    # Streamlit deployment app (Phase 6)
├── random_forest.pkl         # Saved Random Forest model (joblib)
├── gradient_boosting.pkl     # Saved Gradient Boosting model (joblib)
├── logistic_regression.pkl   # Saved Logistic Regression model (joblib)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation

⚙️ Installation & Setup
1. Clone the repository
bashgit clone https://github.com/ma-nuel08/travel-insurance-prediction.git
cd travel-insurance-prediction
2. Create and activate a virtual environment
bashpython -m venv .vit
.vit\Scripts\activate        # Windows
source .vit/bin/activate     # Mac/Linux
3. Install dependencies
bashpip install -r requirements.txt
4. Run the Streamlit app
bashstreamlit run app.py
Then open your browser at http://localhost:8501

📦 Requirements
streamlit
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
joblib

📈 Dataset Features
FeatureDescriptionAgeAge of the customerEmployment TypeGovernment or Private SectorGraduateOrNotWhether the customer is a graduateAnnualIncomeAnnual income in local currencyFamilyMembersNumber of family membersChronicDiseasesWhether the customer has a chronic diseaseFrequentFlyerWhether the customer flies frequentlyEverTravelledAbroadWhether the customer has travelled abroadTravelInsuranceTarget — 1 = Bought, 0 = Not Bought

👨‍💻 Author
OJO EMMANUEL TELEOLA
Student — SQI College of ICT
Machine Learning Classification Project