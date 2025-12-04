📌 SLA Breach Prediction — End-to-End Machine Learning Project

A complete end-to-end Machine Learning project that predicts whether an IT service ticket will breach SLA.
Designed to demonstrate real-world ML skills, including:

Data cleaning

Feature engineering

EDA

Model training & tuning

SHAP model explainability

Batch & single predictions

Streamlit deployment


🚀 Live Demo

Click below to open the deployed Streamlit app:
https://sla-breach-prediction.streamlit.app/

## 📁 Project Structure

```
sla-breach-prediction/
│
├── data/
│   ├── raw/                      # Original dataset (ignored in git)
│   └── processed/                # Cleaned / feature-engineered data
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis notebook
│
├── src/
│   ├── eda.py                    # Automated EDA + plot generation
│   ├── feature_engineering.py    # Feature engineering functions
│   ├── train_model.py            # Model training + tuning + saving
│   ├── evaluate_model.py         # Model evaluation utilities
│   └── streamlit_app.py          # Streamlit UI (single + batch + SHAP)
│
├── models/                       # Saved ML models
├── plots/                        # EDA plots + confusion matrix
├── reports/                      # SHAP background, metrics, reports
│
├── requirements.txt
├── README.md
└── .gitignore
```


🧠 Key Features
✔ End-to-End ML Pipeline

Cleans raw data

Automated EDA

Feature engineering

Builds preprocessing pipelines

Supports hyperparameter tuning

Saves trained models

✔ Streamlit Application

Single prediction with SHAP explanations

Batch prediction via CSV upload

Dynamic results table & plots

Model interpretation panel

✔ Model Explainability (SHAP)

Per-feature contribution values

SHAP bar charts

Transparent prediction reasoning

✔ Recruiter-Ready

Professional folder structure

Polished documentation

Live web app demo

Downloadable dataset

Clean, modular code

📦 Installation

Clone the repository:

git clone https://github.com/teja05-45/sla-breach-prediction.git
cd sla-breach-prediction


Create a virtual environment:

python -m venv venv


Activate it:

Windows

venv\Scripts\activate


macOS/Linux

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🧪 Run EDA

Generates cleaned dataset + EDA plots:

python src/eda.py

🤖 Train the Model
python src/train_model.py


This will generate:

models/sla_model.joblib

SHAP background sample

Evaluation reports

Confusion matrix

Feature importances

🌐 Run Streamlit App
streamlit run src/streamlit_app.py


App Pages:

Page	Features
Single Prediction	Manual inputs → model output + SHAP
Batch Prediction	Upload CSV → predictions + downloadable file
Reports & Plots	Confusion matrix, EDA charts, metrics
📸 Screenshots (Add after deployment)

Place your screenshots here:

![App Screenshot](https://github.com/teja05-45/sla-breach-prediction/blob/main/assets/app.png)
![SHAP Example](https://github.com/teja05-45/sla-breach-prediction/blob/main/assets/shap.png)

🧭 Future Enhancements

Add XGBoost / LightGBM models

Add CI/CD pipeline with GitHub Actions

Deploy to Render / AWS / Azure

Add API layer using FastAPI

Add more SHAP visualizations (waterfall, force plot)

👨‍💻 Author

Teja
Machine Learning & Data Science Enthusiast
GitHub: https://github.com/teja05-45

LinkedIn: https://www.linkedin.com/in/teja-matta-602b3531a


