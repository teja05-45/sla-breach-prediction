import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Ensure matplotlib doesn't require a display (avoids tkinter errors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# local module imports (assumes src/ is on PYTHONPATH or run from project root)
from feature_engineering import load_cleaned
from evaluate_model import save_classification_report, save_confusion_matrix, save_roc_auc, save_feature_importances

# Paths
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def make_onehot_robust(**kwargs):
    """Return OneHotEncoder compatible with installed sklearn version."""
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # fallback for older scikit-learn versions
        if 'sparse_output' in kwargs:
            k = dict(kwargs)
            k['sparse'] = k.pop('sparse_output')
            return OneHotEncoder(**k)
        raise

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if 'sla_breached' in numeric_features:
        numeric_features.remove('sla_breached')
    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    ohe = make_onehot_robust(handle_unknown='ignore', sparse_output=False)
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', ohe)])

    preproc = ColumnTransformer([('num', num_pipe, numeric_features), ('cat', cat_pipe, categorical_features)], remainder='drop')
    return preproc, numeric_features, categorical_features

def get_feature_names(preproc, numeric_features, categorical_features):
    names = numeric_features.copy()
    # attempt to extract onehot names
    try:
        for name, transformer, cols in preproc.transformers_:
            if name == 'num':
                continue
            if name == 'cat':
                pipe = transformer
                ohe = None
                if hasattr(pipe, 'named_steps') and 'onehot' in pipe.named_steps:
                    ohe = pipe.named_steps['onehot']
                elif hasattr(pipe, 'get_feature_names_out'):
                    ohe = pipe
                if ohe is not None:
                    try:
                        cat_ohe_names = list(ohe.get_feature_names_out(cols))
                    except Exception:
                        cat_ohe_names = [f"{c}_ohe" for c in cols]
                    names.extend(cat_ohe_names)
                else:
                    names.extend([f"{c}_ohe" for c in cols])
                break
    except Exception:
        names = numeric_features + categorical_features
    return names

def train_and_save(output_model_path: Path = MODEL_DIR / "sla_model_v1.joblib"):
    # Load cleaned data prepared by src/eda.py or feature_engineering.load_cleaned
    df = load_cleaned()  # loads data/processed/sla_dataset_cleaned.csv by default
    if 'sla_breached' not in df.columns:
        raise ValueError("Target 'sla_breached' not found in cleaned dataframe.")

    X = df.drop(columns=['sla_breached'])
    y = df['sla_breached'].astype(int)

    preproc, numeric_features, categorical_features = build_preprocessor(X)
    pipeline = Pipeline([('preprocessor', preproc), ('classifier', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # ---- SHAP background sample (for explanations in Streamlit) ----
    shap_bg_path = REPORTS_DIR / "shap_background_raw.csv"
    try:
        # Save raw X_train sample (not preprocessed). Up to 200 rows for speed.
        n_sample = min(len(X_train), 200)
        X_train.sample(n_sample, random_state=RANDOM_STATE).to_csv(shap_bg_path, index=False)
        print(f"Saved SHAP background sample to {shap_bg_path}")
    except Exception as e:
        print("Could not save SHAP background sample:", e)
    # ---------------------------------------------------------------

    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    print("Model fitted.")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None

    # classification report
    save_classification_report(y_test, y_pred, out_path=REPORTS_DIR / "classification_report.csv")
    # confusion matrix
    save_confusion_matrix(y_test, y_pred, out_path=PLOTS_DIR / "confusion_matrix.png")
    # roc auc
    if y_proba is not None:
        save_roc_auc(y_test, y_proba, out_path=REPORTS_DIR / "roc_auc.txt")

    # feature importances mapping
    try:
        feature_names = get_feature_names(preproc, numeric_features, categorical_features)
        importances = pipeline.named_steps['classifier'].feature_importances_
        save_feature_importances(feature_names, importances, out_path=REPORTS_DIR / "feature_importances.csv")
    except Exception as e:
        print("Could not save feature importances cleanly:", e)
        try:
            pd.DataFrame({'importance': pipeline.named_steps['classifier'].feature_importances_}).to_csv(REPORTS_DIR / "feature_importances_raw.csv", index=False)
        except Exception:
            pass

    # Save pipeline
    joblib.dump(pipeline, output_model_path)
    print("Saved pipeline to", output_model_path)
    return output_model_path

if __name__ == "__main__":
    model_path = train_and_save()
    print("Done. Model saved to:", model_path)
