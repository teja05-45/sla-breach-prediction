import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Ensure non-GUI matplotlib backend if your scripts use plotting
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Config (change N_ITER for quick vs thorough)
N_ITER = 20  # quick tuning; set to 50 for more thorough tuning
CV_FOLDS = 3
RANDOM_STATE = 42

# Paths (relative to project root)
RAW_PATH = Path("data/processed/sla_dataset_cleaned.csv")  # cleaned data from Step 1
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def make_onehot_robust(**kwargs):
    """Return OneHotEncoder compatible with installed sklearn version."""
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # fall back to legacy 'sparse' param name if needed
        if 'sparse_output' in kwargs:
            k = dict(kwargs)
            k['sparse'] = k.pop('sparse_output')
            return OneHotEncoder(**k)
        else:
            raise

def load_and_prepare(path):
    print("Loading:", path)
    df = pd.read_csv(path)
    # If dates are still present in this cleaned file, parse them and recreate features.
    for col in ['created_at','resolved_at','sla_deadline']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # If resolution/time_to_deadline not present, recompute
    if 'resolution_seconds' not in df.columns and {'created_at','resolved_at'}.issubset(df.columns):
        df['resolution_seconds'] = (df['resolved_at'] - df['created_at']).dt.total_seconds()
    if 'time_to_deadline_seconds' not in df.columns and {'sla_deadline','created_at'}.issubset(df.columns):
        df['time_to_deadline_seconds'] = (df['sla_deadline'] - df['created_at']).dt.total_seconds()

    # Drop raw time columns if present
    for c in ['ticket_id','created_at','resolved_at','sla_deadline']:
        if c in df.columns:
            df = df.drop(columns=[c])
    print("Loaded dataframe shape:", df.shape)
    return df

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if 'sla_breached' in numeric_features:
        numeric_features.remove('sla_breached')
    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    ohe = make_onehot_robust(handle_unknown='ignore', sparse_output=False)
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])
    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ], remainder='drop')
    return preproc, numeric_features, categorical_features

def tune(df, n_iter=N_ITER):
    target = 'sla_breached'
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # quick check target balance
    dist = y.value_counts()
    print("Target distribution:\n", dist.to_dict())

    preproc, numeric_features, categorical_features = build_preprocessor(X)

    # Baseline (optional): quick 3-fold CV roc_auc
    from sklearn.pipeline import make_pipeline
    baseline = make_pipeline(preproc, RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    print("Running quick baseline CV (3-fold) ROC AUC ...")
    baseline_scores = cross_val_score(baseline, X, y, cv=3, scoring='roc_auc', n_jobs=-1)
    print("Baseline CV ROC AUC mean:", baseline_scores.mean())
    pd.Series(baseline_scores).to_csv(REPORTS_DIR / "baseline_cv_roc_auc_scores.csv", index=False)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Search space
    param_dist = {
        'classifier__n_estimators': [100,200,300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2,5,10],
        'classifier__min_samples_leaf': [1,2,4],
        'classifier__max_features': ['auto', 'sqrt', 0.5],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    pipeline = Pipeline([('preprocessor', preproc), ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])

    rs = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print(f"Running RandomizedSearchCV with n_iter={n_iter} ... (this can take minutes)")
    rs.fit(X_train, y_train)
    print("Best params:", rs.best_params_)
    print("Best CV ROC AUC:", rs.best_score_)

    # Evaluate on test set
    best = rs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1] if hasattr(best.named_steps['classifier'], 'predict_proba') else None
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print("Test ROC AUC:", auc)

    # Save model and reports
    model_path = MODEL_DIR / "sla_model_tuned.joblib"
    joblib.dump(best, model_path)
    print("Saved tuned pipeline to", model_path)

    # Save classification report and confusion matrix
    rep = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(rep).transpose().to_csv(REPORTS_DIR / "classification_report_tuned.csv")
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv(REPORTS_DIR / "confusion_matrix_tuned.csv", index=False, header=False)
    if auc is not None:
        with open(REPORTS_DIR / "roc_auc_tuned.txt", "w") as f:
            f.write(f"{auc:.6f}\n")

    # Save feature importances (try to map back names)
    try:
        preproc_obj = best.named_steps['preprocessor']
        feat_names = numeric_features.copy()
        # get onehot encoder
        ohe_obj = None
        for name, trans, cols in preproc_obj.transformers_:
            if name == 'cat':
                pipe = trans
                if hasattr(pipe, 'named_steps') and 'onehot' in pipe.named_steps:
                    ohe_obj = pipe.named_steps['onehot']
                elif hasattr(pipe, 'get_feature_names_out'):
                    ohe_obj = pipe
                break
        if ohe_obj is not None:
            try:
                cat_names = list(ohe_obj.get_feature_names_out(categorical_features))
            except Exception:
                cat_names = [f"{c}_?" for c in categorical_features]
            feat_names.extend(cat_names)
        importances = best.named_steps['classifier'].feature_importances_
        fi = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
        fi.to_csv(REPORTS_DIR / "feature_importances_tuned.csv", index=False)
    except Exception as e:
        print("Warning: could not map feature importances:", e)

    return {
        'model_path': model_path,
        'roc_auc_test': auc,
        'best_cv': rs.best_score_,
        'best_params': rs.best_params_
    }

if __name__ == "__main__":
    df = load_and_prepare(RAW_PATH)
    results = tune(df, n_iter=N_ITER)
    print("Tuning complete. Results:", results)
