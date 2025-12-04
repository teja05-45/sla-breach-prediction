import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# try to import shap; we'll show a helpful message if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from feature_engineering import single_row_df_from_inputs, prepare_for_model, load_cleaned

ROOT = Path(".")
MODEL_PATHS = [
    ROOT / "models" / "sla_model_tuned.joblib",
    ROOT / "models" / "sla_model_v1.joblib",
    ROOT / "models" / "sla_model_example.joblib"
]
PLOTS_DIR = ROOT / "plots"
REPORTS_DIR = ROOT / "reports"
SHAP_BG_PATH = REPORTS_DIR / "shap_background_raw.csv"

@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if p.exists():
            model = joblib.load(p)
            return model, p
    return None, None

def list_files(dir_path: Path, exts):
    if not dir_path.exists():
        return []
    files = []
    for ext in exts:
        files.extend(sorted(dir_path.glob(f"*{ext}")))
    return files

def load_feature_importances():
    path1 = REPORTS_DIR / "feature_importances_tuned.csv"
    path2 = REPORTS_DIR / "feature_importances.csv"
    path3 = REPORTS_DIR / "feature_importances_example.csv"
    for p in [path1, path2, path3]:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None

def load_shap_background():
    """Load raw background sample saved during training (or None)."""
    if SHAP_BG_PATH.exists():
        try:
            df_bg = pd.read_csv(SHAP_BG_PATH)
            return df_bg
        except Exception as e:
            st.warning(f"Could not load SHAP background sample: {e}")
            return None
    return None

def compute_shap_for_model_and_input(model, df_input, shap_bg_raw=None):
    """
    Compute SHAP values for a fitted pipeline 'model' and one-row df_input (raw features).
    Returns: (feature_names, shap_values_for_class1 (1d array), base_value) or None on failure.
    """
    if not SHAP_AVAILABLE:
        st.warning("SHAP library not installed. Install shap (pip install shap) and restart the app to enable explanations.")
        return None

    # model must be a pipeline with named_steps 'preprocessor' and 'classifier'
    if not hasattr(model, "named_steps") or 'preprocessor' not in model.named_steps or 'classifier' not in model.named_steps:
        st.warning("Model pipeline does not expose 'preprocessor' and 'classifier' steps required for SHAP.")
        return None

    preproc = model.named_steps['preprocessor']
    clf = model.named_steps['classifier']

    if shap_bg_raw is None:
        shap_bg_raw = load_shap_background()
    if shap_bg_raw is None:
        st.warning("No SHAP background sample found (reports/shap_background_raw.csv). Please run training script that saves it.")
        return None

    # drop target if present
    if 'sla_breached' in shap_bg_raw.columns:
        shap_bg_raw = shap_bg_raw.drop(columns=['sla_breached'])

    try:
        X_bg_trans = preproc.transform(shap_bg_raw)
        X_input_trans = preproc.transform(df_input)
    except Exception as e:
        st.error(f"Error transforming data with pipeline preprocessor: {e}")
        return None

    try:
        explainer = shap.TreeExplainer(clf, data=X_bg_trans, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_input_trans)
        # shap_vals might be list (per class) or array
        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            shap_for_class1 = np.array(shap_vals[1]).reshape(-1)
            base_value = explainer.expected_value[1] if hasattr(explainer, "expected_value") else None
        else:
            shap_arr = np.array(shap_vals).reshape(-1)
            shap_for_class1 = shap_arr
            base_value = explainer.expected_value if hasattr(explainer, "expected_value") else None
    except Exception as e:
        st.error(f"SHAP explainer error: {e}")
        return None

    # reconstruct feature names after preprocessing
    try:
        feature_names = []
        for name, transformer, cols in preproc.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
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
                    feature_names.extend(cat_ohe_names)
                else:
                    feature_names.extend([f"{c}_ohe" for c in cols])
    except Exception:
        # fallback to input raw column names
        feature_names = list(df_input.columns)

    return feature_names, shap_for_class1, base_value

def show_reports_and_plots():
    st.header("Reports & Plots")
    st.markdown("Browse generated reports (CSV/TXT) and plots (images). These are produced by the training/tuning scripts.")

    # Plots
    imgs = list_files(PLOTS_DIR, exts=[".png", ".jpg", ".jpeg"])
    if imgs:
        st.subheader("Plots")
        img_names = [p.name for p in imgs]
        sel = st.selectbox("Choose plot to display", options=["--select--"] + img_names)
        if sel != "--select--":
            sel_path = PLOTS_DIR / sel
            st.image(str(sel_path), use_column_width=True)
            st.caption(f"Path: {sel_path}")
    else:
        st.write("No plots found in `plots/`.")

    # Reports (CSV/TXT)
    csvs = list_files(REPORTS_DIR, exts=[".csv", ".txt"])
    if csvs:
        st.subheader("Reports (CSV / TXT)")
        csv_names = [p.name for p in csvs]
        sel_csv = st.selectbox("Choose report to open", options=["--select--"] + csv_names, index=0)
        if sel_csv != "--select--":
            sel_path = REPORTS_DIR / sel_csv
            if sel_path.suffix.lower() == ".txt":
                st.markdown(f"**{sel_csv}**")
                text = sel_path.read_text(encoding='utf-8')
                st.code(text)
                st.download_button("Download report", data=text.encode('utf-8'), file_name=sel_csv, mime="text/plain")
            else:
                try:
                    df = pd.read_csv(sel_path)
                    st.markdown(f"**{sel_csv}** — {df.shape[0]} rows × {df.shape[1]} cols")
                    st.dataframe(df)
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", data=csv_bytes, file_name=sel_csv, mime="text/csv")
                except Exception as e:
                    st.error(f"Could not load CSV: {e}")
    else:
        st.write("No reports found in `reports/`.")

def predict_single(model):
    st.header("Single ticket prediction")
    st.markdown("Enter ticket values (date-times parseable, e.g. 2023-07-01 09:30).")
    with st.form("single_form"):
        priority = st.selectbox("priority", options=["Low","Medium","High","Critical"], index=1)
        category = st.text_input("category", value="software")
        channel = st.selectbox("channel", options=["email","phone","portal","chat"], index=0)
        customer_tier = st.selectbox("customer_tier", options=["bronze","silver","gold","platinum"], index=1)
        assigned_group = st.text_input("assigned_group", value="L1 Support")
        num_reassignments = st.number_input("num_reassignments", min_value=0, max_value=100, value=0)
        response_time_first = st.number_input("response_time_first (seconds)", min_value=0.0, value=3600.0, step=60.0, format="%.1f")
        created_at = st.text_input("created_at", value="2023-07-01 08:00")
        resolved_at = st.text_input("resolved_at", value="2023-07-01 08:30")
        sla_deadline = st.text_input("sla_deadline", value="2023-07-01 09:00")
        submitted = st.form_submit_button("Predict")
    if submitted:
        inputs = {
            "priority": priority,
            "category": category,
            "channel": channel,
            "customer_tier": customer_tier,
            "assigned_group": assigned_group,
            "num_reassignments": num_reassignments,
            "response_time_first": response_time_first,
            "created_at": created_at,
            "resolved_at": resolved_at,
            "sla_deadline": sla_deadline
        }
        df_in = single_row_df_from_inputs(inputs)
        try:
            proba = model.predict_proba(df_in)[:,1][0] if hasattr(model, "predict_proba") else None
            pred = int(model.predict(df_in)[0])
            st.metric("Predicted SLA breached (1 = breached)", pred)
            if proba is not None:
                st.write(f"Predicted probability of breach: **{proba:.3f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Input dataframe head:")
            st.write(df_in.head())
            return

        # SHAP explanations for single input (robust handling for length mismatches)
        if SHAP_AVAILABLE:
            shap_bg = load_shap_background()
            shap_res = compute_shap_for_model_and_input(model, df_in, shap_bg_raw=shap_bg)
            if shap_res is not None:
                feature_names, shap_vals, base = shap_res

                # Safety: ensure arrays and fallback names
                try:
                    shap_arr = np.asarray(shap_vals).flatten()
                except Exception:
                    st.warning("SHAP values could not be converted to array.")
                    shap_arr = np.array([])

                # If feature_names is not a list, try to recover from df_in or preprocessor
                if not isinstance(feature_names, (list, tuple, np.ndarray)):
                    try:
                        feature_names = list(df_in.columns)
                    except Exception:
                        feature_names = []

                feature_names = list(feature_names)  # ensure list

                # Debug info: show lengths so you can see what's wrong if mismatch
                st.write(f"SHAP: feature_names_len = {len(feature_names)}, shap_values_len = {len(shap_arr)}")
                if len(feature_names) == 0:
                    st.warning("No feature names were recovered; showing input columns instead.")
                    feature_names = list(df_in.columns)

                # If lengths mismatch, align by truncating to the smaller length (safe fallback)
                if len(feature_names) != len(shap_arr):
                    min_len = min(len(feature_names), len(shap_arr))
                    st.warning(
                        f"SHAP length mismatch detected — truncating to min length {min_len}. "
                        "This happens when the preprocessor's one-hot expansion differs from the saved feature names."
                    )
                    feature_names = feature_names[:min_len]
                    shap_arr = shap_arr[:min_len]

                # If still empty, abort gracefully
                if len(feature_names) == 0 or shap_arr.size == 0:
                    st.info("No SHAP contributions available to display.")
                else:
                    # Build contribution dataframe and show top contributors
                    contrib = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': shap_arr
                    })
                    contrib['abs_shap'] = contrib['shap_value'].abs()
                    contrib = contrib.sort_values('abs_shap', ascending=False).head(20)
                    st.subheader("Top SHAP feature contributions (for predicted breach probability)")
                    try:
                        st.table(contrib[['feature', 'shap_value']].set_index('feature'))
                    except Exception:
                        st.write(contrib[['feature', 'shap_value']])

                    # Bar chart of absolute SHAP importance
                    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(contrib))))
                    ax.barh(contrib['feature'], contrib['abs_shap'])
                    ax.set_xlabel("abs(SHAP value)")
                    ax.invert_yaxis()
                    st.pyplot(fig)

                    # Helpful debug sample of names/values (hidden by default)
                    with st.expander("Debug: show top 10 (feature name, shap)"):
                        sample = contrib[['feature','shap_value']].head(10)
                        st.write(sample)
            else:
                st.info("SHAP explanation not available for this input.")
        else:
            st.info("Install `shap` to enable explanations (add to requirements and pip install shap).")

def predict_batch(model):
    st.header("Batch predictions (CSV upload)")
    st.markdown("Upload a CSV with the same columns as the original dataset (or at least the required fields). The app will run feature engineering and predictions.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Sample of uploaded data:")
        st.write(df.head())
        df_prepared = prepare_for_model(df, drop_raw_time_cols=False, group_rare=True, rare_min_count=50, add_hour_flag=True)
        # drop raw timecols
        for c in ['ticket_id','created_at','resolved_at','sla_deadline']:
            if c in df_prepared.columns:
                df_prepared = df_prepared.drop(columns=[c])
        try:
            preds = model.predict(df_prepared)
            proba = model.predict_proba(df_prepared)[:,1] if hasattr(model, "predict_proba") else None
            out = df.copy()
            out["pred_sla_breached"] = preds
            if proba is not None:
                out["pred_proba_breach"] = proba
            st.write("Predictions (first 10 rows):")
            st.write(out.head(10))
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name=f"predictions_{int(time.time())}.csv", mime="text/csv")
            save_path = Path("data/processed") / f"predictions_{int(time.time())}.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(save_path, index=False)
            st.success(f"Saved predictions to {save_path}")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

        # Optionally show SHAP summary for the uploaded batch
        if SHAP_AVAILABLE and st.checkbox("Show SHAP summary for uploaded batch (slow for large files)"):
            try:
                shap_bg = load_shap_background()
                preproc = model.named_steps['preprocessor']
                if shap_bg is None:
                    st.warning("No shap background sample available; cannot compute batch SHAP.")
                else:
                    X_bg_trans = preproc.transform(shap_bg.drop(columns=['sla_breached'], errors='ignore'))
                    X_trans = preproc.transform(df_prepared)
                    explainer = shap.TreeExplainer(model.named_steps['classifier'], data=X_bg_trans, feature_perturbation="interventional")
                    shap_vals_batch = explainer.shap_values(X_trans)
                    if isinstance(shap_vals_batch, list) and len(shap_vals_batch) >= 2:
                        shap_arr = np.array(shap_vals_batch[1])
                    else:
                        shap_arr = np.array(shap_vals_batch)
                    mean_abs = np.abs(shap_arr).mean(axis=0)
                    # try to get feature names
                    feature_names = []
                    for name, transformer, cols in preproc.transformers_:
                        if name == 'num':
                            feature_names.extend(cols)
                        elif name == 'cat':
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
                                feature_names.extend(cat_ohe_names)
                    imp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).head(30)
                    fig, ax = plt.subplots(figsize=(6, max(3, 0.2 * len(imp_df))))
                    ax.barh(imp_df['feature'], imp_df['mean_abs_shap'])
                    ax.invert_yaxis()
                    st.pyplot(fig)
            except Exception as e:
                st.error("Could not compute batch SHAP summary: " + str(e))

def main():
    st.title("SLA Breach Prediction App")
    st.sidebar.title("Navigation")
    model, model_path = load_model()
    if model is None:
        st.sidebar.error("No trained model found in models/. Run training first.")
    else:
        st.sidebar.write(f"Loaded model: {model_path.name}")

    page = st.sidebar.radio("Page", ["Predict Single", "Predict Batch", "Reports & Plots"])

    if page == "Predict Single":
        if model is None:
            st.error("Model not loaded.")
        else:
            predict_single(model)
    elif page == "Predict Batch":
        if model is None:
            st.error("Model not loaded.")
        else:
            predict_batch(model)
    else:
        show_reports_and_plots()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Project: sla-breach-prediction")
    st.sidebar.markdown("Author: you")

if __name__ == "__main__":
    main()
