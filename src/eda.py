# src/eda.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW_PATH = Path("data/raw/sla_dataset_25000.csv")
PROCESSED_DIR = Path("data/processed")
PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df

def summarize(df: pd.DataFrame):
    print("\nHEAD:")
    print(df.head(10).to_string(index=False))
    print("\nDTYPES:")
    print(df.dtypes)
    print("\nNUMERIC DESCRIBE:")
    print(df.describe(include=[np.number]).transpose())
    print("\nOBJECT DESCRIBE:")
    print(df.describe(include=['object', 'category']).transpose())

def missing_report(df: pd.DataFrame):
    miss_count = df.isnull().sum().sort_values(ascending=False)
    miss_pct = (df.isnull().mean()*100).sort_values(ascending=False)
    miss_table = pd.concat([miss_count, miss_pct], axis=1)
    miss_table.columns = ['missing_count', 'missing_pct']
    miss_table.to_csv(REPORTS_DIR / "missing_values.csv")
    print("Saved missing values to", REPORTS_DIR / "missing_values.csv")
    return miss_table

def parse_dates_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['created_at', 'resolved_at', 'sla_deadline']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if {'resolved_at','created_at'}.issubset(df.columns):
        df['resolution_seconds'] = (df['resolved_at'] - df['created_at']).dt.total_seconds()
    else:
        df['resolution_seconds'] = np.nan
    if {'sla_deadline','created_at'}.issubset(df.columns):
        df['time_to_deadline_seconds'] = (df['sla_deadline'] - df['created_at']).dt.total_seconds()
    else:
        df['time_to_deadline_seconds'] = np.nan

    # drop raw datetimes and ticket id
    for c in ['ticket_id','created_at','resolved_at','sla_deadline']:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

def save_cleaned(df: pd.DataFrame, out_path: Path = PROCESSED_DIR / "sla_dataset_cleaned.csv"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved cleaned dataset to", out_path)

def plot_numeric(df: pd.DataFrame, max_plots: int = 6):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'sla_breached' in numeric_cols:
        numeric_cols.remove('sla_breached')
    for col in numeric_cols[:max_plots]:
        plt.figure(figsize=(6,3))
        df[col].dropna().hist(bins=30)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"hist_{col}.png")
        plt.close()

def plot_categorical_counts(df: pd.DataFrame, cat_cols=None, max_cats=6):
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for col in cat_cols[:max_cats]:
        plt.figure(figsize=(6,3))
        df[col].value_counts().nlargest(20).plot(kind='bar')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"count_{col}.png")
        plt.close()

def run_all(raw_path: Path = RAW_PATH):
    df = load_raw(raw_path)
    summarize(df)
    missing_report(df)
    df_clean = parse_dates_and_engineer(df)
    save_cleaned(df_clean)
    plot_numeric(df_clean)
    plot_categorical_counts(df_clean, cat_cols=['priority','category','channel','customer_tier','assigned_group'], max_cats=5)
    print("EDA complete.")

if __name__ == "__main__":
    run_all()
