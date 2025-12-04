from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DIR = Path("data/processed")

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create resolution_seconds and time_to_deadline_seconds from datetimes, if possible."""
    df = df.copy()
    for col in ['created_at', 'resolved_at', 'sla_deadline']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if {'resolved_at', 'created_at'}.issubset(df.columns):
        df['resolution_seconds'] = (df['resolved_at'] - df['created_at']).dt.total_seconds()
    else:
        df['resolution_seconds'] = df.get('resolution_seconds', np.nan)

    if {'sla_deadline', 'created_at'}.issubset(df.columns):
        df['time_to_deadline_seconds'] = (df['sla_deadline'] - df['created_at']).dt.total_seconds()
    else:
        df['time_to_deadline_seconds'] = df.get('time_to_deadline_seconds', np.nan)

    return df

def add_hour_and_business_flag(df: pd.DataFrame, timezone=None) -> pd.DataFrame:
    """
    Add hour_of_day (from created_at) and is_business_hour (Mon-Fri 9-17).
    timezone argument is reserved for future use if you want to localize timestamps.
    """
    df = df.copy()
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['hour_of_day'] = df['created_at'].dt.hour
        df['is_business_hour'] = df['created_at'].dt.weekday.isin(range(0,5)) & df['created_at'].dt.hour.between(9, 17)
        # convert boolean to int for modeling
        df['is_business_hour'] = df['is_business_hour'].astype(int)
    else:
        df['hour_of_day'] = np.nan
        df['is_business_hour'] = np.nan
    return df

def group_rare_categories(df: pd.DataFrame, cat_cols, min_count=50) -> pd.DataFrame:
    """
    For each column in cat_cols, replace categories that appear less than min_count with 'OTHER'.
    Useful to reduce number of one-hot encoded columns.
    """
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        mask_rare = ~df[col].isin(counts[counts >= min_count].index)
        df.loc[mask_rare, col] = 'OTHER'
    return df

def prepare_for_model(df: pd.DataFrame, 
                      drop_raw_time_cols=True,
                      group_rare=True,
                      rare_min_count=50,
                      add_hour_flag=True) -> pd.DataFrame:
    """
    Run a sequence of feature engineering steps and return dataframe ready for the pipeline.
    This DOES NOT run the sklearn preprocessing pipeline (SimpleImputer/OneHot/Scaler) — that lives in train code.
    """
    df = df.copy()

    # create base time features if not already present
    df = create_time_features(df)

    # optionally add hour and business hour flag
    if add_hour_flag:
        df = add_hour_and_business_flag(df)

    # group rare categories (if any)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if group_rare and len(categorical_columns) > 0:
        df = group_rare_categories(df, categorical_columns, min_count=rare_min_count)

    # optionally drop raw datetime columns and ticket id (model pipeline expects numeric/categorical features only)
    if drop_raw_time_cols:
        for c in ['ticket_id', 'created_at', 'resolved_at', 'sla_deadline']:
            if c in df.columns:
                df = df.drop(columns=[c])

    return df

def load_cleaned(path: Path = PROCESSED_DIR / "sla_dataset_cleaned.csv") -> pd.DataFrame:
    """Load the cleaned dataset saved in Step 1 (if present)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {path}")
    df = pd.read_csv(path)
    # ensure desired engineered columns exist
    if 'resolution_seconds' not in df.columns or 'time_to_deadline_seconds' not in df.columns:
        df = create_time_features(df)
    return df

# small helper for building a single-row dataframe from a dict of user inputs
def single_row_df_from_inputs(inputs: dict) -> pd.DataFrame:
    """
    Given a dict with keys:
       ['priority','category','channel','customer_tier','assigned_group',
        'num_reassignments','response_time_first',
        'created_at','resolved_at','sla_deadline']
    returns a one-row DataFrame with engineered features applied.
    """
    df = pd.DataFrame([inputs])
    df = prepare_for_model(df, drop_raw_time_cols=False)  # keep raw for time parsing, drop later if needed
    # after prepare_for_model, drop raw timecols to match training data expected features
    for c in ['ticket_id','created_at','resolved_at','sla_deadline']:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df
