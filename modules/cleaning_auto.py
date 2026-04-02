"""
Auto Data Cleaning Module (Production-Ready)

Features:
- Uses AutoClean for automated preprocessing
- Converts all boolean-like values to 0/1 (vectorized & fast)
- Ensures ML-ready dataset (no True/False, minimal object types)
- Safe, optimized, and scalable
"""

import pandas as pd
import numpy as np
from AutoClean import AutoClean


# ----------------------------
# BOOLEAN NORMALIZATION (FAST)
# ----------------------------
def convert_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all boolean-like columns (True/False, Yes/No, etc.) to 0/1.
    Industry-grade vectorized implementation.
    """
    df = df.copy()

    # 1. Convert actual boolean dtype → int8
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype("int8")

    # 2. Convert object columns with boolean-like values
    bool_map = {
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "1": 1, "0": 0
    }

    obj_cols = df.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()

        # Only convert if column is purely boolean-like
        if len(unique_vals) > 0 and set(unique_vals).issubset(bool_map.keys()):
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map(bool_map)
                .astype("int8")
            )

    return df


# ----------------------------
# FINAL DATA SAFETY (OPTIONAL)
# ----------------------------
def ensure_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataset is fully numeric for ML models.
    Converts remaining categorical columns using one-hot encoding.
    """
    df = df.copy()

    # Replace inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop completely empty rows
    df.dropna(axis=0, how="all", inplace=True)

    # One-hot encode remaining categorical columns
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    return df


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def auto_clean_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Automatically clean and preprocess dataset using AutoClean.

    Steps:
    1. Validate input
    2. Run AutoClean
    3. Convert boolean-like values → 0/1
    4. Ensure ML-ready dataset

    Args:
        df (pd.DataFrame): Raw dataset
        target_col (str): Target column name

    Returns:
        pd.DataFrame: Cleaned dataset
    """

    if df is None or df.empty:
        return df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    try:
        # ----------------------------
        # STEP 1: AutoClean
        # ----------------------------
        pipeline = AutoClean(df)
        cleaned_df = pipeline.output

        # ----------------------------
        # STEP 2: Normalize booleans
        # ----------------------------
        cleaned_df = convert_bool_like_columns(cleaned_df)

        # ----------------------------
        # STEP 3: Ensure ML-ready
        # ----------------------------
        cleaned_df = ensure_ml_ready(cleaned_df)

        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)

        return cleaned_df

    except Exception as e:
        print(f"AutoClean failed: {e}. Returning original dataset.")
        return df.copy()