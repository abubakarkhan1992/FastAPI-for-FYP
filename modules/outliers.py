import pandas as pd
import numpy as np

def analyze_outliers(df):
    numeric_df = df.select_dtypes(include=np.number).copy()
    
    cols_to_drop = []
    for col in numeric_df.columns:
        if "id" in col.lower() or numeric_df[col].nunique() == len(df):
            cols_to_drop.append(col)
            
    numeric_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    
    if numeric_df.empty:
        return {"outlier_counts": {}, "outlier_ratio": 0}
        
    outlier_counts = {}
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        count = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        outlier_counts[str(col)] = count

    total_outliers = sum(outlier_counts.values())
    total_values = numeric_df.size
    ratio = float((total_outliers / total_values) * 100) if total_values > 0 else 0
    
    return {
        "outlier_counts": outlier_counts,
        "outlier_ratio": ratio
    }