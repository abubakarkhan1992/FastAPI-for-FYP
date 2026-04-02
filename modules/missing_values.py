import pandas as pd

def analyze_missing(df):
    missing_count = df.isnull().sum()
    missing_percent = round((missing_count / len(df)) * 100, 2)
    
    return {
        "missing_count": missing_count.to_dict(),
        "missing_percent": missing_percent.to_dict(),
        "missing_percent_mean": missing_percent.mean() if not missing_percent.empty else 0
    }