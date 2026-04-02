import pandas as pd

def detect_inconsistencies(df):
    issues = []
    
    for col in df.columns:
        series = df[col]
        
        # 1. Case Inconsistency
        if series.dtype == "object":
            values = series.dropna().astype(str)
            sample = values.sample(min(len(values), 1000), random_state=42)
            if sample.str.lower().nunique() < sample.nunique():
                issues.append({
                    "column": str(col),
                    "issue": "Case inconsistency",
                    "suggestion": "Convert all values to lowercase or standard format"
                })
                
        # 2. Numeric Stored as Object
        if series.dtype == "object":
            cleaned = series.astype(str).str.replace(",", "", regex=False)
            numeric_conversion = pd.to_numeric(cleaned, errors="coerce")
            valid_numeric = numeric_conversion.notnull().sum()
            if valid_numeric > len(series) * 0.6:
                invalid_values = series[numeric_conversion.isnull()].dropna().unique()
                issues.append({
                    "column": str(col),
                    "issue": "Numeric column contains invalid values",
                    "invalid_values": [str(v) for v in invalid_values[:5]],
                    "suggestion": "Remove non-numeric text or replace invalid values"
                })
                
        # 3. Special Case: Units
        if series.dtype == "object":
            values = series.dropna().astype(str)
            if values.str.contains("kms", case=False).any():
                issues.append({
                    "column": str(col),
                    "issue": "Contains unit 'kms'",
                    "suggestion": "Remove 'kms' and convert to numeric"
                })
            if values.str.contains("price", case=False).any():
                issues.append({
                    "column": str(col),
                    "issue": "Contains non-numeric price labels",
                    "suggestion": "Replace 'Ask for Price' with NaN or estimated value"
                })
                
        # 4. Date Detection
        if series.dtype == "object":
            date_keywords = ["date", "time", "year", "month", "day"]
            if any(k in str(col).lower() for k in date_keywords):
                sample = series.dropna().astype(str).sample(min(len(series), 1000), random_state=42)
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notnull().sum() > len(sample) * 0.6:
                    issues.append({
                        "column": str(col),
                        "issue": "Stored as object but appears to be datetime",
                        "suggestion": "Convert column to datetime format"
                    })
                    
    return issues