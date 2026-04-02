def detect_imbalance(df):
    imbalances = {}
    
    for col in df.columns:
        unique_vals = df[col].nunique()
        # Consider a column a classification target candidate if it has 2 to 20 unique values
        if 1 < unique_vals <= 20:
            counts = df[col].value_counts()
            percentages = round((counts / counts.sum()) * 100, 1)
            
            imbalances[str(col)] = {
                "imbalance_score": float(percentages.max()),
                "counts": {str(k): int(v) for k, v in counts.items()},
                "percentages": {str(k): float(v) for k, v in percentages.items()}
            }
            
    return imbalances