def analyze_duplicates(df):
    cols_to_check = []
    for col in df.columns:
        col_lower = str(col).lower()
        is_id_name = ('serial' in col_lower or 
                      'index' in col_lower or 
                      col_lower == 'id' or 
                      col_lower.endswith('_id') or 
                      col_lower.startswith('id_') or
                      col_lower.endswith('id'))
        if df[col].nunique(dropna=True) >= df.shape[0] * 0.8 and is_id_name:
            continue
        cols_to_check.append(col)
        
    if not cols_to_check:
        cols_to_check = df.columns
        
    duplicates = int(df.duplicated(subset=cols_to_check).sum())
    percent = float(round(duplicates / len(df) * 100, 2))
    
    return {
        "duplicate_count": duplicates,
        "duplicate_percent": percent
    }