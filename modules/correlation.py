import numpy as np
import pandas as pd

def correlation_analysis(df):
    # Filter out ID-like columns and 100% unique string columns
    cols_to_keep = []
    for col in df.columns:
        col_lower = str(col).lower()
        is_id_name = ('serial' in col_lower or 
                      'index' in col_lower or 
                      col_lower == 'id' or 
                      col_lower.endswith('_id') or 
                      col_lower.startswith('id_') or
                      col_lower.endswith('id'))
        
        nunique = df[col].nunique(dropna=True)
        # Drop ID-like columns that are mostly unique
        if is_id_name and nunique >= df.shape[0] * 0.8:
            continue
            
        # Drop text columns that are 100% unique (e.g. Names, UUIDs) 
        # because they provide zero correlative variance
        if df[col].dtype == 'object' and nunique == df.shape[0]:
            continue
            
        cols_to_keep.append(col)
        
    filtered_df = df[cols_to_keep].copy()
    
    # Factorize categorical columns into numeric labels for matrix generation
    for col in filtered_df.columns:
        if filtered_df[col].dtype == 'object' or filtered_df[col].dtype.name == 'category' or filtered_df[col].dtype == 'bool':
            labels, _ = pd.factorize(filtered_df[col])
            filtered_df[col] = np.where(labels == -1, np.nan, labels)
            
    numeric_df = filtered_df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return {"matrix": {}}
        
    corr = numeric_df.corr().round(2)
    # Convert correlation matrix index and columns to string to ensure JSON serialization
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    
    return {"matrix": corr.to_dict()}