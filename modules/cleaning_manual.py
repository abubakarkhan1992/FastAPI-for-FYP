import pandas as pd
import numpy as np
from scipy import stats

def smart_type_conversion(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    
    # 1. Standardize Nulls
    null_strings = ["nan", "null", "none", "-", "--", "...", "n/a", "na", "", " ", "ask for price", "tbd", "unknown"]
    for col in cleaned.columns:
        if cleaned[col].dtype == 'object':
            cleaned[col] = cleaned[col].apply(lambda x: np.nan if str(x).strip().lower() in null_strings else x)

    # 2. Extract numerics from messy strings
    for col in cleaned.columns:
        if cleaned[col].dtype == 'object':
            valid_count_before = cleaned[col].dropna().shape[0]
            if valid_count_before == 0:
                continue
            
            temp_series = cleaned[col].astype(str).str.replace(',', '')
            extracted = temp_series.str.extract(r'([-+]?\d*\.\d+|\d+)')[0]
            numeric_series = pd.to_numeric(extracted, errors='coerce')
            
            valid_count_numeric = numeric_series.dropna().shape[0]
            if valid_count_numeric > 0 and (valid_count_numeric / valid_count_before) >= 0.8:
                cleaned[col] = numeric_series
                continue
                
            # Attempt datetime conversion instead
            try:
                datetime_series = pd.to_datetime(cleaned[col], errors='coerce', format='mixed')
                valid_count_dt = datetime_series.dropna().shape[0]
                if valid_count_dt > 0 and (valid_count_dt / valid_count_before) >= 0.8:
                    cleaned[col] = datetime_series
            except:
                pass
    return cleaned

def encode_categoricals(df: pd.DataFrame, method: str) -> pd.DataFrame:
    cleaned = df.copy()
    if method == "One-Hot Encoding":
        obj_cols = cleaned.select_dtypes(include=['object', 'category', 'bool']).columns
        cols_to_encode = [col for col in obj_cols if cleaned[col].nunique() < 50]
        cleaned = pd.get_dummies(cleaned, columns=cols_to_encode, dtype=int)
    elif method == "Label Encoding":
        from sklearn.preprocessing import LabelEncoder
        obj_cols = cleaned.select_dtypes(include=['object', 'category', 'bool']).columns
        for col in obj_cols:
            le = LabelEncoder()
            mask = cleaned[col].notnull()
            if mask.sum() > 0:
                cleaned.loc[mask, col] = le.fit_transform(cleaned.loc[mask, col].astype(str))
            try:
                cleaned[col] = cleaned[col].astype('float64')
            except:
                pass
    return cleaned

def feature_engineering(df: pd.DataFrame, option: str, protected_cols=None) -> pd.DataFrame:
    cleaned = df.copy()
    protected_cols = set(protected_cols or [])

    if option == "Drop ID & Constant Columns":
        cols_to_drop = []
        for col in cleaned.columns:
            if col in protected_cols:
                continue

            nunique = cleaned[col].nunique(dropna=True)
            col_lower = str(col).lower()
            is_id_name = ('serial' in col_lower or 
                          'index' in col_lower or 
                          col_lower == 'id' or 
                          col_lower.endswith('_id') or 
                          col_lower.startswith('id_') or
                          col_lower.endswith('id'))
            
            if nunique <= 1:
                cols_to_drop.append(col)
            elif is_id_name and nunique >= max(2, cleaned.shape[0] * 0.5):
                cols_to_drop.append(col)
            elif nunique >= cleaned.shape[0] * 0.8 and cleaned[col].dtype == 'object':
                cols_to_drop.append(col)
        cleaned.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return cleaned

def manual_clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cleaned_df = df.copy()
    
    # 0. Feature Engineering
    fe_option = config.get("feature_engineering", "None")
    cleaned_df = feature_engineering(cleaned_df, fe_option)
    
    # 0.5 Smart Type Conversion
    cleaned_df = smart_type_conversion(cleaned_df)
    
    # 1. Missing Values Imputation
    mv_option = config.get("missing_values", "None")
    
    if mv_option != "None" and cleaned_df.isnull().sum().sum() > 0:
        if mv_option == "Drop missing values":
            cleaned_df.dropna(inplace=True)
        elif mv_option == "Time Series: Forward Fill":
            cleaned_df.ffill(inplace=True)
        elif mv_option == "Time Series: Interpolate":
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].interpolate(method='linear', limit_direction='both')
            cat_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
            if len(cat_cols) > 0:
                cleaned_df[cat_cols] = cleaned_df[cat_cols].ffill().bfill()
        else:
            from sklearn.impute import SimpleImputer, KNNImputer
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            num_imputer = None
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            if mv_option == "Standard Imputation (Mean/Mode)":
                num_imputer = SimpleImputer(strategy='mean')
            elif mv_option == "Robust Imputation (Median/Mode)":
                num_imputer = SimpleImputer(strategy='median')
            elif mv_option == "Advanced: KNN Imputation":
                num_imputer = KNNImputer(n_neighbors=5)
            elif mv_option == "Advanced: Iterative (Model-based)":
                num_imputer = IterativeImputer(random_state=42)
                
            try:
                if len(numeric_cols) > 0 and num_imputer is not None:
                    cleaned_df[numeric_cols] = num_imputer.fit_transform(cleaned_df[numeric_cols])
                if len(categorical_cols) > 0:
                    cleaned_df[categorical_cols] = cat_imputer.fit_transform(cleaned_df[categorical_cols])
            except Exception as e:
                print(f"Sklearn Imputation failed: {e}")
                pass
            
    # 2. Duplicates
    dup_option = config.get("duplicates", "None")
    if dup_option == "Keep First":
        cleaned_df.drop_duplicates(keep='first', inplace=True)
    elif dup_option == "Keep Last":
        cleaned_df.drop_duplicates(keep='last', inplace=True)
    elif dup_option == "Drop All":
        cleaned_df.drop_duplicates(keep=False, inplace=True)
        
    # 3. Outliers (Numeric columns only)
    outlier_option = config.get("outliers", "None")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    if outlier_option == "IQR":
        for col in numeric_cols:
            if len(cleaned_df) == 0: break
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Keep rows within bounds, or where the value is NaN
            mask = ((cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)) | cleaned_df[col].isnull()
            cleaned_df = cleaned_df[mask]
            
    elif outlier_option == "Z-score":
        for col in numeric_cols:
            if len(cleaned_df) > 0:
                mean, std = cleaned_df[col].mean(), cleaned_df[col].std()
                if std > 0:
                    mask = ((cleaned_df[col] >= mean - 3*std) & (cleaned_df[col] <= mean + 3*std)) | cleaned_df[col].isnull()
                    cleaned_df = cleaned_df[mask]
                    
    # 4. Inconsistencies (Text columns only)
    inc_option = config.get("inconsistencies", "None")
    if inc_option == "Standardize (Lower & Strip)":
        obj_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
                except:
                    pass

    # 4.5 Encoding (Must happen after Outliers so OHE binary cols aren't dropped, and after Inconsistencies)
    enc_option = config.get("encoding", "None")
    cleaned_df = encode_categoricals(cleaned_df, enc_option)

    # 5. Class Imbalance
    imbalance_option = config.get("imbalance", "None")
    target_col = config.get("imbalance_target", "None")
    if imbalance_option in ["Undersample to balance", "Fill with synthetic data (SMOTE)"] and target_col in cleaned_df.columns:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            
            # Clean NaNs as SMOTE fails on them
            cleaned_df.dropna(inplace=True)
            if len(cleaned_df) > 0:
                y = cleaned_df[target_col]
                X = cleaned_df.drop(columns=[target_col])
                
                # Encode text to numeric for SMOTE to run
                is_object = X.select_dtypes(include=['object', 'category']).columns
                for col in is_object:
                    X[col] = X[col].astype("category").cat.codes
                    
                if imbalance_option == "Undersample to balance":
                    rus = RandomUnderSampler(random_state=42)
                    X_res, y_res = rus.fit_resample(X, y)
                else:
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)
                    
                cleaned_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)
        except Exception as e:
            print(f"Imbalance handling failed: {e}")
            pass
            
    return cleaned_df
