from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import io
import json
from pathlib import Path
from ydata_profiling import ProfileReport
from fpdf import FPDF

# Import your existing pure modules
from modules.missing_values import analyze_missing
from modules.duplicates import analyze_duplicates
from modules.outliers import analyze_outliers
from modules.inconsistency import detect_inconsistencies
from modules.imbalance import detect_imbalance
from modules.correlation import correlation_analysis
from modules.quality_score import compute_quality_score
from modules.cleaning_manual import manual_clean_dataset
from modules.cleaning_auto import auto_clean_dataset
from modules.automl_training import (
    detect_problem_type,
    train_automl_model,
    save_model_pickle,
    load_model_pickle,
    get_model_summary,
    make_predictions,
    get_feature_columns
)

app = FastAPI(title="Dataset Analyser API")

# Store report paths for download
stored_reports = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Dataset Analyser API! The server is running successfully. You can test endpoints at http://localhost:8000/docs"}

# ==================== MANUAL ANALYSIS ENDPOINTS ====================

@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Manual data quality analysis endpoint.
    Returns comprehensive quality metrics.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
        
    # Run modular functions
    missing = analyze_missing(df)
    duplicates = analyze_duplicates(df)
    outliers = analyze_outliers(df)
    inconsistencies = detect_inconsistencies(df)
    imbalance = detect_imbalance(df)
    correlation = correlation_analysis(df)
    
    score = compute_quality_score(missing, duplicates, outliers, len(inconsistencies))
    
    return {
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_metrics": missing,
        "duplicate_metrics": duplicates,
        "outlier_metrics": outliers,
        "inconsistencies": inconsistencies,
        "imbalance_metrics": imbalance,
        "correlation": correlation,
        "quality_score": score
    }

@app.post("/analyze/download")
async def download_manual_report(file: UploadFile = File(...)):
    """
    Download manual analysis report as JSON.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
        
    # Run modular functions
    missing = analyze_missing(df)
    duplicates = analyze_duplicates(df)
    outliers = analyze_outliers(df)
    inconsistencies = detect_inconsistencies(df)
    imbalance = detect_imbalance(df)
    correlation = correlation_analysis(df)
    
    score = compute_quality_score(missing, duplicates, outliers, len(inconsistencies))
    
    report_data = {
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_metrics": missing,
        "duplicate_metrics": duplicates,
        "outlier_metrics": outliers,
        "inconsistencies": inconsistencies,
        "imbalance_metrics": imbalance,
        "correlation": correlation,
        "quality_score": score
    }
    
    # Create JSON file for download
    report_json = json.dumps(report_data, indent=2, default=str)
    report_filename = f"manual_report_{Path(file.filename).stem}.json"
    report_path = Path(f"temp_{report_filename}")
    
    with open(report_path, 'w') as f:
        f.write(report_json)
    
    return FileResponse(report_path, media_type='application/json', filename=report_filename)

@app.post("/analyze/download/pdf")
async def download_manual_report_pdf(file: UploadFile = File(...)):
    """
    Download manual analysis report as PDF.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
        
    # Run modular functions
    missing = analyze_missing(df)
    duplicates = analyze_duplicates(df)
    outliers = analyze_outliers(df)
    inconsistencies = detect_inconsistencies(df)
    
    score = compute_quality_score(missing, duplicates, outliers, len(inconsistencies))
    
    report_filename = f"manual_report_{Path(file.filename).stem}.pdf"
    report_path = Path(f"temp_{report_filename}")
    
    # Generate PDF using fpdf2
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Data Quality Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Dataset: {file.filename}", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Total Rows: {int(df.shape[0])}", ln=True)
    pdf.cell(0, 10, f"Total Columns: {int(df.shape[1])}", ln=True)
    pdf.cell(0, 10, f"Quality Score: {score}/10", ln=True)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Missing Values", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Missing Count: {missing.get('missing_count', 0)}", ln=True)
    pdf.cell(0, 10, f"Missing Percent: {missing.get('missing_percent', 0.0)}%", ln=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Duplicates", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Duplicate Count: {duplicates.get('duplicate_count', 0)}", ln=True)
    pdf.cell(0, 10, f"Duplicate Percent: {duplicates.get('duplicate_percent', 0.0)}%", ln=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Outliers", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Overall Outlier Ratio: {outliers.get('outlier_ratio', 0.0)}%", ln=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Inconsistencies", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Detected Patterns: {len(inconsistencies)}", ln=True)
    
    pdf.output(str(report_path))
    
    return FileResponse(report_path, media_type='application/pdf', filename=report_filename)

# ==================== AUTOMATED PROFILING ENDPOINTS ====================

@app.post("/profile")
async def profile_dataset(file: UploadFile = File(...)):
    """
    Automated data profiling using ydata-profiling.
    Returns profile report as HTML report status.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    
    try:
        # Generate profile report
        profile = ProfileReport(df, title=f"Data Profile Report - {file.filename}", minimal=False)
        
        # Store report for later download
        report_id = Path(file.filename).stem
        stored_reports[report_id] = profile
        
        # Get summary stats from profile
        summary = {
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": int(df.select_dtypes(include=['number']).shape[1]),
            "categorical_columns": int(df.select_dtypes(include=['object']).shape[1]),
            "report_id": report_id,
            "status": "Profile generated successfully"
        }
        
        return summary
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/profile/download/{report_id}")
async def download_profile_report(report_id: str):
    """
    Download the profile report as HTML.
    """
    if report_id not in stored_reports:
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    
    profile = stored_reports[report_id]
    
    # Generate HTML report
    report_filename = f"data_profile_{report_id}.html"
    report_path = Path(f"temp_{report_filename}")
    
    profile.to_file(str(report_path))
    
    return FileResponse(report_path, media_type='text/html', filename=report_filename)

@app.post("/profile/download")
async def download_profile_report_direct(file: UploadFile = File(...)):
    """
    Generate and download profile report directly as HTML.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    
    try:
        # Generate profile report
        profile = ProfileReport(df, title=f"Data Profile Report - {file.filename}", minimal=False)
        
        # Generate HTML report
        report_filename = f"data_profile_{Path(file.filename).stem}.html"
        report_path = Path(f"temp_{report_filename}")
        
        profile.to_file(str(report_path))
        
        return FileResponse(report_path, media_type='text/html', filename=report_filename)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ==================== DATA CLEANING ENDPOINTS ====================

@app.post("/clean/manual")
async def clean_manual(file: UploadFile = File(...), config: str = Form(...)):
    """
    Apply manual cleaning based on the config JSON string.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
        
    config_dict = json.loads(config)
    cleaned_df = manual_clean_dataset(df, config_dict)
    
    cleaned_filename = f"cleaned_{file.filename}"
    cleaned_path = Path(f"temp_{cleaned_filename}")
    
    if file.filename.endswith('.csv'):
        cleaned_df.to_csv(cleaned_path, index=False)
        media_type = 'text/csv'
    else:
        cleaned_df.to_excel(cleaned_path, index=False)
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    
    return FileResponse(cleaned_path, media_type=media_type, filename=cleaned_filename)

@app.post("/clean/auto")
async def clean_auto(file: UploadFile = File(...), target_col: str = Form("None")):
    """
    Apply automated cleaning and prepare the dataset for model training.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))

    if target_col == "None" or target_col not in df.columns:
        return JSONResponse(
            status_code=400,
            content={"error": "Please select a valid target column for automated cleaning."}
        )

    try:
        cleaned_df = auto_clean_dataset(df, target_col)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Automated cleaning failed: {e}"})

    if cleaned_df is None or cleaned_df.empty:
        return JSONResponse(
            status_code=400,
            content={"error": "Automated cleaning produced an empty dataset. Please review the uploaded data."}
        )

    cleaned_filename = f"autocleaned_{file.filename}"
    cleaned_path = Path(f"temp_{cleaned_filename}")

    if file.filename.endswith('.csv'):
        cleaned_df.to_csv(cleaned_path, index=False)
        media_type = 'text/csv'
    else:
        cleaned_df.to_excel(cleaned_path, index=False)
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    return FileResponse(cleaned_path, media_type=media_type, filename=cleaned_filename)

# ==================== MODEL TRAINING ENDPOINTS ====================

@app.post("/train")
async def train_model(file: UploadFile = File(...), target_col: str = Form(...)):
    """
    Train an AutoML model using PyCaret.
    Automatically detects if it's a classification or regression problem.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    
    # Validate target column
    if target_col not in df.columns:
        return JSONResponse(
            status_code=400,
            content={"error": f"Target column '{target_col}' not found in dataset"}
        )
    
    # Check for missing values in target column
    if df[target_col].isna().any():
        return JSONResponse(
            status_code=400,
            content={"error": "Target column contains missing values. Please clean the data first."}
        )
    
    # Additional validations
    if len(df) < 50:
        return JSONResponse(
            status_code=400,
            content={"error": "Dataset too small. Need at least 50 rows for reliable model training."}
        )
    
    if len(df.columns) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Dataset must have at least 2 columns (features + target)."}
        )
    
    # Check if target column has enough unique values
    n_unique_target = df[target_col].nunique()
    if n_unique_target < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Target column must have at least 2 different values."}
        )
    
    try:
        # Detect problem type
        problem_type = detect_problem_type(df, target_col)
        
        # Train the model
        results, _ = train_automl_model(
            df=df,
            target_column=target_col,
            problem_type=problem_type,
            verbose=False
        )
        
        # Get summary for response
        summary = get_model_summary(results)
        
        # Store the trained model and metadata for download
        model_id = Path(file.filename).stem
        model_data = {
            'model': results['best_model'],
            'problem_type': problem_type,
            'target_column': target_col,
            'model_name': results['best_model_name'],
            'dataset_filename': file.filename
        }
        
        # Store in a simple dict (in production, use a database)
        if not hasattr(app, 'trained_models'):
            app.trained_models = {}
        app.trained_models[model_id] = model_data
        
        return {
            "status": "success",
            "model_id": model_id,
            "training_results": summary,
            "message": f"Model trained successfully! Problem type detected: {problem_type}"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model training failed: {str(e)}"}
        )

@app.get("/train/download/{model_id}")
async def download_trained_model(model_id: str):
    """
    Download the trained model as a pickle file.
    """
    if not hasattr(app, 'trained_models') or model_id not in app.trained_models:
        return JSONResponse(
            status_code=404,
            content={"error": "Model not found. Please train a model first."}
        )
    
    try:
        model_data = app.trained_models[model_id]
        model = model_data['model']
        model_name = model_data['model_name']
        
        # Save model to pickle format
        model_filename = f"model_{model_id}_{model_name}.pkl"
        model_path = Path(f"temp_{model_filename}")
        
        if save_model_pickle(model, model_name, str(model_path)):
            return FileResponse(
                model_path,
                media_type='application/octet-stream',
                filename=model_filename
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to save model"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to download model: {str(e)}"}
        )

@app.post("/train/download/direct")
async def download_trained_model_direct(file: UploadFile = File(...), target_col: str = Form(...)):
    """
    Train a model and download it directly in one request.
    """
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    
    # Validate target column
    if target_col not in df.columns:
        return JSONResponse(
            status_code=400,
            content={"error": f"Target column '{target_col}' not found in dataset"}
        )
    
    # Check for missing values in target column
    if df[target_col].isna().any():
        return JSONResponse(
            status_code=400,
            content={"error": "Target column contains missing values. Please clean the data first."}
        )
    
    try:
        # Detect problem type
        problem_type = detect_problem_type(df, target_col)
        
        # Train the model
        results, _ = train_automl_model(
            df=df,
            target_column=target_col,
            problem_type=problem_type,
            verbose=False
        )
        
        model = results['best_model']
        model_name = results['best_model_name']
        
        # Save model to pickle format
        model_filename = f"model_{Path(file.filename).stem}_{model_name}.pkl"
        model_path = Path(f"temp_{model_filename}")
        
        if save_model_pickle(model, model_name, str(model_path)):
            return FileResponse(
                model_path,
                media_type='application/octet-stream',
                filename=model_filename
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to save model"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model training/download failed: {str(e)}"}
        )

# ==================== MODEL PREDICTION ENDPOINTS ====================

@app.post("/train/test/{model_id}")
async def test_model(model_id: str, file: UploadFile = File(...)):
    """
    Test a trained model by making predictions on provided data.
    """
    if not hasattr(app, 'trained_models') or model_id not in app.trained_models:
        return JSONResponse(
            status_code=404,
            content={"error": "Model not found. Please train a model first."}
        )
    
    try:
        # Read test data
        contents = await file.read()
        if file.filename.endswith('.csv'):
            test_df = pd.read_csv(io.BytesIO(contents))
        else:
            test_df = pd.read_excel(io.BytesIO(contents))
        
        # Get the trained model
        model_data = app.trained_models[model_id]
        model = model_data['model']
        target_column = model_data['target_column']
        
        # Remove target column if it exists
        if target_column in test_df.columns:
            X_test = test_df.drop(columns=[target_column])
        else:
            X_test = test_df
        
        # Make predictions
        predictions_data = make_predictions(model, X_test)
        
        return {
            "status": "success",
            "model_id": model_id,
            "n_predictions": predictions_data['n_predictions'],
            "predictions": predictions_data['predictions'][:10],  # Return first 10 for preview
            "total_predictions": len(predictions_data['predictions']),
            "probabilities": predictions_data.get('probabilities', [])[:10] if 'probabilities' in predictions_data else None,
            "message": f"Made {len(predictions_data['predictions'])} predictions successfully"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@app.post("/train/test/upload")
async def test_model_upload(model_file: UploadFile = File(...), data_file: UploadFile = File(...)):
    """
    Test a previously saved pickle model by uploading both the model file and test data.
    """
    try:
        # Read model pickle file
        model_contents = await model_file.read()
        import pickle as pkl
        model = pkl.loads(model_contents)
        
        # Read test data
        data_contents = await data_file.read()
        if data_file.filename.endswith('.csv'):
            test_df = pd.read_csv(io.BytesIO(data_contents))
        else:
            test_df = pd.read_excel(io.BytesIO(data_contents))
        
        # Use all columns as features (no target column to remove)
        X_test = test_df
        
        # Make predictions
        predictions_data = make_predictions(model, X_test)
        
        return {
            "status": "success",
            "n_predictions": predictions_data['n_predictions'],
            "predictions": predictions_data['predictions'][:10],  # Return first 10 for preview
            "total_predictions": len(predictions_data['predictions']),
            "probabilities": predictions_data.get('probabilities', [])[:10] if 'probabilities' in predictions_data else None,
            "message": f"Made {len(predictions_data['predictions'])} predictions successfully"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model loading or prediction failed: {str(e)}"}
        )

@app.post("/train/test/sample/{model_id}")
async def test_model_with_sample(model_id: str, sample_data: str = Form(...)):
    """
    Test a trained model with sample input data (JSON format).
    """
    if not hasattr(app, 'trained_models') or model_id not in app.trained_models:
        return JSONResponse(
            status_code=404,
            content={"error": "Model not found. Please train a model first."}
        )
    
    try:
        # Parse sample data
        data_dict = json.loads(sample_data)
        sample_df = pd.DataFrame([data_dict])
        
        # Get the trained model
        model_data = app.trained_models[model_id]
        model = model_data['model']
        
        # Make prediction
        predictions_data = make_predictions(model, sample_df)
        
        return {
            "status": "success",
            "prediction": predictions_data['predictions'][0] if predictions_data['predictions'] else None,
            "probability": predictions_data.get('probabilities', [[]])[0] if 'probabilities' in predictions_data and predictions_data['probabilities'] else None,
            "message": "Single prediction made successfully"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )
