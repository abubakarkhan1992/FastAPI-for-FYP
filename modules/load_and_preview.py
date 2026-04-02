import pandas as pd
import streamlit as st

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

def show_preview(df, uploaded_file):
    st.subheader("Dataset Overview")

    st.subheader("Preview")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    file_size_mb = uploaded_file.size / (1024 * 1024)

    col3.metric("File Size (MB)", round(file_size_mb, 2))

