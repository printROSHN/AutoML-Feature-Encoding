import streamlit as st
import plotly.express as px 

import os
import pandas as pd

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto Streamlit")
    choice = st.radio("Navigation", ["Upload", "Profile", "Machine Learning", "Download"])
    st.info("Hello")

if os.path.exists("dataset.csv"): 
    df = pd.read_csv('dataset.csv', index_col=None)


if choice == "Upload": 
    st.title("Upload")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

if choice == "Profile": 
    st.title("Profile")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Machine Learning": 
    target = st.selectbox("Choose the Target", df.columns)
    if st.button("Run Modelling"): 
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        save_model(best_model, 'best_model')
        st.dataframe(compare_df)

if choice == "Download":
    with open("best_model.pkl", 'rb') as f: 
        st.download_button("Download Model", f, "best_model_test.pkl")


# numpy~=1.20.3
# pandas~=1.2.4
# scikit-learn~=0.24.2
# python-dateutil~=2.8.1
# lightgbm>=3.2.1
# imbalanced-learn>=0.8.0
# matplotlib~=3.4.2
# category-encoders~=2.2.2
# tqdm~=4.61.1
# Analysis

# shap>=0.38.0
# interpret>=0.2.7
# umap-learn>=0.5.2
# pandas-profiling>=3.1.0
# explainerdashboard>=0.3.8  # For dashboard method
# Flask==2.2.3  # https://github.com/oegedijk/explainerdashboard/issues/259
# bokeh<3.0.0  # For autoviz
# autoviz>=0.1.36  # For EDA method
# fairlearn==0.7.0  # For check_fairness method
# deepchecks>=0.9.2  # For deep_check method
# streamlit==0.88.0
# pandas-profiling==3.1.0
# streamlit-pandas-profiling==0.1.3
# scikit-learn==0.24.2
# numpy==1.20.3 
# pandas==1.2.5 
# tqdm==4.61.2
# lazytransform==1.1