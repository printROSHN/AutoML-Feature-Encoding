import streamlit as st
#import plotly.express as px
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from lazytransform import LazyTransformer
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
import os




if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoEncode-Internal")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Encoding","Download"])
    st.info("This project application helps in Feature Encoding")

if choice == "Upload":
    st.cache_data.clear()

    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.cache_data.clear()

    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df,minimal=True)
    
    st_profile_report(profile_df)


if choice == "Encoding":
    st.cache_data.clear()
    st.cache_resource.clear()

    # 1. read target column
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    

    # 2. Train test split & Data cleaning

    # null_columns = df.columns[df.isnull().mean() > 0.6]
    # df = df.drop(null_columns,axis=1)

    X = df.drop(chosen_target, axis=1)
    y = df[chosen_target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)


    # 3. Apply Encoding and transformation
    encodings_choice = ["auto","onehot","label","target","hash","count","helmert","bdc",
    "sum","loo","base","woe","james"]
    scalers_choice = ["None","std","minmax","robust","maxabs"]
    target_encode = True
    profile_df = ProfileReport(df,minimal=True)



    has_datetime_column = any(df[col].dtype == 'datetime64[ns]' for col in df.columns)

    #choose encoder
    encoder = st.selectbox('Choose the Encoder', encodings_choice)
    scalers_choice = st.selectbox('Choose the Scalers', scalers_choice)

    if(st.button("Encode !")):

        #LAZYTRANSFORMER
        lazy = LazyTransformer(model=None, encoders=encoder, scalers=scalers_choice, 
            date_to_string=has_datetime_column, transform_target=target_encode, #imbalanced=imbalance_value,
            combine_rare=False, verbose=0)
        X_trainm, y_trainm = lazy.fit_transform(X_train, y_train)
        X_testm = lazy.transform(X_test)
        ch_target = y_trainm

        # 4. Print Pipeline

        st.dataframe(X_trainm)
        X,y,tar = X_trainm,y_trainm,chosen_target

        yy = pd.DataFrame(data=y)
        new_dataframe = pd.concat([
            X,
            yy
        ], axis=1, ignore_index=True)


        pipe = lazy.plot_pipeline()
        html_content = estimator_html_repr(pipe)
        st.markdown(html_content, unsafe_allow_html=True)



if choice == "Download":
    st.write("Download Encoded CSV !!")
    with open('encoded_dataset.csv', 'r') as f:
        st.download_button('Download', f, file_name="encoded.csv")
