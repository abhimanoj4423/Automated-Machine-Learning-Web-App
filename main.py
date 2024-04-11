import streamlit as st
import os
import pandas as pd
import joblib

from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

from pycaret import classification as cl
from pycaret import regression as reg
from pycaret import clustering as cu

with st.sidebar:
    st.title('__Automated Machine Learning WebApp__')
    choice = option_menu(menu_title=None,
                         options=["Upload Dataset", "Profiling", "Model Building", "Download Model", "Prediction"])
    st.info("This Webapp will automate the end-to-end process of building a machine learning model. Just provide the necessary inputs in the above given navigation pages.")

if os.path.exists("dataset.csv"):
    df = pd.read_csv('dataset.csv', index_col=None)


if choice == "Upload Dataset":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset in Comma Separated Format(.csv)")
    if file:
        st.success('File Upload Successfully')
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=False)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if st.button("Generate Profile Report"):
        profile = ProfileReport(df)
        st_profile_report(profile)

if choice == "Model Building":
    st.title('Select the type of ML problem')
    choice1 = option_menu(menu_title=None,
                          options=['Regression', 'Classification', 'Clustering'],
                          orientation='horizontal')
    if choice1 == 'Regression':
        st.title('Regression Model Building')
        target = st.selectbox("Select the Target column", df.columns)
        if st.button("Run Modelling"):
            st.info('The process might take a few minutes, progress is displayed top right corner of the screen.')
            reg.setup(df, target=target)
            setup_df = reg.pull()
            st.dataframe(setup_df)
            best_model = reg.compare_models()
            compare_df = reg.pull()
            reg.save_model(best_model, 'best_model')
            st.dataframe(compare_df)

    if choice1 == 'Classification':
        st.title('Classification Model Building')
        target = st.selectbox("Select the Target column", df.columns)
        if st.button("Run Modelling"):
            st.info('The process might take a few minutes, progress is displayed top right corner of the screen.')
            cl.setup(df, target=target)
            setup_df = cl.pull()
            st.dataframe(setup_df)
            best_model = cl.compare_models()
            compare_df = cl.pull()
            cl.save_model(best_model, 'best_model')
            st.dataframe(compare_df)

    if choice1 == 'Clustering':
        empty = []
        st.title('Clustering Model Building')
        algo = ['kmeans', 'ap', 'sc', 'dbscan', 'hclust']
        clust = st.selectbox("Choose the Clustering Algorithm", algo)
        st.info('There is no automated best model selection currently available for Clustering algorithms, please select the model having the __highest silhouette score__.')
        target = st.selectbox("Choose the target Column", empty)
        if st.button("Run Modelling"):
            st.info('The process might take a few minutes, progress is displayed top right corner of the screen.')
            cu.setup(df)
            setup_df = cu.pull()
            st.dataframe(setup_df)
            model = cu.create_model(clust)
            best_model = cu.evaluate_model(model)
            compare_df = cu.pull()
            cu.save_model(best_model, 'best_model')
            st.dataframe(compare_df)

if choice == "Download Model":
    st.title('Download the Model')
    st.info('The model downloadable is the one with the highest corresponding metrics.')
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model", f, "best_model_test.pkl")

if choice == "Prediction":
    st.title('Import your Test dataset')
    test = st.file_uploader("__Note:__ The test dataset should NOT contain the __target column__")
    if test:
        df1 = pd.read_csv(test, index_col=None)
        df1.to_csv("predicted.csv", index=False)
        with open("best_model.pkl", 'rb') as f:
            clf = joblib.load(f)
        y_pred = clf.predict(df1)
        pred = pd.concat([df1, y_pred], axis=1)
        if st.button("Predict"):
            st.dataframe(pred)
            st.download_button("Download Dataset", pred.to_csv(), "predicted.csv")
