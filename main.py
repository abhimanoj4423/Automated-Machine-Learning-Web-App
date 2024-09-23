import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib

from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import dtale
from dtale.app import build_app
from streamlit.components.v1 import iframe
import threading

from pycaret import classification as cl
from pycaret import regression as reg
from pycaret import clustering as cu

from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
from sklearn.decomposition import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.manifold import TSNE

import google.generativeai as genai

genai_api_key = st.secrets["GENAI_API_KEY"]
type = st.secrets["TYPE"]
id = st.secrets["ID"]

st.set_page_config(layout="wide",initial_sidebar_state="expanded")

with st.sidebar:
    st.title('__Automated Machine Learning WebApp v 2.0__')
    choice = option_menu(menu_title=None,
                         options=["Home","Upload Dataset", "Exploratory Analysis","Model Building","Code Generation", "New Data Prediction"])
    st.info("This Webapp will automate the end-to-end process of building a machine learning model. Just provide the necessary inputs in the above given navigation pages.")

if choice == "Home":
    st.title('Automated Machine Learning WebApp v2.0')
    st.write("The AutoML web app automates the end-to-end process of creating a Machine Learning model. With Automated features like __Model Building, Automatic Fine Tuning, Encoding Catergorical Columns, Performing EDA__ and Semi-Automated features such as __Data Preprocessing, Feature Selection, Feature Reduction__. Moreover, providing a downloadable fine-tuned model along with __generating code__ for the steps undertaken using GenAI.")

    with st.container(border=True):
        st.subheader("Upload your dataset",divider='rainbow')
        st.write("Upload the train Dataset in __CSV Format__ and the dataframe will be displayed along with the shape.")
    with st.container(border=True):
        st.subheader("Exploratory Data Analysis",divider='rainbow')
        st.write("Perform automated and semi-automated Explorotary Analyis to understand the data distribution and the relationship between the different Features.")
    with st.container(border=True):
        st.subheader("Model Building",divider='rainbow')
        st.write("Perform assisted Feature Selection or Feature Reduction on the Data along with conducting various Data Preprocessings steps such as __Encoding Catergorical Feature, Imputing Missing Values and Scaling the Numerical Features. The system automatically detects the type of Machine learning problem when the target column is provided and Automated Machine learning can be achieved by the click of one button, which will provide the __Best fitting model__ on the training data along with an option for __Hyperparameter Tuning the Best model__")
    with st.container(border=True):
        st.subheader("Code Generation",divider='rainbow')
        st.write("The Code Generation feature will generate python code blocks based on the each steps the user have performed during the __Model Building__ phase.")
    with st.container(border=True):
        st.subheader("New Data Prediction",divider='rainbow')
        st.write("The Created model predicts the output of New data given in the form of a CSV file, after autamatically performing all the preprocessing steps taked during the model building phase.")

if os.path.exists("dataset.csv"):
    df = pd.read_csv('dataset.csv', index_col=None)

if os.path.exists("steps.csv"):
    steps_df = pd.read_csv('steps.csv', index_col=None)

if choice == "Upload Dataset":
    st.title("Upload Your Dataset")
    st.subheader("Upload Your Dataset in Comma Separated Format(.csv)",divider='rainbow')
    file = st.file_uploader("")
    if file:
        st.success('File Upload Successfully')
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=False)
        st.dataframe(df)
        st.text(f'Shape: {df.shape}')

def run_dtale():
    dtale_app = build_app()
    dtale_app.run(host="localhost", port=40000, debug=False)

if choice == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    choice1 = option_menu(menu_title=None,options=["Automated EDA Generation","Manual EDA"],orientation='horizontal')
    if choice1 == 'Automated EDA Generation':
        st.subheader("Click the button below to generate Automated EDA",divider='rainbow')
        if st.button("Generate Automated EDA"):
            profile = ProfileReport(df)
            st_profile_report(profile)
    if choice1 == 'Manual EDA':
        st.subheader("Click the button below to generate D-Tale EDA",divider='rainbow')
        if st.button('Generate Manual EDA'):
            thread = threading.Thread(target=run_dtale)
            thread.daemon = True
            thread.start()
            d = dtale.show(df, ignore_duplicate=True, open_browser=False)
            dtale_url = d._main_url
            iframe(dtale_url, height=800, width=1000)

data=df.copy()
X=pd.DataFrame()
y=pd.DataFrame()

Simple_Imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

Imp_op = {"No Imputation":"None",'KNN_Imputer': KNNImputer(n_neighbors=1),
'Simple_Imputer': SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
"Iterative_Imputer": IterativeImputer(random_state=0)}

# Determine the type of Machine Learning Problem
ml = {'Clustering':cu,'Classification':cl,'Regression':reg} 


def ml_type(dataframe,target):
    if target == None:
        return 'Clustering'
    elif dataframe[target].nunique() < 5:
        return 'Classification'
    else:
        return 'Regression'

# Feature selection Function for Supervised
def select_k_best_features(dataframe,target, k, select_scores,score_func):
    if score_func== "Custom Features":
        selected_features = []

    else:
        fs_data=dataframe.copy()
        fs_data = pd.DataFrame(Simple_Imputer.fit_transform(dataframe),columns=dataframe.columns)

        cat_col=list(fs_data.select_dtypes(include='object').columns)
        le = LabelEncoder()

        for col in cat_col:
            fs_data[col] = fs_data[col].astype(str)
            fs_data[col]=(le.fit_transform(fs_data[col]))

        X=fs_data.drop(target,axis=1)
        y=fs_data[target]

        sf = select_scores[score_func]
        selector = SelectKBest(score_func=sf, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = X.columns[mask]

    return list(selected_features)

def feature_reduction(scaled_data,n_comp,fr,tech):
  unsup_col=[]
  for i in range(0,n_comp):
      col='PCA'+str(i+1)
      unsup_col.append(col)
  dim_reducer=fr[tech]
  reducer = dim_reducer(n_components=n_comp)
  fs_df = reducer.fit_transform(scaled_data)
  fs_df = pd.DataFrame(fs_df, columns=unsup_col)
  return fs_df,reducer

# Automating Encoding Function
def encode_categorical_columns(df):
    df_encoded = df.copy()  # Make a copy of the DataFrame
    label_encoders = {}
    one_hot_encoders = {}
    
    for column in df_encoded.select_dtypes(include=['object', 'category']).columns:
        unique_values = df_encoded[column].nunique()
        
        if unique_values == 2:
            # Label Encoding for binary columns
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            label_encoders[column] = le

        else:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            ohe_df = pd.DataFrame(ohe.fit_transform(df_encoded[[column]]), columns=ohe.get_feature_names_out([column]),
                                  index=df_encoded.index)
            df_encoded = df_encoded.drop(column, axis=1).join(ohe_df)
            one_hot_encoders[column] = ohe
    
    return df_encoded,label_encoders,one_hot_encoders

# Imputer Function
def imputer(encoded_data,Imputer,Imp_op):
    Imputer_op = Imp_op[Imputer]
    if Imputer_op is not "None":
        imputer_df = Imputer_op.fit_transform(encoded_data)
        imputer_df = pd.DataFrame(imputer_df,columns=encoded_data.columns)
    else:
        imputer_df = encoded_data.copy()

    return imputer_df

# Scaling and Normalization
scalers = {'No Scaling':"none",'Standard Scaling': StandardScaler(),'Normalization': Normalizer(),'Min Max Scaling': MinMaxScaler()}

def scale_dataframe(df,scaling_type):
    scaled_df = df.copy()
    if scaling_type == 'No Scaling':
        return scaled_df

    columns_to_scale = [col for col in scaled_df.columns if scaled_df[col].nunique() > 3]
    stype = scalers[scaling_type]

    if scaling_type in scalers:
        scaled_df[columns_to_scale] = stype.fit_transform(scaled_df[columns_to_scale])

    return scaled_df

#metrics_df = pd.DataFrame(columns=["Feature selection fn",'No of features', 'Imputer','Scaling Type','Model','Accuracy','AUC','Recall','Prec.','F1',	'Kappa'	,'MCC'	,'TT (Sec)'])

def evaluate_and_append_metrics(metrics_df,df,step):
    step = pd.DataFrame(step, index=[0])
    new_metrics_df = pd.concat([step,df],axis=1,join='outer')
    # Append the new metrics to the existing DataFrame using pd.concat
    metrics_df = pd.concat([metrics_df, new_metrics_df], axis=0)

    # Display the updated DataFrame
    return metrics_df

if choice == "Model Building":

    c3,c4=st.columns([3,2])
    with c3:
        st.title("Data preprocessing")
    with c4:
        st.caption("")
        st.caption("")
        with st.popover("View Original DataFrame",use_container_width=True):
            st.dataframe(data)
            st.text(f'Shape: {data.shape}')
    st.subheader("Target column")
    with st.container(border=True):
        col1, col2= st.columns([3,2])
    
        with col1:
            target = st.selectbox("__Select the Target column__", df.columns,index=None,placeholder="None (Clustering)")
            if target in data.select_dtypes(include=['object',"category"]):
                label = LabelEncoder()
                data[target]=label.fit_transform(data[target])
        ml_typ = ml_type(data,target)
        with col2:
            st.caption("")
            st.info(f' {ml_typ} Problem')

    
    if ml_typ == 'Classification' or ml_typ == 'Regression':
        if ml_typ == 'Regression':
            select_scores = {"No feature selection":"No selection","Chi Square": chi2,"ANOVA F-value regression": f_regression,"Information Gain Regression": mutual_info_regression,
                            "Pearson correlation coefficient":r_regression,"Custom Features":"Custom Features"}

        elif ml_typ == 'Classification':
            select_scores = {"No feature selection":"No selection","Chi Square": chi2,"ANOVA F-value classification": f_classif,"Information Gain Classification": mutual_info_classif,
                             "Custom Features":"Custom Features"}

        st.subheader("Feature Selection")
        with st.container(border=True):
            c1, c2 = st.columns([3,2])
            with c1:
                score_func_name = st.selectbox("Select the scoring function",list(select_scores.keys()))
                k = st.select_slider("Select the number of features", options=list(range(1, len(df.columns))))

            with c2:
                if select_scores[score_func_name] is not "No selection":
                    selected_list = select_k_best_features(data,target, k, select_scores,score_func_name)
                    options = st.multiselect("__Add or Remove selected columns__",data.columns,selected_list)
                    options.append(target)

                else:
                    options = data.columns
                    st.info("If no Feature selection is used the Data will remain as the Original Dataframe")
        
        st.subheader("Encoding")
        with st.container(border=True):
            c5,c6 = st.columns([3,2])
            with c5:
                df_fs=data[options]
                df_encoded,le,ohe = encode_categorical_columns(df_fs)
                st.success("Automatically encoded __Binary and non Binary Categorical Columns__")
            with c6:
                with st.popover("View Encoded DataFrame",use_container_width=True):
                    st.dataframe(df_encoded)
                    st.text(f'Shape: {df_encoded.shape}')

        c7,c8=st.columns(2)
        with c7:
            st.subheader("Missing value imputation")
        with c8:
            st.subheader("Feature Scaling")
        
        with st.container(border=True):
            c9,c10=st.columns(2)
            with c9:
                imp_option = st.selectbox("Select the Missing value imputation technique",list(Imp_op.keys()))
                if imp_option is not "No Imputation":
                    df_imputed = imputer(df_encoded,imp_option,Imp_op)
                    
                else:
                    df_imputed = df_encoded
            with c10:
                scaler_option = st.selectbox("Select the Feature scaling technique",list(scalers.keys()))
                if scaler_option is not "No Scaling":        
                    df_scaled = scale_dataframe(df_imputed, scaler_option)
                
                else:
                    df_scaled = df_imputed.copy()

        with st.expander("View Final DataFrame"):
        #with st.popover("View Final DataFrame",use_container_width=True):
            st.dataframe(df_scaled)
            st.text(f'Shape: {df_scaled.shape}')

        options_copy = list(options)
        options_copy.remove(target)
        dep_col=",".join(options_copy)
        steps = {"Feature selection fn":score_func_name,'No of features': k, 'Imputer': imp_option,'Scaling Type':scaler_option,'Target column':target,'Features':dep_col}


        st.title("Automated Model Building")
        st.subheader(f'{ml_typ} Model Building',divider='rainbow')
        model=ml[ml_typ]

        if model == cl:
            columns=["Feature selection fn",'No of features', 'Imputer','Scaling Type','Model','Accuracy',
                                        'AUC','Recall','Prec.','F1',	'Kappa'	,'MCC'	,'TT (Sec)','Target column','Features','Tuned Model']
        elif model == reg:
            columns=["Feature selection fn",'No of features', 'Imputer','Scaling Type','Model','MAE',
                                        'MSE','RMSE','R2','RMSLE','MAPE','TT (Sec)','Target column','Features','Tuned Model']

        if 'metrics_df' not in st.session_state:
            st.session_state.metrics_df = pd.DataFrame(columns=columns)
        
        if st.button("Run Modelling",type="primary"):
            st.info('The process might take a few minutes, progress is displayed top right corner of the screen.')
            steps_df = pd.DataFrame(steps, index=[0])
            model.setup(df_scaled, target=target)
            setup_df = model.pull()
            st.dataframe(setup_df)
            best_model = model.compare_models()
            compare_df = model.pull()
            model.save_model(best_model, 'best_model')
            tuned_model = model.tune_model(best_model)
            best_info = pd.DataFrame(compare_df.iloc[0]).T.reset_index()

            st.session_state.metrics_df = evaluate_and_append_metrics(st.session_state.metrics_df,best_info,steps)
            st.dataframe(compare_df)
            with st.popover("Experiment Tracking",use_container_width=True):
                st.dataframe(st.session_state.metrics_df)
                st.session_state.metrics_df.to_csv('steps.csv', index=False)

            
            st.subheader("Best model")
            st.text_area("Best Model",best_model)
            st.text_area("Hyperparameter Tuned Model",tuned_model)
            st.header('Download the Model')
            st.subheader("Tuned Model")
            st.info('The model downloadable is the one with the highest corresponding KPIs.')
            with open("best_model.pkl", 'rb') as f:
                st.download_button("Download Tuned Model", f, "best_model_test.pkl")

    elif ml_typ == 'Clustering':
        model=ml[ml_typ]
        select_scores = {"T-distributed Stochastic Neighbor Embedding":TSNE,"Kernel Principal component analysis":KernelPCA,
                            "Principal component analysi":PCA,"Mini-batch Sparse Principal Components Analysis":MiniBatchSparsePCA,}

        st.subheader("Feature Selection")
        with st.container(border=True):
            c1, c2 = st.columns([3,2])
            with c1:
                score_func_name = st.selectbox("Select the Feature Selection Method",["No Feature Selection","Variance Threshold","Custom Features"])
                if score_func_name == "Variance Threshold":
                    threshold = st.number_input("Select the threshold value")

            with c2:
                if score_func_name == "Variance Threshold":
                    selector = VarianceThreshold(threshold=threshold)
                    df1 = df.copy()
                    le = LabelEncoder()
                    for i in df1.columns:
                        df1[i] = le.fit_transform(df[i])
                    selected_features = selector.fit_transform(df1)

                    # Get the mask of selected features
                    feature_mask = selector.get_support()

                    selected_columns = df1.columns[feature_mask]
                    options = st.multiselect("__Add or Remove selected columns__",data.columns,selected_columns)
                
                elif score_func_name == "Custom Features":
                    options = st.multiselect("__Add or Remove selected columns__",data.columns)
                else:
                    options = df.columns
                    st.info("If no Feature selection is used the Data will remain as the Original Dataframe")

        st.subheader("Encoding")
        with st.container(border=True):
            c5,c6 = st.columns([3,2])
            with c5:
                df_fs=data[options]
                df_encoded,le,ohe = encode_categorical_columns(df_fs)
                st.success("Automatically encoded __Binary and non Binary Categorical Columns__")
            with c6:
                with st.popover("View Encoded DataFrame",use_container_width=True):
                    st.dataframe(df_encoded)
                    st.text(f'Shape: {df_encoded.shape}')

        c7,c8=st.columns(2)
        with c7:
            st.subheader("Missing value imputation")
        with c8:
            st.subheader("Feature Scaling")

        with st.container(border=True):
            c9,c10=st.columns(2)
            with c9:
                imp_option = st.selectbox("Select the Missing value imputation technique",list(Imp_op.keys()))
                if imp_option is not "No Imputation":
                    df_imputed = imputer(df_encoded,imp_option,Imp_op)
                    
                else:
                    df_imputed = df_encoded
            with c10:
                scaler_option = st.selectbox("Select the Feature scaling technique",list(scalers.keys()))
                if scaler_option is not "No Scaling":        
                    df_scaled = scale_dataframe(df_imputed, scaler_option)
                else:
                    df_scaled = df_imputed.copy()

        st.subheader("Feature Reduction")
        with st.container(border=True):
            c1, c2 = st.columns([3,2])
            with c1:
                score_func_name = st.selectbox("Select the Feature Reduction Method",list(select_scores.keys()),index=None,placeholder="No Feature Reduction")
                n_comp = st.select_slider("Select the number of features", options=list(range(1, 5)))
        
            with c2:
                if score_func_name is not None:
                    df_fr = feature_reduction(df_scaled,n_comp,select_scores,score_func_name)
                    df_fr = pd.DataFrame(df_fr)
                else:
                    df_fr = df_scaled.copy()
                    st.error("__Attention:__ If no Feature selection is used the Data will remain as the Original Dataframe")

                with st.popover("View Final DataFrame",use_container_width=True):
                    st.dataframe(df_fr)
                    st.text(f'Shape: {df_fr.shape}')

        st.title("Automated Model Building")
        st.subheader(f'{ml_typ} Model Building',divider='rainbow')
        empty = []
        algo = ['kmeans', 'ap', 'sc', 'dbscan', 'hclust']
        clust = st.selectbox("Choose the Clustering Algorithm", algo)
        st.info('There is no automated best model selection currently unavailable for Clustering algorithms, please select the model having the __highest silhouette score__.')  
        if st.button("Run Modelling",type="primary"):
            st.info('The process might take a few minutes, progress is displayed top right corner of the screen.')
            model.setup(df_scaled)
            setup_df = model.pull()
            st.dataframe(setup_df)
            model = model.create_model(clust)
            best_model = model
            st.dataframe(best_model)
            st.subheader("Best model")
            st.text_area("",best_model)


if choice == "New Data Prediction":
    st.title('Import your Test dataset')
    test = st.file_uploader("__Note:__ The test dataset should NOT contain the __target column__")

    dict_steps = dict(steps_df.iloc[-1])

    features_1 = list(dict_steps["Features"].split(","))
    k = dict_steps['No of features']
    score_func = dict_steps['Feature selection fn']
    Imp_fn = dict_steps['Imputer']
    Scal_fn = dict_steps['Scaling Type']
    target = dict_steps['Target column']

    if test:
        test = pd.read_csv(test, index_col=None)
        test.to_csv("predicted.csv", index=False)
        with st.expander("View Original Test DataFrame"):
            st.dataframe(test)
            st.text(f'Shape: {test.shape}')

        fs_test = test[features_1]
        en_test,ohe1,le1 = encode_categorical_columns(fs_test)

        if Imp_fn != "No Imputation":
            imp_test = imputer(en_test,Imp_fn,Imp_op)

        if Scal_fn != 'No Scaling':
            sc_test = scale_dataframe(imp_test,Scal_fn)
        else:
            sc_test = en_test

        with st.expander("View Preprocessed Test DataFrame"):
            st.dataframe(sc_test)
            st.text(f'Shape: {sc_test.shape}')

        with open("best_model.pkl", 'rb') as f:
            clf = joblib.load(f)
        
        y_pred = pd.DataFrame(clf.predict(sc_test))
        test[target] = y_pred
        if st.button("Predict",type='primary'):
            st.subheader("The Predicted Dataframe")
            st.dataframe(test)
            st.text(f'Shape: {test.shape}')
            st.download_button("Download Dataset", test.to_csv(), "predicted.csv")

if choice == "Code Generation":
    st.title("Code Generation")
    st.subheader("Click the button below to generate the code",divider='rainbow')
    st.info("The Code for the final experiment in Model Building will be Generated")
    dict_steps = dict(steps_df.iloc[-1])

    steps=["generate code for importing pandas, numpy along with everything from sklearn libraries"]

    features_1 = list(dict_steps["Features"].split(","))
    k = dict_steps['No of features']
    score_func = dict_steps['Feature selection fn']
    Imp_fn = dict_steps['Imputer']
    Scal_fn = dict_steps['Scaling Type']

    if score_func == 'No feature selection':
        steps = steps
    elif score_func == "Custom Features":
        steps.append(f"generate code for creating a Datframe by only using {features_1} columns of {data}")
    else:
        steps.append(f"generate code for importing and using kselectbest with score function as {score_func} for selecting {k} top features of {data}")

    steps.append(f"generate code for importing and using OneHotEncoding for catergorical column having number of unique value greater than two in {data} and importing and using LabelEncoding for catergorical column having two unique catergories in {data}")
    
    if Imp_fn != "No Imputation":
        steps.append(f"generate code for importing and using {Imp_fn} to impute null values in{data}")

    if Scal_fn != 'No Scaling':
        steps.append(f"generate code for importing and using {Scal_fn} to scale numerical columns that are not binary numerical columns in {data}")
    
    os.environ["GOOGLE_API"] = genai_api_key
    # Access your API key as an environment variable.
    genai.configure(api_key=os.environ["GOOGLE_API"])
    # Choose a model that's appropriate for your use case.
    model = genai.GenerativeModel('gemini-pro')
    if st.button("Generate Code",type="primary"):
        for step in steps:
            response = model.generate_content(step, stream=False)
            response.text
            st.caption("")
            st.caption("")
