import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def auto_eda(data):
    st.subheader("Summary Statistics")
    st.write(data.describe())
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Column Distribution")
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        selected_column = st.selectbox("Select column for histogram", numeric_columns)
        fig1 = px.histogram(data, x=selected_column, color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Scatter Plot between Two Columns")
        scatter_x = st.selectbox("Select X-axis column", numeric_columns)
        scatter_y = st.selectbox("Select Y-axis column", numeric_columns, index=1)
        fig2 = px.scatter(data, x=scatter_x, y=scatter_y, color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig2, use_container_width=True)

    # Second row of visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Box Plot")
        selected_column_box = st.selectbox("Select column for box plot", numeric_columns)
        fig3 = px.box(data, y=selected_column_box, color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.subheader("Correlation Matrix")
        d=data.drop(columns=["brand","processor_brand","processor_tier","primary_storage_type","secondary_storage_type","gpu_brand","gpu_type","OS","year_of_warranty"])
        corr_matrix = d.corr()
        fig4 = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                         x=corr_matrix.columns,
                                         y=corr_matrix.columns,
                                         colorscale='Viridis'))
        st.plotly_chart(fig4, use_container_width=True)

def train(data):
    model=st.selectbox("Choose the Algorithm:",["Linear Regression","Decision Tree","Random Forest"])
    per=st.slider("Choose the percentage of data for Training:",min_value=10,max_value=100)
    y=data['Price']
    X=data.drop(['Price'], axis=1)
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
    if st.button("Train"):   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-per)/100,random_state=42)
        if model == "Linear Regression":
            model=LinearRegression()
        elif model == "Decision Tree":
            model=DecisionTreeRegressor()
        elif model == "Random Forest":
            model=RandomForestRegressor()
        model.fit(X_train,y_train)
        y_pred= model.predict(X_test)
        r2=r2_score(y_test,y_pred)
        mae=mean_absolute_error(y_test,y_pred)
        st.write(f"R2 score: {r2:.2f}")
        st.write(f"Mean Absolute Error: {mae:,.2f}")
        print(mean_absolute_error(y_test,y_pred))
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
        st.success("Model trained successfully!")

def predict():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        df=data.drop(columns=['Price'])
        st.success("Model loaded successfully.")
        input_data={}
        
        for i in df.columns:
            a=st.selectbox(f"{i}",sorted(df[f"{i}"].unique()))
            input_data[i]=a

        if st.button('Predict Price'):
            label_encoders = {}
            for column in df.columns:
                if df[column].dtype == 'object':
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    label_encoders[column] = le
            
            input_df = pd.DataFrame([input_data])
            for column in label_encoders:
                input_df[column] = label_encoders[column].transform(input_df[column])

            prediction = model.predict(input_df)[0]
            st.subheader(f"Laptop Price: ₹{prediction:,.2f}")
        
    except FileNotFoundError:
        st.error("The model has not been trained yet. Please train the model in the 'Training & Evaluation' tab.")


st.title("Laptop Price Prediction")
data=pd.read_csv("laptops.csv")
data.drop(columns=["index","Model","Rating","num_threads"],inplace=True)

tab1,tab2,tab3,tab4=st.tabs(["Home","Exploratory Data Analysis","Training & Evaluation","Predict"])

with tab1:
    st.subheader("What is this?")
    st.markdown("This project aims to predict the price of laptops based on various features and specifications. Whether you're a tech enthusiast, a data scientist, or someone interested in machine learning, this project provides an opportunity to explore the world of predictive modelling.")
    st.subheader("How to use it?")
    st.markdown("**Exploratory Data Analysis** - Explore the dataset here")
    st.markdown("**Taining & Evaluation** - Train and evaluate the model tab to create the model.pkl file")
    st.markdown("**Predict** - Based on the model.pkl file created in 'Taining & Evaluation' tab make predictions")
    st.subheader("Dataset Overview:")
    st.write(data.sample(3))


with tab2:
    auto_eda(data)

with tab3:
    train(data)

with tab4:
    predict()
