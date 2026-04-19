import pandas as pd
import numpy as np 
import pickle
import streamlit as st

loaded_model = pickle.load(open(r'C:\Users\Raghavendra G N\python_practice\Logistic_Regression.sav','rb'))

st.title('A Logistic Regression Model')
st.write('Titanic survival prediction')
st.write('Enter passenger details to predict if they would survive or not')

#creating function

def get_input():

    #input fields

    pclass = st.selectbox('Passenger class (pclass)',[1,2,3])
    sex = st.selectbox('sex',['male','female'])
    age = st.number_input('age',min_value=0,max_value=100,value=25)
    sibsp = st.number_input('number of siblings/spouses (sibsp)',min_value=0,max_value=10,value=0)
    parch = st.number_input('number of parents/children (parch)',min_value=0,max_value=10,value=0)
    fare = st.number_input('fare',min_value=0.0,max_value=600.0,value=32.0)
    embarked = st.selectbox('embarked port',['C = Cherbourg (1)','Q = Queenstown (2)','S = Southampton (0)'])   

     #encode categorical variables

    sex = 1 if sex == 'female' else 0

    if 'Cherbourg' in embarked:
        embarked = 1
    elif 'Queenstown' in embarked:
        embarked = 2
    else:
        embarked = 0

    data_dict = {
        'Pclass' : pclass, 
        'Sex' : sex,
        'Age' : age,
        'SibSp' : sibsp,
        'Parch' : parch,
        'Fare' : fare,
        'Embarked' : embarked
    }

    #convert to dataframe

    input_data = pd.DataFrame(data_dict, index=[1])
    return input_data

features = get_input()

#predict on button click

if st.button('predict survival'):
    st.write('input features')
    st.write(features)

    res = loaded_model.predict(features)[0]
    proba = loaded_model.predict_proba(features)[0][1]

    st.write('prediction result')
    if res == 1:
        st.write(f'Survived! (probability : {proba:.2f})')
    else:
        st.write(f'did not survive (probability : {proba:.2f})')
