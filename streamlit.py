import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
os.chdir('C:\\Users\\adima\\OneDrive\\Documents\\Cohort 128_ML_ Day 43\\Stremlit Deployment and PBI Dashboard\\')

st.set_page_config(layout='wide')

lr2=pickle.load(open('linear_model.pkl','rb'))
image_sidebar=Image.open('Pic 1.png')
st.sidebar.image(image_sidebar,use_container_width=True)
st.sidebar.header('Vehicle Features')

def get_user_input():
    horsepower=st.sidebar.number_input('HorsePower (No)',min_value=0,max_value=1000,step=1,value=300)
    torque=st.sidebar.number_input('torque (No)',min_value=0,max_value=1500,step=1,value=400)
    make=st.sidebar.selectbox('Make',['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size = st.sidebar.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style = st.sidebar.selectbox('Body Style', [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])

    userdata={
    'Horsepower_No':horsepower,
    'Torque_No':torque,
    f'Make_{make}':1,
    f'Body Size_{body_size}': 1, #When ever you select a value related category assigned to 1, Default all categories 0
        f'Body Style_{body_style}': 1,
        f'Engine Aspiration_{engine_aspiration}': 1,
        f'Drivetrain_{drivetrain}': 1,
        f'Transmission_{transmission}': 1,
    }
    return userdata

image_banner=Image.open('Pic 2.png')
st.image(image_banner,use_container_width=True)

st.markdown("<h1 style= 'text align:center;'>Vehicle Price Prediction App</h1>",unsafe_allow_html=True)
            
left_col, right_col = st.columns(2)

with left_col:
    st.header("Feature Details")
    
    user_data = get_user_input()
    st.write (user_data)

with right_col:
    st.header("Predict Vehicle Price")
    
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    features = [
        'Horsepower_No', 'Torque_No', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley',
        'Make_Ford', 'Make_Mercedes-Benz', 'Make_Nissan', 'Body Size_Compact', 'Body Size_Large',
        'Body Size_Midsize', 'Body Style_Cargo Minivan', 'Body Style_Cargo Van', 
        'Body Style_Convertible', 'Body Style_Convertible SUV', 'Body Style_Coupe', 
        'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van',
        'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
        'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated',
        'Engine Aspiration_Supercharged', 'Engine Aspiration_Turbocharged',
        'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged', 
        'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD', 
        'Transmission_automatic', 'Transmission_manual'
    ]

    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = lr2.predict(input_array)
        st.subheader("Predicted Price")
        st.write(f"${prediction[0]:,.2f}")

# streamlit run "C:/Noble/Training/Course Content/Stremlit Deployment and PBI Dashboard/Car_Price.py"
# Press CTRL +C to exist stremlit 
         
            
            
            
            
            
            
            
            
            
            
            
            
