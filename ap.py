import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
import joblib
# Load the pre-trained model
model = joblib.load('RFC.pkl')
df=pd.read_csv("C:\\Users\\saipr\\Downloads\\weather_classification_data (1).csv")

X=df.drop(['Cloud Cover','Season','Weather Type','Location'],axis=1)
y=df['Weather Type']


# Define a function to preprocess input data
def preprocess_input(data):
    # Encode categorical features
    # Create a DataFrame from the input data
    df = pd.DataFrame([data])
    
    # Apply one-hot encoding or other preprocessing as necessary
    df_encoded = pd.get_dummies(df, columns=['Cloud Cover', 'Season', 'Location'])
    
    # Ensure all columns are present
    missing_cols = set(X.columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded[X.columns]
    
    return df_encoded

# Define the Streamlit app
st.title('Weather Type Prediction')

# Input fields for user
temperature = st.number_input('Temperature', min_value=-50.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity', min_value=0, max_value=100, value=80)
wind_speed = st.number_input('Wind Speed', min_value=0.0, max_value=50.0, value=10.0)
precipitation = st.number_input('Precipitation (%)', min_value=0.0, max_value=100.0, value=70.0)
cloud_cover = st.selectbox('Cloud Cover', ['clear', 'partly cloudy', 'cloudy', 'overcast'])
atmospheric_pressure = st.number_input('Atmospheric Pressure', min_value=900.0, max_value=1100.0, value=1015.0)
uv_index = st.number_input('UV Index', min_value=0, max_value=11, value=3)
season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
visibility = st.number_input('Visibility (km)', min_value=0.0, max_value=100.0, value=8.0)
location = st.selectbox('Location', ['inland', 'coastal', 'mountain'])

# Create a dictionary with user input
input_data = {
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind Speed': wind_speed,
    'Precipitation (%)': precipitation,
    'Cloud Cover': cloud_cover,
    'Atmospheric Pressure': atmospheric_pressure,
    'UV Index': uv_index,
    'Season': season,
    'Visibility (km)': visibility,
    'Location': location
}

# Preprocess the input data
preprocessed_data = preprocess_input(input_data)

# Make the prediction
prediction = model.predict(preprocessed_data)
inverse_weather_map = {1: 'Rainy', 2: 'Cloudy', 3: 'Sunny', 4: 'Snowy'}

predicted_weather = inverse_weather_map[prediction[0]]

# Display the result
st.write(f'The predicted weather type for the given data point is: {predicted_weather}')
