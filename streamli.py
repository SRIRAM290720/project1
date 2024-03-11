

import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy import stats
import pickle

# Assuming 'new_dfs' is loaded here
@st.experimental_singleton
def load_data():
    with open('new_dfs.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

new_dfs = load_data()

# Streamlit app
st.title('Spice Price Prediction')

# Extracting unique spices and states for the dropdown
spices = list(set([k[0] for k in new_dfs.keys()]))
selected_spice = st.selectbox('Select Spice', spices)

# Filter states based on selected spice
states = [k[1] for k in new_dfs.keys() if k[0] == selected_spice]
selected_state = st.selectbox('Select State', states)

# Select number of months for prediction
num_months = st.number_input('Select number of months for prediction', min_value=1, max_value=12, value=3)

# Button to trigger prediction
if st.button('Predict'):

    df = new_dfs[(selected_spice, selected_state)]

    df.sort_values('Month&Year', inplace=True)
    data = df['Prices'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    sequence_length = 5
    generator = TimeseriesGenerator(data_scaled, data_scaled, length=sequence_length, batch_size=1)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=100, verbose=0)

    # Prepare the last part of the data as input_sequence
    input_sequence = data_scaled[-sequence_length:]

    # Initialize empty lists to store predictions and confidence intervals
    predictions = []
    lower_bounds = []
    upper_bounds = []

    current_sequence = input_sequence.reshape((1, sequence_length, 1))
    
    for _ in range(num_months):
        predicted_value = model.predict(current_sequence)[0]
        predictions.append(predicted_value)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[:, -1, :] = predicted_value

    predictions = scaler.inverse_transform(predictions).flatten()

    # Assuming errors are normally distributed, calculate mean and std deviation of errors
    predicted_all = model.predict(generator).flatten()
    true_values = data_scaled[sequence_length:].flatten()
    errors = true_values - predicted_all
    error_std = np.std(errors)

    # Confidence intervals (assuming errors are normally distributed)
    confidence_level =  2.576  # 99% confidence
    lower_bounds = predictions - (confidence_level * error_std)
    upper_bounds = predictions + (confidence_level * error_std)

    last_date = df['Month&Year'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]

    # Displaying results with confidence intervals
    results = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predictions,
        'Lower Bound': lower_bounds,
        'Upper Bound': upper_bounds
    })
    st.write(results)
