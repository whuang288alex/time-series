import torch
import torch.nn as nn
import datetime
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from utils import MyLSTM


# load the model
in_window, out_window = 50, 50
model_type = "Linear"
if model_type == "LSTM":
    model = MyLSTM(in_dim=1, hidden_size=128, out_size=out_window, num_layers=3)
elif model_type == "Linear":
    model = nn.Sequential(
        nn.Linear(in_window, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, out_window)
    )
else:
    raise ValueError("Invalid model type!")   
model.load_state_dict(torch.load(f'./ckpt/{model_type}.pt'))
model = model.double()


# load the data
original_data = np.load('data/time_series_data.npy')
time_series_data = np.zeros(365*2, dtype=np.float64)
time_series_data[:in_window] = original_data[:in_window]


# run the inference
model.eval()
for index in range(in_window, 365*2, out_window):
    if model_type == "LSTM":
        data = torch.tensor(time_series_data[index-in_window: index]).to(torch.float64).view(1, -1, 1)
    elif model_type == "Linear":
        data = torch.tensor(time_series_data[index-in_window: index]).to(torch.float64).view(1, -1)
    with torch.no_grad():
        predicted = model(data)
        if index + out_window > 365*2:
            time_series_data[index:] = predicted[0].numpy()[:365*2-index]
        else:
            time_series_data[index : index + out_window] = predicted[0].numpy()

# denormalize the data
original_mean = np.load("./data/mean.npy")
original_std = np.load("./data/std.npy")
time_series_data = time_series_data * original_std + original_mean

# transform daily data to monthly data
month_day_count = np.load("./data/month_day_count.npy").tolist()
month_day_count += month_day_count
predictions = []
total_count = 0
for index in range(24):
    predictions.append(sum(time_series_data[total_count: total_count + month_day_count[index]]))
    total_count += month_day_count[index]
predictions = np.array(predictions) / 1000000  

st.title("Receipt Count Predictor")

# Input form for month
month = st.number_input("Enter month (1-12)", min_value=1, max_value=12, value=datetime.datetime.now().month)
month += 12
month_name = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
st.markdown(
    f'<p style="font-size:24px; font-family: Arial, sans-serif;">'
    f'{month_name[month-1]}, 2022: <strong>{predictions[month-1]:.2f}</strong>'
    '</p>',
    unsafe_allow_html=True
)

# Create a trace for the line
trace = go.Scatter(
    x=list(range(1, 25)),
    y=predictions,
    mode='lines',
    name='Predicted Receipt Count (Million)'
)

# Create a trace for the points
points_trace = go.Scatter(
    x=[month],
    y=[predictions[month - 1]],
    mode='markers',
    marker=dict(size=10, color='blue'),
    name='Point'
)

# Create a layout with a hovermode set to 'closest' to show tooltips
layout = go.Layout(
    xaxis=dict(title='Month (starting from January 2021)'),
    yaxis=dict(title='Predicted Receipt Count (Million)'),
    hovermode='closest'
)

# Create the figure with both traces and the layout
fig = go.Figure(data=[trace, points_trace], layout=layout)
st.plotly_chart(fig)
