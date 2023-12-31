import torch
import torch.nn as nn
import datetime
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from utils import get_model


# # load the model
# model_type, in_window, out_window = "Linear", 50, 50
# model = get_model(model_type, in_window, out_window)
# model.load_state_dict(torch.load(f'./ckpt/{model_type}.pt'))
# model = model.to(torch.float64)


# # load the data
# original_data = np.load('data/time_series_data.npy')
# time_series_data = np.zeros(365*2, dtype=np.float64)
# time_series_data[:in_window] = original_data[:in_window]


# # run the inference
# model.eval()
# max_time =  365*2
# for index in range(in_window, max_time, out_window):
#     data = torch.tensor(time_series_data[index-in_window: index]).to(torch.float64).unsqueeze(0)
#     with torch.no_grad():
#         predicted = model(data).numpy()
#         if index + out_window > max_time:
#             time_series_data[index:] = predicted[0][:max_time -index] # zero to get rid of the batch dimension
#         else:
#             time_series_data[index : index + out_window] = predicted[0]


# # denormalize the data
# original_mean = np.load("./data/mean.npy")
# original_std = np.load("./data/std.npy")
# time_series_data = time_series_data * original_std + original_mean


# ## transform daily data to monthly data
# month_day_count = np.load("./data/month_day_count.npy").tolist()
# month_day_count += month_day_count
# predictions = []
# total_count = 0
# for index in range(24):
#     predictions.append(sum(time_series_data[total_count: total_count + month_day_count[index]]))
#     total_count += month_day_count[index]
# predictions = np.array(predictions) / 1000000 
# np.save("./data/predictions.npy", predictions) 


########## Streamlit App ##########
st.title("Receipt Count Predictor")
predictions = np.load("./data/predictions.npy")

# Input form for month, only allow 1-12 for 2022 January to 2022 December
month = st.number_input("Enter month (1-12)", min_value=1, max_value=12, value=1)
month += 12
month_name = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
st.markdown(
    f'<div style="text-align:center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;">'
    f'<p style="font-size:24px; font-family: Arial, sans-serif; margin: 0;">'
    f'{month_name[month-1]}, 2022</p>'
    f'<p style="font-size:32px; font-weight: bold; margin: 0;">'
    f'{predictions[month-1]:.2f} Million</p>'
    '</div>',
    unsafe_allow_html=True # so html will not be treated as text
)


trace = go.Scatter(
    x=[f"{month_name[i]}, 2021" if i < 12 else f"{month_name[i]}, 2022" for i in range(24)],
    y=predictions,
    mode='lines',
    name='Predicted Receipt Count (Million)'
)


points_trace = go.Scatter(
    x=[f"{month_name[month - 1]}, 2022"],  # Use the selected month name
    y=[predictions[month - 1]],
    mode='markers',
    marker=dict(size=10, color='blue'),
    name='Point'
)


layout = go.Layout(
    xaxis=dict(title='Month'),
    yaxis=dict(title='Predicted Receipt Count (Million)'),
    hovermode='closest'
)

# Create the figure with both traces and the layout
fig = go.Figure(data=[trace, points_trace], layout=layout)
st.plotly_chart(fig)
