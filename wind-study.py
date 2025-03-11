import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title("Contour Plot with Interpolation and Crosshairs")

# Read the CSV data from your GitHub repo.
# Replace 'your-username/your-repo' with your actual repository details.
NA3_csv = "chart-data/NA.3-contour_data.csv"
df = pd.read_csv(NA3_csv)

# Debug: print out the CSV columns to verify their names.
st.write("CSV Columns:", df.columns)

# Set the expected column names. Update these if your CSV uses different names.
x_column = 'x'          # Change this if your CSV has a different name, e.g., 'x_coord'
y_column = 'y'          # Change this if your CSV has a different name, e.g., 'y_coord'
dataset_column = 'dataset'  # This column holds the contour level value (ranging from 0.75 to 1.70)

# Extract the x, y, and dataset (contour level) values.
# If your CSV uses different column names, update the variables above accordingly.
points = df[[x_column, y_column]].values
values = df[dataset_column].values

# Create a grid covering the x and y ranges (logarithmic axes: x from 0.1 to 100, y from 2 to 200)
xi = np.logspace(np.log10(0.1), np.log10(100), 200)
yi = np.logspace(np.log10(2), np.log10(200), 200)
Xgrid, Ygrid = np.meshgrid(xi, yi)

# Interpolate the scattered contour data (from the CSV) onto the grid.
# For each point in the CSV, the value is the contour level ("dataset").
points = df[['x', 'y']].values
values = df['dataset'].values
Zgrid = griddata(points, values, (Xgrid, Ygrid), method='linear')

# Create a Plotly contour plot.
fig = go.Figure(data=go.Contour(
    x=xi,
    y=yi,
    z=Zgrid,
    colorscale='Viridis',  # smooth gradient color fill
    contours=dict(
        showlines=True,
        start=df['dataset'].min(),
        end=df['dataset'].max(),
        size=0.05  # contour interval matches your dataset spacing
    ),
    colorbar=dict(title="Contour Level")
))

# Create Streamlit number inputs for the x and y query coordinates.
x_query = st.number_input("Enter x coordinate (0.1 to 100):", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f")
y_query = st.number_input("Enter y coordinate (2 to 200):", min_value=2.0, max_value=200.0, value=50.0, step=0.1, format="%.2f")

# Interpolate the contour value at the (x_query, y_query) coordinate.
query_point = np.array([[x_query, y_query]])
interp_val = griddata(points, values, query_point, method='linear')[0]

# Add crosshairs at the query point (vertical and horizontal dashed lines).
fig.add_shape(
    type="line",
    x0=x_query, x1=x_query,
    y0=yi.min(), y1=yi.max(),
    line=dict(color="black", dash="dash")
)
fig.add_shape(
    type="line",
    x0=xi.min(), x1=xi.max(),
    y0=y_query, y1=y_query,
    line=dict(color="black", dash="dash")
)

# Add a marker at the query point with a label displaying the interpolated contour level.
fig.add_trace(go.Scatter(
    x=[x_query],
    y=[y_query],
    mode='markers+text',
    marker=dict(color='red', size=10),
    text=[f"{interp_val:.3f}"],
    textposition="top center"
))

# Set the axes to log scale and add axis titles.
fig.update_xaxes(type="log", title="X")
fig.update_yaxes(type="log", title="Y")
fig.update_layout(
    title="Contour Plot with Smoothed Gradient, Interpolated Value, and Crosshairs",
    hovermode="closest"
)

# Display the Plotly figure in the Streamlit app.
st.plotly_chart(fig, use_container_width=True)

