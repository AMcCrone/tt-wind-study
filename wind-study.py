import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title("Contour Plot with Interpolation and Crosshairs")

# Read the CSV data from your GitHub repo.
# Replace 'your-username/your-repo' with your actual repository details.
NA3_csv = "chart-data/NA.3-contour_data.csv"

# Debug: print out the CSV columns to verify their names.
# Fetch the CSV file content using requests.
response = requests.get(NA3_csv)
if response.status_code != 200:
    st.error("Failed to load CSV file")
    st.stop()
data_str = response.content.decode('utf-8')
lines = data_str.splitlines()

# Parse the CSV file.
# Each block in the file:
#   - First row: contour value (e.g. 0.75)
#   - Second row: header ("x,y")
#   - Following rows: data rows with x and y values, until an empty line or a new block starts.
data_list = []
i = 0
while i < len(lines):
    # Skip any empty lines.
    if lines[i].strip() == "":
        i += 1
        continue
    try:
        # First row: contour value.
        contour_val = float(lines[i].strip())
    except ValueError:
        st.error(f"Error parsing contour value on line {i+1}: {lines[i]}")
        st.stop()
    i += 1
    # Second row: header; we assume it has "x" and "y"
    if i < len(lines):
        header = lines[i].strip().split(',')
        i += 1
    # Process subsequent rows until a row without a comma is encountered (new block) or end-of-file.
    while i < len(lines) and ("," in lines[i]):
        parts = lines[i].strip().split(',')
        if len(parts) < 2:
            i += 1
            continue
        try:
            x_val = float(parts[0].strip())
            y_val = float(parts[1].strip())
            data_list.append((contour_val, x_val, y_val))
        except ValueError:
            # Skip rows that cannot be parsed.
            pass
        i += 1

# Convert the parsed data into a DataFrame with columns: 'dataset', 'x', 'y'.
df = pd.DataFrame(data_list, columns=["dataset", "x", "y"])
st.write("Parsed Data (first few rows):", df.head())

# Create a grid for interpolation.
# The x-axis is logarithmically spaced from 0.1 to 100, and the y-axis from 2 to 200.
xi = np.logspace(np.log10(0.1), np.log10(100), 200)
yi = np.logspace(np.log10(2), np.log10(200), 200)
Xgrid, Ygrid = np.meshgrid(xi, yi)

# Prepare the scattered data points.
points = df[['x', 'y']].values
values = df['dataset'].values

# Interpolate the contour (dataset) values onto the grid.
Zgrid = griddata(points, values, (Xgrid, Ygrid), method='linear')

# Create a Plotly contour plot with smooth gradient fill.
fig = go.Figure(data=go.Contour(
    x=xi,
    y=yi,
    z=Zgrid,
    colorscale='Viridis',  # Smooth gradient colors.
    contours=dict(
        showlines=True,
        start=df['dataset'].min(),
        end=df['dataset'].max(),
        size=0.05  # Use an interval matching your dataset spacing.
    ),
    colorbar=dict(title="Contour Level")
))

# Create Streamlit number inputs for the x and y query coordinates.
x_query = st.number_input("Enter x coordinate (0.1 to 100):", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f")
y_query = st.number_input("Enter y coordinate (2 to 200):", min_value=2.0, max_value=200.0, value=50.0, step=0.1, format="%.2f")

# Interpolate the contour value at the given (x_query, y_query) coordinate.
query_point = np.array([[x_query, y_query]])
interp_val = griddata(points, values, query_point, method='linear')[0]

# Add crosshairs at the query coordinate.
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

# Add a marker with a label at the query point showing the interpolated contour value.
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

