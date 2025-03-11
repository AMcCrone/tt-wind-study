import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

PASSWORD = st.secrets["password"]

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def check_password():
    """Check the password input against the secret password."""
    if st.session_state.get("password_input") == PASSWORD:
        st.session_state["authenticated"] = True
    else:
        st.error("Incorrect password.")

# If the user is not authenticated, show the password input and halt the app.
if not st.session_state["authenticated"]:
    st.text_input("Enter Password:", type="password", key="password_input", on_change=check_password)
    st.stop()

st.title("Contour Plot with Interpolation and Crosshairs")

# Read the CSV file from the local path.
# Make sure the file "chart-data/NA.3-contour_data.csv" is in your repository.
df = pd.read_excel("NA.3-contour_data.xlsx")

# Display a preview of the data
st.write("Data preview:", df.head())

# The CSV file is assumed to have columns: "X", "Y", "Z"
# Where:
#   - "X" is the x-coordinate,
#   - "Y" is the y-coordinate,
#   - "Z" is the contour level (ranging from 0.75 to 1.70).

# Create a grid covering the x and y ranges (logarithmic axes):
# x: from 0.1 to 100, y: from 2 to 200.
xi = np.logspace(np.log10(0.1), np.log10(100), 200)
yi = np.logspace(np.log10(2), np.log10(200), 200)
Xgrid, Ygrid = np.meshgrid(xi, yi)

# Interpolate the scattered contour data onto the grid.
points = df[['X', 'Y']].values
values = df['Z'].values
Zgrid = griddata(points, values, (Xgrid, Ygrid), method='linear')

# Create a Plotly contour plot with smooth gradient fill.
fig = go.Figure(data=go.Contour(
    x=xi,
    y=yi,
    z=Zgrid,
    colorscale='Viridis',
    contours=dict(
        showlines=True,
        start=df['Z'].min(),
        end=df['Z'].max(),
        size=0.1  # Adjust the contour interval as needed.
    ),
    colorbar=dict(title="Contour Level")
))

# Overlay raw contour lines from the CSV:
# For each unique Z value, group the data and join the points into a line.
unique_z = sorted(df['Z'].unique())
for z_val in unique_z:
    # Get the points corresponding to this contour.
    group = df[df['Z'] == z_val]
    # Optionally, sort the points. Here we assume sorting by X produces the correct order.
    group_sorted = group.sort_values(by='X')
    fig.add_trace(go.Scatter(
        x=group_sorted['X'],
        y=group_sorted['Y'],
        mode='lines',
        name=f"Z = {z_val}",
        line=dict(color='black', width=2),
        showlegend=False  # Set to True if you wish to display legends.
    ))

# Create Streamlit number inputs for the query (x, y) coordinate.
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

# Set the axes to logarithmic scale and add axis titles.
fig.update_xaxes(type="log", title="X")
fig.update_yaxes(type="log", title="Y")
fig.update_layout(
    title="Contour Plot with Raw Contour Lines, Gradient, Interpolation, and Crosshairs",
    hovermode="closest"
)

# Display the Plotly figure in the Streamlit app.
st.plotly_chart(fig, use_container_width=True)
