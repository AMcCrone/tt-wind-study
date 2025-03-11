import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata, interp1d

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

# Function to resample points along a contour line based on cumulative distance.
def resample_contour_line(x, y, n_points=1000):
    # Stack x and y into an array of points.
    pts = np.column_stack((x, y))
    # Compute Euclidean distances between consecutive points.
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    # Cumulative distance along the line.
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    # Create new equally spaced distances along the line.
    new_dist = np.linspace(0, cum_dist[-1], n_points)
    # Create interpolation functions for x and y over the cumulative distance.
    fx = interp1d(cum_dist, x, kind='linear')
    fy = interp1d(cum_dist, y, kind='linear')
    return fx(new_dist), fy(new_dist)

# For each unique contour level, resample the line to generate additional points.
unique_z = sorted(df['Z'].unique())
resampled_points = []
for z_val in unique_z:
    # Extract points for the current contour.
    group = df[df['Z'] == z_val]
    # For robustness, sort the points by their cumulative distance.
    # Here we assume the points are roughly in order;
    # if not, additional ordering logic may be required.
    x_vals = group['X'].values
    y_vals = group['Y'].values
    new_x, new_y = resample_contour_line(x_vals, y_vals, n_points=200)
    temp_df = pd.DataFrame({'X': new_x, 'Y': new_y, 'Z': z_val})
    resampled_points.append(temp_df)
    
resampled_df = pd.concat(resampled_points, ignore_index=True)

# Use the resampled points for interpolation.
interp_points = resampled_df[['X', 'Y']].values
interp_values = resampled_df['Z'].values

# Create a grid covering the x and y ranges on a logarithmic scale.
xi = np.logspace(np.log10(0.1), np.log10(100), 300)
yi = np.logspace(np.log10(2), np.log10(200), 300)
Xgrid, Ygrid = np.meshgrid(xi, yi)

# Interpolate the contour values onto the grid.
Zgrid = griddata(interp_points, interp_values, (Xgrid, Ygrid), method='linear')

# Create a Plotly contour plot showing the gradient.
fig = go.Figure(data=go.Contour(
    x=xi,
    y=yi,
    z=Zgrid,
    colorscale='Viridis',
    contours=dict(
        showlines=False  # We overlay raw contour lines below.
    ),
    colorbar=dict(title="Contour Level")
))

# Overlay the raw (resampled) contour lines.
for z_val in unique_z:
    group = resampled_df[resampled_df['Z'] == z_val]
    fig.add_trace(go.Scatter(
        x=group['X'],
        y=group['Y'],
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False
    ))

# Create interactive query inputs for (x, y).
x_query = st.number_input("Enter x coordinate (0.1 to 100):", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f")
y_query = st.number_input("Enter y coordinate (2 to 200):", min_value=2.0, max_value=200.0, value=50.0, step=0.1, format="%.2f")

# Interpolate the contour value at the query point.
query_point = np.array([[x_query, y_query]])
interp_val = griddata(interp_points, interp_values, query_point, method='linear')[0]

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

# Add a marker at the query point with the interpolated contour value.
fig.add_trace(go.Scatter(
    x=[x_query],
    y=[y_query],
    mode='markers+text',
    marker=dict(color='red', size=10),
    text=[f"{interp_val:.3f}"],
    textposition="top center"
))

# Set axes to logarithmic scale and add titles.
fig.update_xaxes(type="log", title="X")
fig.update_yaxes(type="log", title="Y")
fig.update_layout(
    title="Contour Gradient Using Resampled Contour Lines",
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)
