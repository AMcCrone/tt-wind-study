import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

st.title("Contour Plot with Interactive Crosshairs")

# Load data from GitHub
data_url = "chart-data/NA.3-contour_data.xlsx"

@st.cache_data
def load_data(url):
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create sample data for demonstration if GitHub data cannot be loaded
        x_range = np.logspace(1, 3, 50)
        y_range = np.logspace(1, 3, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 0.75 + (X/1000) * (Y/1000)
        df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'z': Z.flatten()
        })
        return df

df = load_data(data_url)

# Create the grid for interpolation
# Determine the range of x and y in log space
x_min, x_max = df['x'].min(), df['x'].max()
y_min, y_max = df['y'].min(), df['y'].max()

# Create a dense grid for smooth interpolation
grid_density = 200
x_grid = np.logspace(np.log10(x_min), np.log10(x_max), grid_density)
y_grid = np.logspace(np.log10(y_min), np.log10(y_max), grid_density)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Use griddata for interpolation - this creates a smooth surface
points = np.column_stack((np.log10(df['x']), np.log10(df['y'])))
Z_grid = griddata(points, df['z'], (np.log10(X_grid), np.log10(Y_grid)), method='cubic')

# User inputs for crosshairs
col1, col2 = st.columns(2)
with col1:
    x_input = st.number_input("X Coordinate", min_value=float(x_min), max_value=float(x_max), value=float(np.sqrt(x_min*x_max)))
with col2:
    y_input = st.number_input("Y Coordinate", min_value=float(y_min), max_value=float(y_max), value=float(np.sqrt(y_min*y_max)))

# Interpolate the z value at the selected x,y coordinates
def get_interpolated_z(x, y):
    # Convert to log space for interpolation
    log_x, log_y = np.log10(x), np.log10(y)
    
    # Use griddata for a single point interpolation
    interp_z = griddata(points, df['z'], np.array([[log_x, log_y]]), method='cubic')[0]
    
    if np.isnan(interp_z):
        # If outside the convex hull, try with 'nearest' method
        interp_z = griddata(points, df['z'], np.array([[log_x, log_y]]), method='nearest')[0]
    
    return interp_z

# Get the interpolated z value
interpolated_z = get_interpolated_z(x_input, y_input)

# Format the z value for display
st.metric("Interpolated Value", f"{interpolated_z:.4f}")

# Create contour lines with specified levels
contour_levels = np.arange(0.75, 1.71, 0.05)  # from 0.75 to 1.70 with step 0.05

# Create the figure
fig = go.Figure()

# Add the contour plot with smooth color filling
fig.add_trace(go.Contour(
    x=x_grid,
    y=y_grid,
    z=Z_grid,
    contours=dict(
        coloring='fill',
        showlabels=True,
        start=0.75,
        end=1.70,
        size=0.05,
        labelfont=dict(size=10, color='white')
    ),
    colorscale='Viridis',
    line=dict(width=1),
    colorbar=dict(
        title='Contour Value',
        titlefont=dict(size=14),
        tickfont=dict(size=12),
        len=0.9,
    )
))

# Add contour lines to ensure they are visible
fig.add_trace(go.Contour(
    x=x_grid,
    y=y_grid,
    z=Z_grid,
    contours=dict(
        coloring='lines',
        showlabels=False,
        start=0.75,
        end=1.70,
        size=0.05
    ),
    showscale=False,
    line=dict(width=1.5, color='rgba(255,255,255,0.8)')
))

# Add crosshairs
fig.add_trace(go.Scatter(
    x=[x_input, x_input],
    y=[y_min, y_max],
    mode='lines',
    line=dict(color='red', width=1, dash='dash'),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[x_min, x_max],
    y=[y_input, y_input],
    mode='lines',
    line=dict(color='red', width=1, dash='dash'),
    showlegend=False
))

# Add marker at the intersection point
fig.add_trace(go.Scatter(
    x=[x_input],
    y=[y_input],
    mode='markers',
    marker=dict(color='red', size=10, symbol='x'),
    showlegend=False
))

# Update layout for log-log axes
fig.update_layout(
    xaxis=dict(
        type='log',
        title='X (log scale)',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)'
    ),
    yaxis=dict(
        type='log',
        title='Y (log scale)',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)'
    ),
    margin=dict(l=40, r=40, t=40, b=40),
    height=700,
    plot_bgcolor='rgba(240,240,240,0.95)'
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Additional information about the interpolation
st.write("""
### About the Contour Interpolation
- The contour plot uses cubic interpolation for smooth gradients between contour lines
- Contour lines are drawn at intervals of 0.05 from 0.75 to 1.70
- The crosshairs show the interpolated value at the specified (x,y) coordinates
- Enter new x,y coordinates above to update the crosshairs and interpolated value
""")

# Display a sample of the raw data
with st.expander("View Raw Data Sample"):
    st.dataframe(df.head(10))
