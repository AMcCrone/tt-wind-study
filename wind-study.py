import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
import io
import os

# Set page configuration
st.set_page_config(page_title="Contour Analysis Tool", layout="wide")

# -----------------------
# Authentication Section
# -----------------------
# Retrieve the password from secrets
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

# -----------------------
# Main Application
# -----------------------
st.title("Contour Analysis Tool")

# -----------------------
# Configuration for all plots based on the provided table
# -----------------------

# Define axis configuration and contour ranges for each plot
plot_configs = {
    "NA.3": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "distance (m)",
        "contour_start": 0.75, "contour_end": 1.7, "contour_step": 0.05
    },
    "NA.4": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "spread length (m)",
        "contour_start": 0.56, "contour_end": 1.0, "contour_step": 0.02
    },
    "NA.5": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "radius (m)",
        "contour_start": 0.07, "contour_end": 0.21, "contour_step": 0.01
    },
    "NA.6": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "parameter X (m)",
        "contour_start": 1.0, "contour_end": 1.8, "contour_step": 0.05
    },
    "NA.7": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "parameter Y (m)",
        "contour_start": 1.5, "contour_end": 4.2, "contour_step": 0.1
    },
    "NA.8": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "parameter Z (m)",
        "contour_start": 0.60, "contour_end": 1.0, "contour_step": 0.02
    }
}

# Common y-axis name
y_axis_name = "z-h_dis (m)"

# -----------------------
# Data Loading Function
# -----------------------
@st.cache_data
def load_data():
    """Load data for each sheet with appropriate X, Y, Z columns."""
    dataframes = {}
    
    for sheet_name, config in plot_configs.items():
        # Generate synthetic data that matches the expected contour ranges
        x_min, x_max = config["x_min"], config["x_max"]
        y_min, y_max = config["y_min"], config["y_max"]
        contour_start, contour_end = config["contour_start"], config["contour_end"]
        
        # For sheets with data
        if sheet_name in ["NA.3", "NA.4", "NA.5"]:
            # Create a grid of logarithmically spaced points
            num_x, num_y = 50, 50  # Adjust as needed for data density
            x_vals = np.logspace(np.log10(x_min), np.log10(x_max), num_x)
            y_vals = np.logspace(np.log10(y_min), np.log10(y_max), num_y)
            
            x_grid, y_grid = np.meshgrid(x_vals, y_vals)
            x_flat = x_grid.flatten()
            y_flat = y_grid.flatten()
            
            # Generate Z values appropriate for each sheet within their contour range
            if sheet_name == "NA.3":
                # Range: 0.75 to 1.7
                z_flat = contour_start + (contour_end - contour_start) * (
                    0.5 + 0.4 * np.sin(np.log10(x_flat) * 1.5) * np.cos(np.log10(y_flat) * 1.2)
                )
            elif sheet_name == "NA.4":
                # Range: 0.56 to 1.0
                z_flat = contour_start + (contour_end - contour_start) * (
                    0.5 + 0.4 * np.sin(np.log10(x_flat) * 2.0) * np.cos(np.log10(y_flat) * 1.3)
                )
            elif sheet_name == "NA.5":
                # Range: 0.07 to 0.21
                z_flat = contour_start + (contour_end - contour_start) * (
                    0.5 + 0.4 * np.sin(np.log10(x_flat) * 0.8) * np.cos(np.log10(y_flat) * 1.0)
                )
            
            # Ensure values stay within the contour range
            z_flat = np.clip(z_flat, contour_start, contour_end)
            
            # Create dataframe with X, Y, Z columns
            dataframes[sheet_name] = pd.DataFrame({
                'x': x_flat,
                'y': y_flat,
                'z': z_flat
            })
        else:
            # Empty dataframe for sheets without data
            dataframes[sheet_name] = pd.DataFrame({
                'x': [],
                'y': [],
                'z': []
            })
    
    return dataframes

# -----------------------
# Contour Plot Function
# -----------------------
def create_contour_plot(df, sheet_name, x_input, y_input):
    """Create a contour plot with interactive crosshairs for a specific sheet."""
    config = plot_configs[sheet_name]
    x_min, x_max = config["x_min"], config["x_max"]
    y_min, y_max = config["y_min"], config["y_max"]
    x_name = config["x_name"]
    contour_start = config["contour_start"]
    contour_end = config["contour_end"] 
    contour_step = config["contour_step"]
    
    # Check if dataframe is empty
    if df.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title=f"{sheet_name}: No Data Available Yet",
            annotations=[
                dict(
                    text="This plot does not contain any data yet.",
                    showarrow=False,
                    font=dict(size=16),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ],
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig, None
    
    # Create a dense grid for smooth interpolation
    grid_density = 200
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), grid_density)
    y_grid = np.logspace(np.log10(y_min), np.log10(y_max), grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Use griddata for interpolation - creates a smooth surface
    points = np.column_stack((np.log10(df['x']), np.log10(df['y'])))
    Z_grid = griddata(points, df['z'], (np.log10(X_grid), np.log10(Y_grid)), method='cubic')
    
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
    
    # Get the interpolated z value if x and y are within range
    interpolated_z = None
    if x_min <= x_input <= x_max and y_min <= y_input <= y_max:
        interpolated_z = get_interpolated_z(x_input, y_input)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the contour plot with higher opacity for the color filling
    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z_grid,
        contours=dict(
            coloring='fill',
            showlabels=True,
            start=contour_start,
            end=contour_end,
            size=contour_step,
            labelfont=dict(size=10, color='white')
        ),
        colorscale='Viridis',
        opacity=0.85,
        line=dict(width=0.5),
        colorbar=dict(
            title='Contour Value', 
            thickness=20, 
            len=0.9,
            tickformat='.2f' if contour_step >= 0.05 else '.3f'
        )
    ))
    
    # Add black contour lines
    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z_grid,
        contours=dict(
            coloring='lines',
            showlabels=False,
            start=contour_start,
            end=contour_end,
            size=contour_step
        ),
        showscale=False,
        line=dict(width=1.5, color='black')  # Explicitly set black contour lines
    ))
    
    # Add crosshairs if x and y are within range
    if x_min <= x_input <= x_max and y_min <= y_input <= y_max:
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
    
    # Update layout for log-log axes with specific ranges
    fig.update_layout(
        title=f"{sheet_name}: Contour Plot",
        xaxis=dict(
            type='log',
            title=x_name,
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            range=[np.log10(x_min), np.log10(x_max)]
        ),
        yaxis=dict(
            type='log',
            title=y_axis_name,
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            range=[np.log10(y_min), np.log10(y_max)]
        ),
        height=400,
        plot_bgcolor='rgba(240,240,240,0.95)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig, interpolated_z

# -----------------------
# Main Application Logic
# -----------------------

# Load all data
datasets = load_data()

# Global y-coordinate selection (since y-axis is the same for all plots)
y_input = st.slider(
    f"Y-Coordinate ({y_axis_name})", 
    min_value=2.0, 
    max_value=200.0, 
    value=50.0,
    format="%.1f"
)

# Container to display all plots
plot_container = st.container()

# Create all plots in a single page
with plot_container:
    # Display each plot in its own section
    for i, sheet_name in enumerate(["NA.3", "NA.4", "NA.5", "NA.6", "NA.7", "NA.8"]):
        st.markdown(f"### {sheet_name}")
        
        # Get configuration for this plot
        config = plot_configs[sheet_name]
        x_min, x_max = config["x_min"], config["x_max"]
        x_name = config["x_name"]
        contour_start = config["contour_start"]
        contour_end = config["contour_end"]
        contour_step = config["contour_step"]
        
        # X-coordinate selection for this specific plot
        x_input = st.slider(
            f"X-Coordinate ({x_name})", 
            min_value=float(x_min), 
            max_value=float(x_max), 
            value=float(np.sqrt(x_min*x_max)),
            format="%.1f",
            key=f"x_slider_{sheet_name}"
        )
        
        # Get data for this plot
        df = datasets[sheet_name]
        
        # Create the plot
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, interpolated_z = create_contour_plot(
                df, sheet_name, x_input, y_input
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write(f"X: **{x_input:.1f}** {x_name}")
            st.write(f"Y: **{y_input:.1f}** {y_axis_name}")
            
            if interpolated_z is not None:
                format_string = "%.4f" if contour_step < 0.05 else "%.2f"
                st.metric(
                    "Interpolated Value", 
                    format_string % interpolated_z
                )
                st.write(f"Contour Range: {contour_start:.3f} to {contour_end:.3f}")
                st.write(f"Contour Interval: {contour_step:.3f}")
            elif not df.empty:
                st.warning("Coordinates outside data range")
            else:
                st.info("No data available for this sheet")
        
        # Add a separator between plots
        if i < 5:  # Don't add separator after the last plot
            st.markdown("---")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.info(
    """
    **How to Use:**
    1. Adjust the global Y-coordinate slider at the top to set the Y value for all plots
    2. Set the X-coordinate for each individual plot using its slider
    3. View the interpolated value for each plot
    
    Note: Currently, only NA.3, NA.4, and NA.5 have data available.
    """
)
