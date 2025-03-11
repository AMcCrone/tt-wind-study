import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

st.title("Contour Plot with Interactive Crosshairs")

# Load data from GitHub
data_url = "NA.3-contour_data.xlsx"

@st.cache_data
def load_data(url):
    try:
        df = pd.read_excel(url)
        # Display the first few rows to see the actual column names
        st.write("Preview of loaded data:")
        st.write(df.head())
        
        # Check column names
        st.write("Column names in the dataset:", list(df.columns))
        
        return df
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

# Load the data
df = load_data(data_url)

# Add column selection
st.subheader("Column Selection")
# Get all column names
all_columns = list(df.columns)

# Only proceed if we have at least 3 columns
if len(all_columns) >= 3:
    # Let user select which columns to use
    col1, col2, col3 = st.columns(3)
    with col1:
        x_column = st.selectbox("X Column", all_columns, index=0)
    with col2:
        y_column = st.selectbox("Y Column", all_columns, index=min(1, len(all_columns)-1))
    with col3:
        z_column = st.selectbox("Z Column (Contour Value)", all_columns, index=min(2, len(all_columns)-1))
    
    # Create the grid for interpolation
    # Determine the range of x and y in log space
    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_min, y_max = df[y_column].min(), df[y_column].max()
    
    # Create a dense grid for smooth interpolation
    grid_density = 200
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), grid_density)
    y_grid = np.logspace(np.log10(y_min), np.log10(y_max), grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Use griddata for interpolation - this creates a smooth surface
    points = np.column_stack((np.log10(df[x_column]), np.log10(df[y_column])))
    Z_grid = griddata(points, df[z_column], (np.log10(X_grid), np.log10(Y_grid)), method='cubic')
    
    # User inputs for crosshairs
    st.subheader("Crosshairs Coordinates")
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
        interp_z = griddata(points, df[z_column], np.array([[log_x, log_y]]), method='cubic')[0]
        
        if np.isnan(interp_z):
            # If outside the convex hull, try with 'nearest' method
            interp_z = griddata(points, df[z_column], np.array([[log_x, log_y]]), method='nearest')[0]
        
        return interp_z
    
    # Get the interpolated z value
    interpolated_z = get_interpolated_z(x_input, y_input)
    
    # Format the z value for display
    st.metric("Interpolated Value", f"{interpolated_z:.4f}")
    
    # Find the range of z values to set appropriate contour levels
    z_min, z_max = df[z_column].min(), df[z_column].max()
    # Create contour levels based on the data range
    # Default to the specified range if it fits
    if z_min >= 0.75 and z_max <= 1.70:
        contour_start, contour_end = 0.75, 1.70
    else:
        # Round to appropriate values
        contour_start = np.floor(z_min * 20) / 20  # Round down to nearest 0.05
        contour_end = np.ceil(z_max * 20) / 20    # Round up to nearest 0.05
    
    contour_step = 0.05
    contour_levels = np.arange(contour_start, contour_end + contour_step, contour_step)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the contour plot with smooth color filling - simplified colorbar
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
        ),
        colorscale='Viridis',
        line=dict(width=1),
        colorbar=dict(title='Contour Value')
    ))
    
    # Add contour lines to ensure they are visible
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
        title=f"Contour Plot of {z_column} vs {x_column} and {y_column}",
        xaxis=dict(
            type='log',
            title=f'{x_column} (log scale)',
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        yaxis=dict(
            type='log',
            title=f'{y_column} (log scale)',
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)'
        ),
        height=700,
        plot_bgcolor='rgba(240,240,240,0.95)'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional information about the interpolation
    st.write(f"""
    ### About the Contour Interpolation
    - The contour plot uses cubic interpolation for smooth gradients between contour lines
    - Contour lines are drawn at intervals of {contour_step} from {contour_start} to {contour_end}
    - The crosshairs show the interpolated value at the specified ({x_column},{y_column}) coordinates
    - Enter new coordinates above to update the crosshairs and interpolated value
    """)
    
    # Display a sample of the raw data
    with st.expander("View Raw Data Sample"):
        st.dataframe(df.head(10))

else:
    st.error("The dataset doesn't have enough columns. At least 3 columns are required for X, Y, and Z values.")
