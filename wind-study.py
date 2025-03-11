import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

# Set page configuration
st.set_page_config(page_title="Contour Analysis Tool", layout="wide")

# -----------------------
# Authentication Section
# -----------------------
# Retrieve the password from secrets
try:
    PASSWORD = st.secrets["password"]
except:
    # For local development without secrets
    PASSWORD = "password123"  # Replace with your actual password in production

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
    st.title("Contour Analysis Tool")
    st.text_input("Enter Password:", type="password", key="password_input", on_change=check_password)
    st.stop()

# -----------------------
# Main Application
# -----------------------
st.title("Contour Analysis Tool")

# -----------------------
# Data Loading Functions
# -----------------------
@st.cache_data
def load_data(url):
    """Load Excel file with multiple sheets from GitHub."""
    try:
        # Read the Excel file
        excel_file = pd.ExcelFile(url)
        
        # Get all sheet names
        sheet_names = excel_file.sheet_names
        
        # Filter for only NA.3 to NA.8 sheets if they exist
        target_sheets = [f"NA.{i}" for i in range(3, 9)]
        available_sheets = [sheet for sheet in sheet_names if sheet in target_sheets]
        
        if not available_sheets:
            available_sheets = sheet_names
            
        # Create a dictionary to store DataFrames for each sheet
        dataframes = {}
        
        # Load each sheet into a DataFrame
        for sheet in available_sheets:
            dataframes[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
            
        return dataframes, available_sheets
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create sample data for demonstration
        sample_dfs = {}
        sample_sheets = [f"NA.{i}" for i in range(3, 9)]
        
        # Create sample data for each sheet (only NA.3 and NA.4 have realistic data)
        for sheet in sample_sheets:
            if sheet in ["NA.3", "NA.4"]:
                # More realistic data for NA.3 and NA.4
                x_vals = np.logspace(1, 3, 50)
                y_vals = np.logspace(1, 3, 50)
                if sheet == "NA.3":
                    z_vals = np.random.uniform(0.75, 1.7, 50)
                else:  # NA.4
                    z_vals = np.random.uniform(0.85, 1.5, 50)
                
                sample_dfs[sheet] = pd.DataFrame({
                    'x': x_vals,
                    'y': y_vals,
                    'z': z_vals
                })
            else:
                # Empty dataframe for others
                sample_dfs[sheet] = pd.DataFrame({
                    'x': [],
                    'y': [],
                    'z': []
                })
        
        return sample_dfs, sample_sheets

# -----------------------
# Contour Plot Function
# -----------------------
def create_contour_plot(df, x_column, y_column, z_column, x_input, y_input):
    """Create a contour plot with interactive crosshairs."""
    # Check if dataframe is empty
    if df.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available for This Sheet",
            annotations=[
                dict(
                    text="This sheet does not contain any data yet.",
                    showarrow=False,
                    font=dict(size=16),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ],
            height=700
        )
        return fig, None, {"contour_start": 0, "contour_end": 0, "contour_step": 0}
    
    # Determine the range of x and y in log space
    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_min, y_max = df[y_column].min(), df[y_column].max()
    
    # Create a dense grid for smooth interpolation
    grid_density = 200
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), grid_density)
    y_grid = np.logspace(np.log10(y_min), np.log10(y_max), grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Use griddata for interpolation - creates a smooth surface
    points = np.column_stack((np.log10(df[x_column]), np.log10(df[y_column])))
    Z_grid = griddata(points, df[z_column], (np.log10(X_grid), np.log10(Y_grid)), method='cubic')
    
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
    
    return fig, interpolated_z, {
        'contour_start': contour_start,
        'contour_end': contour_end,
        'contour_step': contour_step
    }

# -----------------------
# Main Application Logic
# -----------------------

# Sidebar for data selection and configuration
with st.sidebar:
    st.header("Data Configuration")
    
    # GitHub data URL
    data_url = st.text_input(
        "GitHub Data URL",
        value="contour_data"
    )
    
    # Load the data and get sheet names
    datasets, sheet_names = load_data(data_url)
    
    # Dataset (sheet) selection
    selected_sheet = st.selectbox("Select Dataset (NA.3-NA.8)", sheet_names)
    
    # Get the selected DataFrame
    df = datasets[selected_sheet]
    
    # Display data status
    if df.empty:
        st.warning(f"No data available for {selected_sheet} yet.")
    else:
        st.success(f"Data loaded successfully for {selected_sheet}.")
    
    # Get all column names from the selected dataset
    all_columns = list(df.columns) if not df.empty else ["x", "y", "z"]
    
    st.subheader("Column Selection")
    # Let user select which columns to use
    x_column = st.selectbox("X Column", all_columns, index=0)
    y_column = st.selectbox("Y Column", all_columns, index=min(1, len(all_columns)-1))
    z_column = st.selectbox("Z Column (Contour Value)", all_columns, index=min(2, len(all_columns)-1))
    
    st.subheader("Crosshairs Coordinates")
    # Only proceed if we have valid columns and non-empty dataframe
    if not df.empty and x_column in df.columns and y_column in df.columns and z_column in df.columns:
        x_min, x_max = df[x_column].min(), df[x_column].max()
        y_min, y_max = df[y_column].min(), df[y_column].max()
        
        x_input = st.number_input("X Coordinate", 
                                min_value=float(x_min), 
                                max_value=float(x_max), 
                                value=float(np.sqrt(x_min*x_max)))
        
        y_input = st.number_input("Y Coordinate", 
                                min_value=float(y_min), 
                                max_value=float(y_max), 
                                value=float(np.sqrt(y_min*y_max)))
    else:
        # Default values for empty dataframes
        x_input, y_input = 100, 100
        if df.empty:
            st.info("Enter coordinates when data becomes available.")
        else:
            st.error("Selected columns not found in the dataset.")

# Main content area
tab1, tab2 = st.tabs(["Contour Plot", "Data View"])

with tab1:
    if df.empty:
        st.info(f"No data available for {selected_sheet} yet. Data is currently only available for NA.3 and NA.4.")
        
        # Show a placeholder figure
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"No Data Available for {selected_sheet}",
            annotations=[
                dict(
                    text="This sheet does not contain any data yet.",
                    showarrow=False,
                    font=dict(size=16),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ],
            height=700
        )
        st.plotly_chart(empty_fig, use_container_width=True)
    elif len(all_columns) >= 3 and x_column in df.columns and y_column in df.columns and z_column in df.columns:
        # Create the contour plot
        fig, interpolated_z, contour_info = create_contour_plot(
            df, x_column, y_column, z_column, x_input, y_input
        )
        
        # Display information about the interpolated value
        col1, col2 = st.columns([1, 3])
        with col1:
            if interpolated_z is not None:
                st.metric("Interpolated Value", f"{interpolated_z:.4f}")
            st.write(f"Dataset: **{selected_sheet}**")
            st.write(f"X: **{x_input:.4f}**")
            st.write(f"Y: **{y_input:.4f}**")
        
        with col2:
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        
        # Information about the interpolation
        if interpolated_z is not None:
            st.write(f"""
            ### About the Contour Interpolation
            - The contour plot uses cubic interpolation for smooth gradients between contour lines
            - Contour lines are drawn at intervals of {contour_info['contour_step']} from {contour_info['contour_start']} to {contour_info['contour_end']}
            - The crosshairs show the interpolated value at the specified ({x_column},{y_column}) coordinates
            - Enter new coordinates in the sidebar to update the crosshairs and interpolated value
            """)
    else:
        st.error("The dataset doesn't have enough columns. At least 3 columns are required for X, Y, and Z values.")

with tab2:
    # Display the raw data
    st.subheader(f"Raw Data from '{selected_sheet}'")
    
    if df.empty:
        st.info(f"No data available for {selected_sheet} yet.")
    else:
        st.dataframe(df)
        
        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            f"{selected_sheet}.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Dataset summary
        st.subheader("Dataset Summary")
        st.write(f"Number of data points: {len(df)}")
        st.write(f"Columns: {', '.join(df.columns)}")
        
        # Display basic statistics
        st.write("Summary Statistics:")
        st.write(df.describe())

# -----------------------
# Footer
# -----------------------
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **How to Use:**
    1. Select a dataset from NA.3 to NA.8 from the dropdown
    2. Choose columns for X, Y, and Z values
    3. Enter coordinates to position the crosshairs
    4. View the interpolated value at those coordinates
    
    Note: Currently, only NA.3 and NA.4 have data available.
    The contour lines are drawn at 0.05 intervals, and the gradient 
    between them is smooth for easy value interpretation.
    """
)
