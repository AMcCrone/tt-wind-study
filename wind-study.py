import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

# Set page configuration
st.set_page_config(page_title="Contour Analysis Tool", layout="centered")

# -----------------------
# Authentication Section
# -----------------------
if "password" in st.secrets:
    PASSWORD = st.secrets["password"]
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    def check_password():
        if st.session_state.get("password_input") == PASSWORD:
            st.session_state["authenticated"] = True
        else:
            st.error("Incorrect password.")
    if not st.session_state["authenticated"]:
        st.text_input("Enter Password:", type="password", key="password_input", on_change=check_password)
        st.stop()
else:
    pass

# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Main Application
# -----------------------
st.title("Contour Analysis Tool")

# -----------------------
# Configuration for all plots based on the provided table
# -----------------------
plot_configs = {
    "NA.3": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "Distance upwind to shoreline (km)",
        "contour_start": 0.75, "contour_end": 1.7, "contour_step": 0.05,
        "x_type": "upwind",
        "var_name": "c_rz",
        "section_heading": "Values of $C_r(z)$"
    },
    "NA.4": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "Distance inside town terrain (km)",
        "contour_start": 0.56, "contour_end": 1.0, "contour_step": 0.02,
        "x_type": "town",
        "var_name": "c_rT",
        "section_heading": "Values of correction factor $c_{r,T}$ for sites in Town terrain"
    },
    "NA.5": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "Distance upwind to shoreline (km)",
        "contour_start": 0.07, "contour_end": 0.21, "contour_step": 0.01,
        "x_type": "upwind",
        "var_name": "I_vz",
        "section_heading": "Values of $I_v(z)_{flat}$"
    },
    "NA.6": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "parameter X (m)",
        "contour_start": 1.0, "contour_end": 1.8, "contour_step": 0.05,
        "x_type": "town",
        "var_name": "k_IT",
        "section_heading": "Values of turbulence correction factor $k_{I,T}$ for sites in Town terrain"
    },
    "NA.7": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "parameter Y (m)",
        "contour_start": 1.5, "contour_end": 4.2, "contour_step": 0.1,
        "x_type": "upwind",
        "var_name": "c_ez",
        "section_heading": "Values of $c_e(z)$"
    },
    "NA.8": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "parameter Z (m)",
        "contour_start": 0.60, "contour_end": 1.0, "contour_step": 0.02,
        "x_type": "town",
        "var_name": "c_eT",
        "section_heading": "Values of exposure correction factor $c_{e,T}$ for sites in Town terrain"
    }
}

# Common y-axis name
y_axis_name = "z-h_dis (m)"

# -----------------------
# Data Loading Function
# -----------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Load data from a local Excel file in the repository."""
    excel_file_path = "contour_data.xlsx"  # Ensure this file is in your repository
    try:
        excel_file = pd.ExcelFile(excel_file_path)
    except Exception as e:
        # Silently log the error but don't display
        dataframes = {sheet_name: pd.DataFrame() for sheet_name in plot_configs.keys()}
        return dataframes
    
    dataframes = {}
    for sheet_name in plot_configs.keys():
        if sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if len(df.columns) >= 3:
                    column_names = list(df.columns)
                    column_mapping = {
                        column_names[0]: 'x',
                        column_names[1]: 'y',
                        column_names[2]: 'z'
                    }
                    df = df.rename(columns=column_mapping)
                    df = df.dropna(subset=['x', 'y', 'z'])
                    dataframes[sheet_name] = df
                else:
                    dataframes[sheet_name] = pd.DataFrame()
            except Exception:
                dataframes[sheet_name] = pd.DataFrame()
        else:
            dataframes[sheet_name] = pd.DataFrame()
    
    return dataframes

# -----------------------
# Contour Plot Function
# -----------------------
def create_contour_plot(df, sheet_name, x_input, y_input):
    config = plot_configs[sheet_name]
    x_min, x_max = config["x_min"], config["x_max"]
    y_min, y_max = config["y_min"], config["y_max"]
    x_name = config["x_name"]
    contour_start = config["contour_start"]
    contour_end = config["contour_end"] 
    contour_step = config["contour_step"]
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=20, b=50)
        )
        return fig, None
    
    grid_density = 200
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), grid_density)
    y_grid = np.logspace(np.log10(y_min), np.log10(y_max), grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    points = np.column_stack((np.log10(df['x']), np.log10(df['y'])))
    Z_grid = griddata(points, df['z'], (np.log10(X_grid), np.log10(Y_grid)), method='linear')
    
    def get_interpolated_z(x, y):
        log_x, log_y = np.log10(x), np.log10(y)
        interp_z = griddata(points, df['z'], np.array([[log_x, log_y]]), method='linear')[0]
        if np.isnan(interp_z):
            interp_z = griddata(points, df['z'], np.array([[log_x, log_y]]), method='nearest')[0]
        return interp_z
    
    interpolated_z = None
    if x_min <= x_input <= x_max and y_min <= y_input <= y_max:
        interpolated_z = get_interpolated_z(x_input, y_input)
    
    fig = go.Figure()
    
    # Add filled contour plot
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
            labelfont=dict(size=10, color='black')
        ),
        colorscale='Teal',
        opacity=1,
        line=dict(width=0.5),
        showscale=False,
        colorbar=dict(
            title='Contour Value', 
            thickness=20, 
            len=0.9,
            tickformat='.2f' if contour_step >= 0.05 else '.3f'
        )
    ))
    
    # Add crosshairs and marker for the selected point
    if x_min <= x_input <= x_max and y_min <= y_input <= y_max:
        fig.add_trace(go.Scatter(
            x=[x_input, x_input],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='#DB451D', width=2),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[y_input, y_input],
            mode='lines',
            line=dict(color='#DB451D', width=2),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x_input],
            y=[y_input],
            mode='markers',
            marker=dict(color='#DB451D', size=25, symbol='circle-cross-open', line=dict(width=2)),
            showlegend=False
        ))
    
    fig.update_layout(
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
        height=500,
        plot_bgcolor='rgba(240,240,240,0.95)',
        margin=dict(l=50, r=50, t=20, b=50)
    )
    
    return fig, interpolated_z

# -----------------------
# Main Application Logic
# -----------------------
# Load data without showing spinner
datasets = load_data()

# Create a container for global input controls
global_input_container = st.container()
with global_input_container:
    st.markdown("### Eurocode 1991-1-4 NA inputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Global y-coordinate numerical input
        y_input = st.number_input(
            f"({y_axis_name})",
            min_value=2.0,
            max_value=200.0,
            value=50.0,
            format="%.1f"
        )
    
    with col2:
        # Global x-coordinate for upwind plots (NA.3, NA.5, NA.7)
        upwind_title = "Distance upwind to shoreline (km)"
        x_upwind_input = st.number_input(
            upwind_title,
            value=10.0,
            format="%.1f",
            help="Values outside the range [0.1, 100.0] will be clamped to min/max values"
        )
        
        # Apply limits to x_upwind (clamping logic)
        x_upwind = max(0.1, min(x_upwind_input, 100.0))
    
    with col3:
        # Global x-coordinate for town plots (NA.4, NA.6, NA.8)
        town_title = "Distance inside town terrain (km)"
        x_town_input = st.number_input(
            town_title,
            value=5.0,
            format="%.1f",
            help="Values outside the range [0.1, 20.0] will be clamped to min/max values"
        )
        
        # Apply limits to x_town (clamping logic)
        x_town = max(0.1, min(x_town_input, 20.0))

# Create a table to display all interpolated values
interpolated_values = {}
for sheet_name in plot_configs.keys():
    config = plot_configs[sheet_name]
    x_type = config["x_type"]
    x_min, x_max = config["x_min"], config["x_max"]
    var_name = config["var_name"]
    
    # Use the appropriate global X input based on the plot type
    x_input = x_upwind if x_type == "upwind" else x_town
    
    # Check if x_input is within valid range for this plot
    if x_input < x_min:
        x_input = x_min
    elif x_input > x_max:
        x_input = x_max
    
    df = datasets[sheet_name]
    
    # Get interpolated value
    _, interpolated_z = create_contour_plot(df, sheet_name, x_input, y_input)
    if interpolated_z is not None:
        interpolated_values[var_name] = round(interpolated_z, 3)
    else:
        interpolated_values[var_name] = None

# Display the interpolated values table
st.markdown("### Interpolated Values")
values_df = pd.DataFrame({
    "Variable": list(interpolated_values.keys()),
    "Value": list(interpolated_values.values())
})
st.table(values_df)

st.markdown("---")
plot_container = st.container()

with plot_container:
    for i, sheet_name in enumerate(["NA.3", "NA.4", "NA.5", "NA.6", "NA.7", "NA.8"]):
        config = plot_configs[sheet_name]
        section_heading = config["section_heading"]
        x_type = config["x_type"]
        x_min, x_max = config["x_min"], config["x_max"]
        x_name = config["x_name"]
        var_name = config["var_name"]
        
        st.markdown(f"### {section_heading}")
        
        # Use the appropriate global X input based on the plot type
        x_input = x_upwind if x_type == "upwind" else x_town
        
        # Check if x_input is within valid range for this plot
        if x_input < x_min:
            x_input = x_min
        elif x_input > x_max:
            x_input = x_max
        
        df = datasets[sheet_name]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, interpolated_z = create_contour_plot(df, sheet_name, x_input, y_input)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write(f"X: **{x_input:.1f}** {x_name}")
            st.write(f"Y: **{y_input:.1f}** {y_axis_name}")
            
            if interpolated_z is not None:
                st.metric(f"{var_name}", f"{interpolated_z:.3f}")
            elif not df.empty:
                st.write(f"{var_name}: Coordinates outside data range")
            else:
                st.write(f"{var_name}: No data available")
        
        if i < 5:
            st.markdown("---")

st.markdown("---")
st.markdown(
    """
    **How to Use:**
    1. Adjust the global Y-coordinate input at the top to set the Y value for all plots.
    2. Use the "Distance upwind to shoreline" input to set X values for plots NA.3, NA.5, and NA.7.
    3. Use the "Distance inside town terrain" input to set X values for plots NA.4, NA.6, and NA.8.
    4. View the interpolated values in the table and for each plot.
    """
)
