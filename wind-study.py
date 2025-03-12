import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

# Set page configuration
st.set_page_config(page_title="Contour Analysis Tool")
 
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
        "contour_start": 0.75, "contour_end": 1.7, "contour_step": 0.05
    },
    "NA.4": {
        "x_min": 0.1, "x_max": 20, 
        "y_min": 2, "y_max": 200, 
        "x_name": "Distance inside town terrain (km)",
        "contour_start": 0.56, "contour_end": 1.0, "contour_step": 0.02
    },
    "NA.5": {
        "x_min": 0.1, "x_max": 100, 
        "y_min": 2, "y_max": 200, 
        "x_name": "Distance upwind to shoreline (km)",
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
    """Load data from a local Excel file in the repository."""
    excel_file_path = "contour_data.xlsx"  # Ensure this file is in your repository
    try:
        excel_file = pd.ExcelFile(excel_file_path)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        st.stop()
    
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
                    st.warning(f"Sheet {sheet_name} does not have enough columns.")
                    dataframes[sheet_name] = pd.DataFrame()
            except Exception as e:
                st.warning(f"Error reading sheet {sheet_name}: {e}")
                dataframes[sheet_name] = pd.DataFrame()
        else:
            st.warning(f"Sheet {sheet_name} not found in the Excel file.")
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
            title=f"{sheet_name}: No Data Available Yet",
            annotations=[dict(
                text="This plot does not contain any data yet.",
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig, None
    
    grid_density = 200
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), grid_density)
    y_grid = np.logspace(np.log10(y_min), np.log10(y_max), grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    points = np.column_stack((np.log10(df['x']), np.log10(df['y'])))
    Z_grid = griddata(points, df['z'], (np.log10(X_grid), np.log10(Y_grid)), method='cubic')
    
    def get_interpolated_z(x, y):
        log_x, log_y = np.log10(x), np.log10(y)
        interp_z = griddata(points, df['z'], np.array([[log_x, log_y]]), method='cubic')[0]
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
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[y_input, y_input],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x_input],
            y=[y_input],
            mode='markers',
            marker=dict(color='black', size=10, symbol='x'),
            showlegend=False
        ))
    
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
        height=500,
        plot_bgcolor='rgba(240,240,240,0.95)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig, interpolated_z

# -----------------------
# Main Application Logic
# -----------------------
with st.sidebar:
    st.info("Loading data from local Excel file...")
datasets = load_data()
with st.sidebar:
    num_data_loaded = sum(1 for df in datasets.values() if not df.empty)
    st.success(f"Data loaded: {num_data_loaded} sheets found with data")

# Global y-coordinate numerical input
y_input = st.number_input(
    f"Y-Coordinate ({y_axis_name})", 
    min_value=2.0, 
    max_value=200.0, 
    value=50.0,
    format="%.1f"
)

plot_container = st.container()

with plot_container:
    for i, sheet_name in enumerate(["NA.3", "NA.4", "NA.5", "NA.6", "NA.7", "NA.8"]):
        st.markdown(f"### {sheet_name}")
        
        config = plot_configs[sheet_name]
        x_min, x_max = config["x_min"], config["x_max"]
        x_name = config["x_name"]
        contour_start = config["contour_start"]
        contour_end = config["contour_end"]
        contour_step = config["contour_step"]
        
        # X-coordinate numerical input for this specific plot
        x_input = st.number_input(
            f"X-Coordinate ({x_name})", 
            min_value=float(x_min), 
            max_value=float(x_max), 
            value=float(np.sqrt(x_min * x_max)),
            format="%.1f",
            key=f"x_input_{sheet_name}"
        )
        
        df = datasets[sheet_name]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, interpolated_z = create_contour_plot(df, sheet_name, x_input, y_input)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write(f"X: **{x_input:.1f}** {x_name}")
            st.write(f"Y: **{y_input:.1f}** {y_axis_name}")
            
            if interpolated_z is not None:
                format_string = "%.4f" if contour_step < 0.05 else "%.2f"
                st.metric("Interpolated Value", format_string % interpolated_z)
                st.write(f"Contour Range: {contour_start:.3f} to {contour_end:.3f}")
                st.write(f"Contour Interval: {contour_step:.3f}")
            elif not df.empty:
                st.warning("Coordinates outside data range")
            else:
                st.info("No data available for this sheet")
        
        if i < 5:
            st.markdown("---")

st.markdown("---")
st.info(
    """
    **How to Use:**
    1. Adjust the global Y-coordinate input at the top to set the Y value for all plots.
    2. Type the X-coordinate for each individual plot using its input box.
    3. View the interpolated value for each plot.
    
    Data is loaded directly from the local Excel file in the repository.
    """
)
