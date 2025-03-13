import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
import tempfile

# Define a TT colours.
TT_Orange = "rgb(211,69,29)"
TT_Olive = "rgb(139,144,100)"
TT_LightBlue = "rgb(136,219,223)"
TT_MidBlue = "rgb(0,163,173)"
TT_DarkBlue = "rgb(0,48,60)"
TT_Grey = "rgb(99,102,105)"
TT_Purple = "rgb(128,0,128)"  # Color for custom section

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

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Section Design Tool", layout="wide")
st.title("Mullion Design Widget - *Beta Version*")
st.markdown("Time to find that one in a mullion...")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")
plot_material = st.sidebar.selectbox("Select Material", options=["Aluminium", "Steel"], index=0)

# Read the Excel file (stored in the repository) using a relative path.
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "Cross_Sections_Database.xlsx")
SHEET = "Alu Mullion Database" if plot_material == "Aluminium" else "Steel Mullion Database"
try:
    df = pd.read_excel(file_path, sheet_name=SHEET, engine="openpyxl")
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

selected_columns = ["Supplier", "Profile Name", "Material", "Reinf", "Depth", "Iyy", "Wyy"]
df_selected = df[selected_columns].iloc[1:].reset_index(drop=True)

# Material properties
material_props = {
    "Aluminium": {"fy": 160, "E": 70000},
    "Steel": {"fy": 355, "E": 210000}
}

suppliers_all = sorted(df_selected["Supplier"].unique())
selected_suppliers = st.sidebar.multiselect("Select Suppliers", options=suppliers_all, default=suppliers_all)
barrier_load_option = st.sidebar.radio("Barrier Load (kN/m)", options=["None", "0.74", "1.5", "3"], index=0)
selected_barrier_load = 0 if barrier_load_option == "None" else float(barrier_load_option)
ULS_case = st.sidebar.radio("ULS Load Case", options=[
    "ULS 1: 1.5WL + 0.75BL", 
    "ULS 2: 0.75WL + 1.5BL", 
    "ULS 3: 1.5WL", 
    "ULS 4: 1.5BL"
], index=0)
SLS_case = st.sidebar.radio("SLS Load Case", options=["SLS 1: WL", "SLS 2: BL"], index=0)
view_3d_option = st.sidebar.radio("3D View", options=[
    "Isometric: Overview", 
    "XY Plane: Utilisation", 
    "XZ Plane: Section Depth"
], index=0)
wind_pressure = st.sidebar.slider("Wind Pressure (kPa)", 0.1, 5.0, 1.0, 0.1)
bay_width = st.sidebar.slider("Bay Width (mm)", 500, 10000, 3000, 250)
mullion_length = st.sidebar.slider("Mullion Length (mm)", 2500, 12000, 4000, 250)
barrier_L = 1100  # constant barrier length (mm)

# ---------------------------
# Custom Section Input
# ---------------------------
st.sidebar.header("Custom Section")
use_custom_section = st.sidebar.checkbox("Add Custom Section?", False)

if use_custom_section:
    custom_section_name = st.sidebar.text_input("Section Name", "My Custom Section")
    custom_section_material = plot_material  # Use the main material selection
    custom_section_depth = st.sidebar.number_input("Section Depth (mm)", 50, 500, 150)
    custom_section_modulus = st.sidebar.number_input("Section Modulus (cm³)", 1.0, 1000.0, 50.0)
    custom_section_inertia = st.sidebar.number_input("Moment of Inertia (cm⁴)", 1.0, 10000.0, 500.0)
    
    # Convert to mm units to match the main dataframe
    custom_section_modulus_mm3 = custom_section_modulus * 1000  # cm³ to mm³
    custom_section_inertia_mm4 = custom_section_inertia * 10000  # cm⁴ to mm⁴

# ---------------------------
# Helper Functions for PDF Export using reportlab
# ---------------------------
def get_pdf_bytes(fig):
    buffer = io.BytesIO()
    fig.write_image(buffer, format="pdf")
    buffer.seek(0)
    return buffer.getvalue()

def create_report_pdf(
    inputs, load_cases, calculations, passing_sections, 
    uls_fig, sls_fig, util_fig
):
    """Create a comprehensive PDF report with all calculation details"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=landscape(A4),
        topMargin=10*mm, 
        bottomMargin=10*mm,
        leftMargin=10*mm, 
        rightMargin=10*mm
    )
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkblue
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("Mullion Design Calculation Report", title_style))
    elements.append(Spacer(1, 10*mm))
    
    # Input Parameters
    elements.append(Paragraph("Input Parameters", section_title))
    data = [
        ["Parameter", "Value"],
        ["Material", inputs["material"]],
        ["Wind Pressure", f"{inputs['wind_pressure']:.2f} kPa"],
        ["Bay Width", f"{inputs['bay_width']} mm"],
        ["Mullion Length", f"{inputs['mullion_length']} mm"],
        ["Barrier Load", f"{inputs['barrier_load']:.2f} kN/m"]
    ]
    t = Table(data, colWidths=[100*mm, 80*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10*mm))
    
    # Load Cases
    elements.append(Paragraph("Load Cases", section_title))
    data = [
        ["Case", "Description"],
        ["ULS Case", load_cases["uls_case"]],
        ["SLS Case", load_cases["sls_case"]]
    ]
    t = Table(data, colWidths=[100*mm, 80*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10*mm))
    
    # Calculation Results
    elements.append(Paragraph("Calculation Results", section_title))
    data = [
        ["Parameter", "Value"],
        ["Required Section Modulus", f"{calculations['Z_req_cm3']:.2f} cm³"],
        ["Deflection Limit", f"{calculations['defl_limit']:.2f} mm"]
    ]
    t = Table(data, colWidths=[100*mm, 80*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10*mm))
    
    # Passing Sections
    elements.append(Paragraph("Passing Sections", section_title))
    if len(passing_sections) > 0:
        headers = ["Supplier", "Profile", "Depth (mm)", "Z (cm³)", "I (cm⁴)", "ULS Util. (%)", "SLS Util. (%)"]
        data = [headers]
        for section in passing_sections:
            data.append([
                section["supplier"],
                section["profile"],
                f"{section['depth']:.1f}",
                f"{section['z_cm3']:.2f}",
                f"{section['i_cm4']:.2f}",
                f"{section['uls_util']*100:.2f}%",
                f"{section['sls_util']*100:.2f}%"
            ])
        
        table_width = 250*mm
        col_widths = [table_width/7] * 7
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph("No passing sections found!", normal_style))
    elements.append(Spacer(1, 10*mm))
    
    # Save the figures as temporary images and include them
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_uls:
        uls_fig.write_image(f_uls.name, format="png", width=800, height=400, scale=2)
        elements.append(Paragraph("ULS Design Chart", heading2_style))
        elements.append(Image(f_uls.name, width=7*inch, height=3.5*inch))
        uls_temp_file = f_uls.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_sls:
        sls_fig.write_image(f_sls.name, format="png", width=800, height=400, scale=2)
        elements.append(Paragraph("SLS Design Chart", heading2_style))
        elements.append(Image(f_sls.name, width=7*inch, height=3.5*inch))
        sls_temp_file = f_sls.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_util:
        util_fig.write_image(f_util.name, format="png", width=800, height=400, scale=2)
        elements.append(Paragraph("3D Utilisation Chart", heading2_style))
        elements.append(Image(f_util.name, width=7*inch, height=3.5*inch))
        util_temp_file = f_util.name
        
    # Build and return the PDF
    doc.build(elements)
    buffer.seek(0)
    
    # Clean up temp files
    os.unlink(uls_temp_file)
    os.unlink(sls_temp_file)
    os.unlink(util_temp_file)
    
    return buffer.getvalue()

# ---------------------------
# Generate Plotly Figures
# ---------------------------
def generate_plots():
    # Basic calculations
    p = wind_pressure * 0.001  # kPa to N/mm²
    bay = bay_width
    L = mullion_length
    w = p * bay
    M_WL = (w * L**2) / 8
    M_BL = ((selected_barrier_load * bay) * barrier_L) / 2

    if ULS_case.startswith("ULS 1"):
        M_ULS = 1.5 * M_WL + 0.75 * M_BL
    elif ULS_case.startswith("ULS 2"):
        M_ULS = 0.75 * M_WL + 1.5 * M_BL
    elif ULS_case.startswith("ULS 3"):
        M_ULS = 1.5 * M_WL
    else:  # ULS 4
        M_ULS = 1.5 * M_BL

    # Calculate required section modulus
    fyc = material_props[plot_material]["fy"]
    E = material_props[plot_material]["E"]
    Z_req = M_ULS / fyc
    Z_req_cm3 = Z_req / 1000  # Convert from mm³ to cm³

    # Calculate SLS deflection
    if SLS_case.startswith("SLS 1"):
        M_SLS = M_WL
    else:  # SLS 2
        M_SLS = M_BL

    # Calculate deflection limit
    defl_limit = L / 175  # Standard deflection limit
    
    # Filter by selected suppliers
    df_filtered = df_selected[df_selected["Supplier"].isin(selected_suppliers)].copy()
    
    # Add custom section if enabled
    if use_custom_section:
        custom_row = pd.DataFrame({
            "Supplier": ["Custom"],
            "Profile Name": [custom_section_name],
            "Material": [custom_section_material],
            "Reinf": ["N/A"],
            "Depth": [custom_section_depth],
            "Iyy": [custom_section_inertia_mm4],
            "Wyy": [custom_section_modulus_mm3]
        })
        df_filtered = pd.concat([df_filtered, custom_row], ignore_index=True)
    
    # Calculate utilization for each section
    df_filtered["Z_cm3"] = df_filtered["Wyy"] / 1000  # Convert from mm³ to cm³
    df_filtered["I_cm4"] = df_filtered["Iyy"] / 10000  # Convert from mm⁴ to cm⁴
    df_filtered["ULS_Util"] = Z_req_cm3 / df_filtered["Z_cm3"]
    
    # Calculate SLS deflection
    df_filtered["SLS_Defl"] = (5 * w * L**4) / (384 * E * df_filtered["Iyy"]) if SLS_case.startswith("SLS 1") else \
                            (selected_barrier_load * bay * barrier_L**3) / (3 * E * df_filtered["Iyy"])
    df_filtered["SLS_Util"] = df_filtered["SLS_Defl"] / defl_limit
    
    # Mark passing sections
    df_filtered["Passing"] = (df_filtered["ULS_Util"] <= 1.0) & (df_filtered["SLS_Util"] <= 1.0)
    
    # Sort by utilization
    df_filtered = df_filtered.sort_values(by=["ULS_Util", "SLS_Util"], ascending=[True, True])
    
    # Create passing sections list for PDF export
    passing_sections = []
    for _, row in df_filtered[df_filtered["Passing"]].iterrows():
        passing_sections.append({
            "supplier": row["Supplier"],
            "profile": row["Profile Name"],
            "depth": row["Depth"],
            "z_cm3": row["Z_cm3"],
            "i_cm4": row["I_cm4"],
            "uls_util": row["ULS_Util"],
            "sls_util": row["SLS_Util"]
        })
    
    # Create ULS plot
    fig_uls = go.Figure()
    
    for supplier in selected_suppliers:
        supplier_data = df_filtered[df_filtered["Supplier"] == supplier]
        if not supplier_data.empty:
            fig_uls.add_trace(go.Scatter(
                x=supplier_data["Depth"],
                y=supplier_data["Z_cm3"],
                mode="markers",
                marker=dict(
                    size=10,
                    opacity=0.8,
                    color=supplier_data["ULS_Util"],
                    colorscale="Viridis",
                    colorbar=dict(title="ULS Utilization"),
                    cmin=0,
                    cmax=1.5
                ),
                name=supplier,
                text=supplier_data["Profile Name"],
                hovertemplate="<b>%{text}</b><br>Depth: %{x} mm<br>Z: %{y:.2f} cm³<br>ULS Util: %{marker.color:.2f}<extra></extra>"
            ))
    
    # Add custom section if enabled
    if use_custom_section:
        custom_data = df_filtered[df_filtered["Supplier"] == "Custom"]
        if not custom_data.empty:
            fig_uls.add_trace(go.Scatter(
                x=custom_data["Depth"],
                y=custom_data["Z_cm3"],
                mode="markers",
                marker=dict(
                    size=15,
                    opacity=1.0,
                    color=TT_Purple,
                    line=dict(width=2, color="black")
                ),
                name="Custom Section",
                text=custom_data["Profile Name"],
                hovertemplate="<b>%{text}</b><br>Depth: %{x} mm<br>Z: %{y:.2f} cm³<br>ULS Util: %{customdata[0]:.2f}<extra></extra>",
                customdata=custom_data[["ULS_Util"]]
            ))
    
    # Add required section modulus line
    fig_uls.add_shape(
        type="line",
        x0=0,
        y0=Z_req_cm3,
        x1=max(df_filtered["Depth"]) * 1.1,
        y1=Z_req_cm3,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig_uls.add_annotation(
        x=max(df_filtered["Depth"]) * 0.95,
        y=Z_req_cm3 * 1.05,
        text=f"Required Z = {Z_req_cm3:.2f} cm³",
        showarrow=False,
        font=dict(color="red", size=14),
    )
    
    fig_uls.update_layout(
        title="ULS Design Chart: Section Modulus vs Depth",
        xaxis_title="Section Depth (mm)",
        yaxis_title="Section Modulus (cm³)",
        legend_title="Supplier",
        template="plotly_white",
        height=600,
    )
    
    # Create SLS plot
    fig_sls = go.Figure()
    
    for supplier in selected_suppliers:
        supplier_data = df_filtered[df_filtered["Supplier"] == supplier]
        if not supplier_data.empty:
            fig_sls.add_trace(go.Scatter(
                x=supplier_data["Depth"],
                y=supplier_data["SLS_Defl"],
                mode="markers",
                marker=dict(
                    size=10,
                    opacity=0.8,
                    color=supplier_data["SLS_Util"],
                    colorscale="Viridis",
                    colorbar=dict(title="SLS Utilization"),
                    cmin=0,
                    cmax=1.5
                ),
                name=supplier,
                text=supplier_data["Profile Name"],
                hovertemplate="<b>%{text}</b><br>Depth: %{x} mm<br>Defl: %{y:.2f} mm<br>SLS Util: %{marker.color:.2f}<extra></extra>"
            ))
    
    # Add custom section if enabled
    if use_custom_section:
        custom_data = df_filtered[df_filtered["Supplier"] == "Custom"]
        if not custom_data.empty:
            fig_sls.add_trace(go.Scatter(
                x=custom_data["Depth"],
                y=custom_data["SLS_Defl"],
                mode="markers",
                marker=dict(
                    size=15,
                    opacity=1.0,
                    color=TT_Purple,
                    line=dict(width=2, color="black")
                ),
                name="Custom Section",
                text=custom_data["Profile Name"],
                hovertemplate="<b>%{text}</b><br>Depth: %{x} mm<br>Defl: %{y:.2f} mm<br>SLS Util: %{customdata[0]:.2f}<extra></extra>",
                customdata=custom_data[["SLS_Util"]]
            ))
    
    # Add deflection limit line
    fig_sls.add_shape(
        type="line",
        x0=0,
        y0=defl_limit,
        x1=max(df_filtered["Depth"]) * 1.1,
        y1=defl_limit,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig_sls.add_annotation(
        x=max(df_filtered["Depth"]) * 0.95,
        y=defl_limit * 1.05,
        text=f"Deflection Limit = {defl_limit:.2f} mm",
        showarrow=False,
        font=dict(color="red", size=14),
    )
    
    fig_sls.update_layout(
        title="SLS Design Chart: Deflection vs Depth",
        xaxis_title="Section Depth (mm)",
        yaxis_title="Deflection (mm)",
        legend_title="Supplier",
        template="plotly_white",
        height=600,
    )
    
    # Create 3D utilization plot
    fig_3d = go.Figure()
    
    for supplier in selected_suppliers:
        supplier_data = df_filtered[df_filtered["Supplier"] == supplier]
        if not supplier_data.empty:
            marker_color = supplier_data["ULS_Util"] if view_3d_option == "XY Plane: Utilisation" else supplier_data["Depth"]
            colorscale = "Viridis" if view_3d_option == "XY Plane: Utilisation" else "Turbo"
            cmin = 0 if view_3d_option == "XY Plane: Utilisation" else min(df_filtered["Depth"])
            cmax = 1.5 if view_3d_option == "XY Plane: Utilisation" else max(df_filtered["Depth"])
            colorbar_title = "ULS Utilization" if view_3d_option == "XY Plane: Utilisation" else "Depth (mm)"
            
            fig_3d.add_trace(go.Scatter3d(
                x=supplier_data["Z_cm3"],
                y=supplier_data["SLS_Defl"],
                z=supplier_data["Depth"],
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.8,
                    color=marker_color,
                    colorscale=colorscale,
                    colorbar=dict(title=colorbar_title),
                    cmin=cmin,
                    cmax=cmax
                ),
                name=supplier,
                text=supplier_data["Profile Name"],
                hovertemplate="<b>%{text}</b><br>Z: %{x:.2f} cm³<br>Defl: %{y:.2f} mm<br>Depth: %{z} mm<br>ULS Util: %{customdata[0]:.2f}<br>SLS Util: %{customdata[1]:.2f}<extra></extra>",
                customdata=supplier_data[["ULS_Util", "SLS_Util"]]
            ))
    
    # Add custom section if enabled
    if use_custom_section:
        custom_data = df_filtered[df_filtered["Supplier"] == "Custom"]
        if not custom_data.empty:
            fig_3d.add_trace(go.Scatter3d(
                x=custom_data["Z_cm3"],
                y=custom_data["SLS_Defl"],
                z=custom_data["Depth"],
                mode="markers",
                marker=dict(
                    size=12,
                    opacity=1.0,
                    color=TT_Purple,
                    line=dict(width=2, color="black")
                ),
                name="Custom Section",
                text=custom_data["Profile Name"],
                hovertemplate="<b>%{text}</b><br>Z: %{x:.2f} cm³<br>Defl: %{y:.2f} mm<br>Depth: %{z} mm<br>ULS Util: %{customdata[0]:.2f}<br>SLS Util: %{customdata[1]:.2f}<extra></extra>",
                customdata=custom_data[["ULS_Util", "SLS_Util"]]
            ))
    
    # Add reference planes
    x_vals = np.linspace(min(df_filtered["Z_cm3"]) * 0.9, max(df_filtered["Z_cm3"]) * 1.1, 10)
    z_vals = np.linspace(min(df_filtered["Depth"]) * 0.9, max(df_filtered["Depth"]) * 1.1, 10)
    X, Z = np.meshgrid(x_vals, z_vals)
    Y_uls = np.ones(X.shape) * defl_limit
    
    fig_3d.add_trace(go.Surface(
        x=X, y=Y_uls, z=Z,
        colorscale=[[0, "rgba(255,0,0,0.2)"], [1, "rgba(255,0,0,0.2)"]],
        showscale=False,
        name="Deflection Limit"
    ))
    
    # Set 3D view based on selection
    if view_3d_option == "Isometric: Overview":
        camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    elif view_3d_option == "XY Plane: Utilisation":
        camera = dict(eye=dict(x=1.5, y=1.5, z=0.1))
    else:  # XZ Plane: Section Depth
        camera = dict(eye=dict(x=0.1, y=1.5, z=1.5))
    
    fig_3d.update_layout(
        title="3D Utilization Chart",
        scene=dict(
            xaxis_title="Section Modulus (cm³)",
            yaxis_title="Deflection (mm)",
            zaxis_title="Section Depth (mm)",
            camera=camera
        ),
        template="plotly_white",
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Return all calculated data and figures
    return {
        "Z_req_cm3": Z_req_cm3,
        "defl_limit": defl_limit,
        "df_filtered": df_filtered,
        "passing_sections": passing_sections,
        "fig_uls": fig_uls,
        "fig_sls": fig_sls,
        "fig_3d": fig_3d,
        "M_ULS": M_ULS,
        "M_SLS": M_SLS,
        "M_WL": M_WL,
        "M_BL": M_BL
    }

# ---------------------------
# Main App UI
# ---------------------------
st.markdown("## Calculation Results")

# Execute calculations and generate plots
results = generate_plots()

# Display calculation results
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"Required Section Modulus: **{results['Z_req_cm3']:.2f} cm³**")
with col2:
    st.info(f"Deflection Limit: **{results['defl_limit']:.2f} mm**")
with col3:
    # Add PDF export button here
    pdf_data = create_report_pdf(
        inputs={
            "material": plot_material,
            "wind_pressure": wind_pressure,
            "bay_width": bay_width,
            "mullion_length": mullion_length,
            "barrier_load": selected_barrier_load
        },
        load_cases={
            "uls_case": ULS_case,
            "sls_case": SLS_case
        },
        calculations={
            "Z_req_cm3": results['Z_req_cm3'],
            "defl_limit": results['defl_limit']
        },
        passing_sections=results['passing_sections'],
        uls_fig=results['fig_uls'],
        sls_fig=results['fig_sls'],
        util_fig=results['fig_3d']
    )
    
    st.download_button(
        label="Save PDF Report",
        data=pdf_data,
        file_name="mullion_design_report.pdf",
        mime="application/pdf",
    )

# Display section database table with improved formatting
st.markdown("## Section Database")
st.markdown("*Passing sections are highlighted, failing sections are greyed out*")

# Prepare the dataframe for display
display_df = results["df_filtered"].copy()
display_df = display_df[["Supplier", "Profile Name", "Depth", "Z_cm3", "I_cm4", "ULS_Util", "SLS_Util", "Passing"]]
display_df.columns = ["Supplier", "Profile", "Depth (mm)", "Z (cm³)", "I (cm⁴)", "ULS Util.", "SLS Util.", "Passing"]

# Format the numerical columns to 2 decimal places
for col in ["Z (cm³)", "I (cm⁴)"]:
    display_df[col] = display_df[col].map("{:.2f}".format)

# Format the utilization columns as percentages with 2 decimal places
display_df["ULS Util."] = display_df["ULS Util."].map("{:.2f}%".format)
display_df["SLS Util."] = display_df["SLS Util."].map("{:.2f}%".format)

# Create a function to style the rows based on passing condition
def style_rows(row):
    if row["Passing"]:
        return ["background-color: #E5F6F2"] * len(row)
    else:
        return ["color: #999999; background-color: #F5F5F5"] * len(row)

# Drop the "Passing" column before display
display_df = display_df.drop(columns=["Passing"])

# Display the styled dataframe
st.dataframe(display_df.style.apply(style_rows, axis=1), height=400, use_container_width=True)

# Display plots in tabs
tab1, tab2, tab3 = st.tabs(["ULS Design", "SLS Design", "3D Visualization"])

with tab1:
    st.plotly_chart(results["fig_uls"], use_container_width=True)
    
with tab2:
    st.plotly_chart(results["fig_sls"], use_container_width=True)
    
with tab3:
    st.plotly_chart(results["fig_3d"], use_container_width=True)

# Add explanation of the calculations
with st.expander("Calculation Details"):
    st.markdown("""
    ### Calculations Explained
    
    #### Loads
    - Wind Load moment: M_WL = (w * L²) / 8, where w = p * bay (N/mm)
    - Barrier Load moment: M_BL = ((barrier_load * bay) * barrier_L) / 2
    
    #### ULS Load Cases
    - ULS 1: 1.5WL + 0.75BL
    - ULS 2: 0.75WL + 1.5BL
    - ULS 3: 1.5WL
    - ULS 4: 1.5BL
    
    #### Section Requirements
    - Required Section Modulus: Z_req = M_ULS / fy
    - Deflection Limit: L / 175
    
    #### SLS Load Cases
    - SLS 1 (Wind Load): δ = (5 * w * L⁴) / (384 * E * I)
    - SLS 2 (Barrier Load): δ = (P * L³) / (3 * E * I), where P = barrier_load * bay
    """)

# Add footer with version info
st.markdown("---")
st.markdown("*Mullion Design Widget - Beta Version © " + str(pd.Timestamp.now().year) + "*")
