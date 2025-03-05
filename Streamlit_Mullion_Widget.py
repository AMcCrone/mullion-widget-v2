import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from PyPDF2 import PdfMerger  # for merging PDFs

# Define a TT colours.
TT_Orange = "rgb(211,69,29)"
TT_Olive = "rgb(139,144,100)"
TT_LightBlue = "rgb(136,219,223)"
TT_MidBlue = "rgb(0,163,173)"
TT_DarkBlue = "rgb(0,48,60)"
TT_Grey = "rgb(99,102,105)"

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Section Design Tool", layout="wide")
st.title("Mullion Design Widget")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")
plot_material = st.sidebar.selectbox("Select Material", options=["Steel", "Aluminium"], index=0)

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
show_data_labels = st.sidebar.checkbox("Show Data Labels", value=False)
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
bay_width = st.sidebar.slider("Bay Width (mm)", 2000, 10000, 3000, 500)
mullion_length = st.sidebar.slider("Mullion Length (mm)", 2500, 12000, 4000, 250)
barrier_L = 1100  # constant barrier length (mm)

# ---------------------------
# Helper Functions for PDF Export
# ---------------------------
def get_pdf_bytes(fig):
    buffer = io.BytesIO()
    fig.write_image(buffer, format="pdf")
    buffer.seek(0)
    return buffer.getvalue()

def merge_pdfs(pdf_bytes_list):
    merger = PdfMerger()
    for pdf_bytes in pdf_bytes_list:
        merger.append(io.BytesIO(pdf_bytes))
    out_buffer = io.BytesIO()
    merger.write(out_buffer)
    merger.close()
    out_buffer.seek(0)
    return out_buffer.getvalue()

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
    elif ULS_case.startswith("ULS 4"):
        M_ULS = 1.5 * M_BL
    else:
        M_ULS = 0

    fy = material_props[plot_material]["fy"]
    Z_req = M_ULS / fy          # mm³
    Z_req_cm3 = Z_req / 1000     # cm³

    if L <= 3000:
        defl_limit = L / 200
    elif L < 7500:
        defl_limit = 5 + L / 300
    else:
        defl_limit = L / 250

    df_mat = df_selected[(df_selected["Material"] == plot_material) & 
                         (df_selected["Supplier"].isin(selected_suppliers))]
    if df_mat.empty:
        st.error("No sections selected.")
        st.stop()

    depths   = df_mat["Depth"].values        # mm
    Wyy_vals = df_mat["Wyy"].values           # mm³
    Iyy_vals = df_mat["Iyy"].values           # mm⁴
    profiles = df_mat["Profile Name"].values
    reinf    = df_mat["Reinf"].values
    supps    = df_mat["Supplier"].values
    available_cm3 = Wyy_vals / 1000           # cm³

    # ----- ULS Plot -----
    uls_passed = available_cm3 >= Z_req_cm3
    uls_colors = ['seagreen' if passed else 'darkred' for passed in uls_passed]
    uls_symbols = ['square' if r else 'circle' for r in reinf]
    uls_hover = [
        f"{profiles[i]}<br>Supplier: {supps[i]}<br>Depth: {depths[i]} mm<br>Z: {available_cm3[i]:.2f} cm³<br>ULS: {'Pass' if uls_passed[i] else 'Fail'}"
        for i in range(len(depths))
    ]
    x_min = np.min(depths) * 0.95
    x_max = np.max(depths) * 1.05
    uls_ymax = 4 * Z_req_cm3

    uls_fig = go.Figure()
    uls_fig.add_shape(type="rect",
                      x0=x_min, x1=x_max,
                      y0=Z_req_cm3, y1=uls_ymax,
                      fillcolor=TT_LightBlue, opacity=0.2, line_width=0)
    uls_fig.add_shape(type="rect",
                      x0=x_min, x1=x_max,
                      y0=0, y1=Z_req_cm3,
                      fillcolor=TT_MidBlue, opacity=0.2, line_width=0)
    uls_fig.add_trace(go.Scatter(
        x=depths,
        y=available_cm3,
        mode='markers',
        marker=dict(color=uls_colors, symbol=uls_symbols, size=10, line=dict(color='black', width=1)),
        text=uls_hover,
        hoverinfo='text'
    ))
    uls_fig.update_layout(
        title=(f"{plot_material} ULS Design ({ULS_case})<br>"
               f"WL: {wind_pressure:.2f} kPa, Bay: {bay} mm, L: {L} mm, BL: {selected_barrier_load:.2f} kN/m<br>"
               f"Req. Z: {Z_req_cm3:.1f} cm³"),
        xaxis_title="Section Depth (mm)",
        yaxis_title="Section Modulus (cm³)",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[0, uls_ymax])
    )

    # ----- SLS Plot -----
    E = material_props[plot_material]["E"]
    defl_values = []
    sls_hover = []
    for i in range(len(depths)):
        d_wl = (5 * w * L**4) / (384 * E * Iyy_vals[i])
        F_BL = selected_barrier_load * bay
        d_bl = ((F_BL * barrier_L) / (12 * E * Iyy_vals[i])) * (0.75 * L**2 - barrier_L**2)
        defl_total = d_wl if SLS_case.startswith("SLS 1") else d_bl
        defl_values.append(defl_total)
        sls_hover.append(
            f"{profiles[i]}<br>Supplier: {supps[i]}<br>Depth: {depths[i]} mm<br>Defl: {defl_total:.2f} mm<br>"
            f"SLS: {'Pass' if defl_total <= defl_limit else 'Fail'}"
        )
    sls_ymax = 1.33 * defl_limit
    valid = np.where(uls_passed)[0]
    sls_fig = go.Figure()
    sls_fig.add_shape(type="rect",
                      x0=x_min, x1=x_max,
                      y0=0, y1=defl_limit,
                      fillcolor=TT_LightBlue, opacity=0.2, line_width=0)
    sls_fig.add_shape(type="rect",
                      x0=x_min, x1=x_max,
                      y0=defl_limit, y1=sls_ymax,
                      fillcolor=TT_MidBlue, opacity=0.2, line_width=0)
    sls_fig.add_trace(go.Scatter(
        x=depths[valid],
        y=np.array(defl_values)[valid],
        mode='markers',
        marker=dict(
            color=['seagreen' if np.array(defl_values)[i] <= defl_limit else 'darkred' for i in valid],
            symbol=[ 'square' if reinf[i] else 'circle' for i in valid],
            size=10,
            line=dict(color='black', width=1)
        ),
        text=np.array(sls_hover)[valid],
        hoverinfo='text'
    ))
    sls_fig.update_layout(
        title=(f"{plot_material} SLS Design ({SLS_case})<br>"
               f"WL: {wind_pressure:.2f} kPa, Bay: {bay} mm, L: {L} mm, BL: {selected_barrier_load:.2f} kN/m<br>"
               f"Defl Limit: {defl_limit:.1f} mm"),
        xaxis_title="Section Depth (mm)",
        yaxis_title="Deflection (mm)",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[0, sls_ymax])
    )

    # ----- 3D Utilisation Plot -----
    uls_util, sls_util, depths_3d = [], [], []
    safe_suppliers, safe_profiles = [], []
    for i in range(len(depths)):
        if available_cm3[i] == 0:
            continue
        ratio_uls = Z_req_cm3 / available_cm3[i]
        ratio_sls = defl_values[i] / defl_limit if defl_limit != 0 else np.inf
        if ratio_uls <= 1 and ratio_sls <= 1:
            uls_util.append(ratio_uls)
            sls_util.append(ratio_sls)
            depths_3d.append(depths[i])
            safe_suppliers.append(supps[i])
            safe_profiles.append(profiles[i])
    if len(uls_util) > 0:
        d_arr = np.sqrt(np.array(uls_util)**2 + np.array(sls_util)**2)
        sizes = 3 + (d_arr / np.sqrt(2)) * 27
    else:
        sizes = 30

    recommended_text = "No suitable profile - choose a custom one!"
    if len(depths_3d) > 0:
        min_depth_val = min(depths_3d)
        indices = [i for i, d in enumerate(depths_3d) if d == min_depth_val]
        if indices:
            d_array = np.sqrt(np.array(uls_util)**2 + np.array(sls_util)**2)
            rec_index = indices[0] if len(indices)==1 else indices[np.argmax(d_array[indices])]
            recommended_text = f"Recommended Profile: {safe_suppliers[rec_index]}: {safe_profiles[rec_index]}"

    util_fig = go.Figure(data=[go.Scatter3d(
        x=uls_util,
        y=sls_util,
        z=depths_3d,
        mode='markers',
        marker=dict(
            size=sizes,
            color=depths_3d,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Depth (mm)")
        ),
        text=[f"{safe_suppliers[i]}: {safe_profiles[i]}<br>Depth: {depths_3d[i]} mm<br>"
              f"ULS Util: {uls_util[i]:.2f}<br>SLS Util: {sls_util[i]:.2f}"
              for i in range(len(depths_3d))],
        hoverinfo='text'
    )])
    util_fig.update_layout(
        title=f"3D Utilisation Plot<br>{recommended_text}",
        scene=dict(
            xaxis_title="ULS Utilisation",
            yaxis_title="SLS Utilisation",
            zaxis_title="Section Depth (mm)"
        )
    )
    if view_3d_option == "Isometric: Overview":
        camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))
    elif view_3d_option == "XY Plane: Utilisation":
        camera = dict(eye=dict(x=0, y=0, z=2.5))
    elif view_3d_option == "XZ Plane: Section Depth":
        camera = dict(eye=dict(x=1.25, y=0, z=1.25))
    util_fig.update_layout(scene_camera=camera)

    return uls_fig, sls_fig, util_fig

# Generate the Plotly figures.
uls_fig, sls_fig, util_fig = generate_plots()

# ---------------------------
# Layout: Arrange all three graphs on one line.
# ---------------------------
col1, col2, col3 = st.columns([1, 1, 1.5])
with col1:
    st.plotly_chart(uls_fig, height=500, use_container_width=True)
    pdf_uls = get_pdf_bytes(uls_fig)
    st.download_button("Download ULS PDF", data=pdf_uls, file_name="ULS_Design.pdf", mime="application/pdf")
with col2:
    st.plotly_chart(sls_fig, height=500, use_container_width=True)
    pdf_sls = get_pdf_bytes(sls_fig)
    st.download_button("Download SLS PDF", data=pdf_sls, file_name="SLS_Design.pdf", mime="application/pdf")
with col3:
    st.plotly_chart(util_fig, height=500, use_container_width=True)
    pdf_util = get_pdf_bytes(util_fig)
    st.download_button("Download 3D PDF", data=pdf_util, file_name="3D_Utilisation.pdf", mime="application/pdf")

# Optional: Merge all PDFs into one dashboard PDF.
all_pdf_bytes = merge_pdfs([pdf_uls, pdf_sls, pdf_util])
st.download_button("Download Full Dashboard PDF", data=all_pdf_bytes, file_name="Dashboard.pdf", mime="application/pdf")
