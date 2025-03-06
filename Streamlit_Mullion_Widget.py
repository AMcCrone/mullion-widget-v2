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
        st.error("Incorrect password. Do you not know your TT colours...? ;)")

# If the user is not authenticated, show the password input and halt the app.
if not st.session_state["authenticated"]:
    st.text_input("Enter Password:", type="password", key="password_input", on_change=check_password)
    st.stop()

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Section Design Tool", layout="wide")
st.title("Mullion Design Widget")

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
        marker=dict(color=uls_colors, symbol=uls_symbols, size=15, line=dict(color='black', width=1)),
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
        yaxis=dict(range=[0, uls_ymax]),
        height=650
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
            size=15,
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
        yaxis=dict(range=[0, sls_ymax]),
        height=650
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
        sizes = 10 + (d_arr / np.sqrt(2)) * 20
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
            colorscale='Emrld',
            colorbar=dict(title="Depth (mm)")
        ),
        text=[f"{safe_suppliers[i]}: {safe_profiles[i]}<br>Depth: {depths_3d[i]} mm<br>"
              f"ULS Util: {uls_util[i]:.2f}<br>SLS Util: {sls_util[i]:.2f}"
              for i in range(len(depths_3d))],
        hoverinfo='text'
    )])
    util_fig.update_layout(
        height=650,
        title=f"3D Utilisation Plot<br>{recommended_text}",
        scene=dict(
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0.0, 1.0]),
            zaxis=dict(range=[50, 1.05 * max(depths)]),
            xaxis_title="ULS Utilisation",
            yaxis_title="SLS Utilisation",
            zaxis_title="Section Depth (mm)"
        )
    )
    if view_3d_option == "Isometric: Overview":
        camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))
    elif view_3d_option == "XY Plane: Utilisation":
        camera = dict(
            eye=dict(x=0, y=0, z=2.5),
            projection=dict(type='orthographic')
        )
    elif view_3d_option == "XZ Plane: Section Depth":
        camera = dict(
            eye=dict(x=0, y=2.5, z=0),
            projection=dict(type='orthographic')
        )
    util_fig.update_layout(scene_camera=camera)

    return uls_fig, sls_fig, util_fig, defl_values, Z_req_cm3, defl_limit

# Generate the Plotly figures.
uls_fig, sls_fig, util_fig, defl_values, Z_req_cm3, defl_limit = generate_plots()

# ---------------------------
# Layout: Arrange all three graphs on one line.
# ---------------------------
col1, col2, col3 = st.columns([1, 1, 1.5])
with col1:
    st.plotly_chart(uls_fig, height=650, use_container_width=True)
    pdf_uls = get_pdf_bytes(uls_fig)
    st.download_button("Download ULS PDF", data=pdf_uls, file_name="ULS_Design.pdf", mime="application/pdf")
with col2:
    st.plotly_chart(sls_fig, height=650, use_container_width=True)
    pdf_sls = get_pdf_bytes(sls_fig)
    st.download_button("Download SLS PDF", data=pdf_sls, file_name="SLS_Design.pdf", mime="application/pdf")
with col3:
    st.plotly_chart(util_fig, height=650, use_container_width=True)
    pdf_util = get_pdf_bytes(util_fig)
    st.download_button("Download Utilisation PDF", data=pdf_util, file_name="3D_Utilisation.pdf", mime="application/pdf")

# -----------------------------------------------------------------------------
# Append Table of Profiles Sorted by Utilisation (No Conditional Colors)
# -----------------------------------------------------------------------------
st.title("Section Database")

# Compute utilisation ratios for each profile in df_mat.
df_mat = df_selected[(df_selected["Material"] == plot_material) & 
                     (df_selected["Supplier"].isin(selected_suppliers))].copy()
df_mat.reset_index(drop=True, inplace=True)

# Calculate ULS utilisation based on the available section modulus (converted to cm³)
df_mat["ULS Utilisation"] = Z_req_cm3 / (df_mat["Wyy"] / 1000)

# Recompute deflection for each row (using the same approach as in generate_plots)
E = material_props[plot_material]["E"]
defl_values_table = []
for i, row in df_mat.iterrows():
    Iyy_val = row["Iyy"]
    d_wl = (5 * wind_pressure * 0.001 * bay_width * mullion_length**4) / (384 * E * Iyy_val)
    F_BL = selected_barrier_load * bay_width
    d_bl = ((F_BL * barrier_L) / (12 * E * Iyy_val)) * (0.75 * mullion_length**2 - barrier_L**2)
    defl_total = d_wl if SLS_case.startswith("SLS 1") else d_bl
    defl_values_table.append(defl_total)
df_mat["SLS Utilisation"] = np.array(defl_values_table) / defl_limit

# Create a 'Max Utilisation' for sorting purposes
df_mat["Max Utilisation"] = df_mat[["ULS Utilisation", "SLS Utilisation"]].max(axis=1)
# Sort profiles so that the best-performing (lowest utilisation) are at the top.
df_sorted = df_mat.sort_values(by="Max Utilisation", ascending=True)

row_colors = []
for _, row in df_sorted.iterrows():
    # If both utilisation checks pass (<= 1), mark as blue; otherwise, grey.
    if (row["ULS Utilisation"] <= 1) and (row["SLS Utilisation"] <= 1):
        row_colors.append("#E7F8F9")  # blue for passing sections
    else:
        row_colors.append("#DFE0E1")  # grey for failing sections

# Create the Plotly Table.
table_fig = go.Figure(data=[go.Table(
    header=dict(
        values=["Supplier", "Profile Name", "Depth (mm)", "Section Modulus (cm³)", "I (cm⁴)", "ULS Util. (%)", "SLS Util. (%)"],
        fill_color=TT_DarkBlue,
        font=dict(color="white", size=18),
        align="center"
    ),
    cells=dict(
        values=[
            df_sorted["Supplier"],
            df_sorted["Profile Name"],
            df_sorted["Depth"],
            (df_sorted["Wyy"] / 1000).round(2),
            (df_sorted["Iyy"] / 10000).round(2),
            (df_sorted["ULS Utilisation"] * 100).round(1).astype(str) + '%',  # ULS Utilisation as percentage
            (df_sorted["SLS Utilisation"] * 100).round(1).astype(str) + '%'   # SLS Utilisation as percentage
        ],
        fill_color=[row_colors] * 7,  # assign the computed row colors to all columns
        font=dict(size=14), 
        align="center"
    )
)])

st.plotly_chart(table_fig, use_container_width=True)


# ---------------------------
# Text and Documentation
# ---------------------------
st.title("The Boring Stuff...")
st.markdown("The following text describes the documentation, limitations, and formulae used in the creation of this **Mullion Check Widget**")
st.header("Stress Calculations")
st.subheader("Wind Load (WL)")
st.latex(r'''
    w_{WL} = q_{WL}W_{bay}
    ''')
st.markdown("Where $w_{WL}$ is the effective UDL on the mullion for a given wind load, $q_{WL}$, and bay width, $W_{bay}$. The maximum bending moment - at midspan - is calculated through:")
st.latex(r'''
    M_{WL,max} = \frac{w_{WL}L^2}{8}
    ''')
st.markdown("Where $L$ is the total span of the mullion.")

st.subheader("Barrier Load (WL)")
st.latex(r'''
    P_{BL} = w_{WL}W_{bay}
    ''')
st.markdown("Where $P_{BL}$ is the effective point load on the mullion at 1100mm from its base for a given barrier load, $w_{BL}$, and bay width, $W_{bay}$. The maximum bending moment - at midspan - is calculated through:")
st.latex(r'''
    M_{BL,max} = \frac{P_{BL}Lx}{2} = \frac{P_{BL}L^2}{4}
    ''')

st.subheader("Stress Limit")
st.markdown("The required section modulus is calculated through:")
st.latex(r'''
    \sigma_{y} = \frac{My}{I} = \frac{M}{Z_{req}} 
    ''')
st.latex(r'''
    Z_{req} = \frac{M}{\sigma_{y}} 
    ''')
st.markdown("Where $\sigma_{y}$ is the yield stress of the material (aluminium or steel), $y$ is the distance of a sections' extreme fibre from its centroid, and $Z_{req}$ is the required section modulus given bending moment $M$. With:")
st.latex(r'''
    M = \alpha M_{WL,max} + \beta M_{BL,max}
    ''')
st.markdown("With combination factors α and β depending on the load case")

st.header("Deflection Calculations")
st.subheader("Wind Load (WL)")
st.markdown("For the UDL $w_{WL}$, the maximum deflection, $\delta_{WL}$, is at midspan with a magnitude of:")
st.latex(r'''
    \delta_{WL}=\frac{5w_{WL}L^4}{384EI}
    ''')
st.markdown("Where $E$ is the elastic modulus of the material (aluminium or steel) and $I$ is the section's second moment of area.")

st.subheader("Barrier Load (WL)")
st.markdown("For the point load $P_{BL}$ at barrier height $L_{BL}$, the deflection, $\delta_{BL}$, is calculated through:")
st.latex(r'''
    \delta_{BL}=\frac{P_{BL}}{12EI}\left(L^2-x^2-L_{BL}^2\right)
    ''')
st.markdown("Where $L_{BL}$ has been assumed to be 1100mm from the base of the mullion and the deflection taken at midspan ($x = L/2$) for superposition with $\delta_{WL}$. Thus:")
st.latex(r'''
    \delta_{BL}=\frac{P_{BL}}{12EI}\left(\frac{3}{4}L^2-L_{BL}^2\right)
    ''')
st.subheader("Deflection Limits")
st.markdown("The deflection limits are as set out in CWCT doc XX...")
st.latex(r'''
    \delta_{lim} = \begin{cases}
                       L/200 &\text{if } L \leq 3000 \text{mm} \\
                       5 + L/300 &\text{if } 3000 < L \leq 7500 \text{mm} \\
                       L/250 &\text{if } L > 7500 \text{mm}
                   \end{cases}
                   ''')

st.header("Load Cases")
st.markdown("The following load cases have selected following guidance from *CWCT Guidance on non-loadbearing building envelopes*")
st.latex(r'''
    \text{Load Case 1:} 1.35DL + 1.5WL + 0.5*1.5BL
    \text{Load Case 2:} 1.35DL + 0.5*1.5WL + 1.5BL
    \text{Load Case 3:} 1.35DL + 1.5WL
    \text{Load Case 4:} 1.35DL + 1.5BL
                   ''')
st.header("Utilisation")
st.markdown("Utilisation is calculated as...")
