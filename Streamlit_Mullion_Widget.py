import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ---------------------------
# Settings and Color Palette
# ---------------------------
small_label = {'fontsize': 8}
small_title = {'fontsize': 10}

# TT Colours
TT_Orange = (0.82745, 0.27059, 0.11373)
TT_Olive = (0.5451, 0.56471, 0.39216)
TT_LightBlue = (0.53333, 0.85882, 0.87451)
TT_MidBlue = (0, 0.63922, 0.67843)
TT_DarkBlue = (0, 0.18824, 0.23529)
TT_Grey = (0.38824, 0.4, 0.41176)

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Section Design Tool", layout="wide")
st.title("Section Design Tool")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")

# 1. Material Selection
plot_material = st.sidebar.selectbox("Select Material", options=["Steel", "Aluminium"], index=0)

# 2. Upload Excel Database
st.sidebar.subheader("Upload Cross Sections Database")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        SHEET = "Alu Mullion Database" if plot_material == "Aluminium" else "Steel Mullion Database"
        df = pd.read_excel(uploaded_file, sheet_name=SHEET, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()
else:
    st.info("Please upload the Cross Sections Database Excel file.")
    st.stop()

# 3. Process the DataFrame
selected_columns = ["Supplier", "Profile Name", "Material", "Reinf", "Depth", "Iyy", "Wyy"]
df_selected = df[selected_columns]
df_selected = df_selected.iloc[1:].reset_index(drop=True)  # Remove first row (units)

# 4. Define Material Properties
material_props = {
    "Aluminium": {"fy": 160, "E": 70000},
    "Steel": {"fy": 355, "E": 210000}
}

# 5. Supplier Selection (multiselect)
suppliers_all = sorted(df_selected["Supplier"].unique())
selected_suppliers = st.sidebar.multiselect("Select Suppliers", options=suppliers_all, default=suppliers_all)

# 6. Option to Show Data Labels on Plots
show_data_labels = st.sidebar.checkbox("Show Data Labels", value=False)

# 7. Barrier Load Selection
barrier_load_option = st.sidebar.radio("Barrier Load (kN/m)", options=["None", "0.74", "1.5", "3"], index=0)
selected_barrier_load = 0 if barrier_load_option == "None" else float(barrier_load_option)

# 8. ULS and SLS Load Cases
ULS_case = st.sidebar.radio("ULS Load Case", options=[
    "ULS 1: 1.5WL + 0.75BL", 
    "ULS 2: 0.75WL + 1.5BL", 
    "ULS 3: 1.5WL", 
    "ULS 4: 1.5BL"
], index=0)

SLS_case = st.sidebar.radio("SLS Load Case", options=["SLS 1: WL", "SLS 2: BL"], index=0)

# 9. 3D View Options
view_3d_option = st.sidebar.radio("3D View", options=[
    "Isometric: Overview", 
    "XY Plane: Utilisation", 
    "XZ Plane: Section Depth"
], index=0)

# 10. Sliders for Wind Pressure, Bay Width, and Mullion Length
wind_pressure = st.sidebar.slider("Wind Pressure (kPa)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
bay_width = st.sidebar.slider("Bay Width (mm)", min_value=2000, max_value=10000, value=3000, step=500)
mullion_length = st.sidebar.slider("Mullion Length (mm)", min_value=2500, max_value=12000, value=4000, step=250)

# Constant Barrier Length (mm)
barrier_L = 1100

# ---------------------------
# Figure Generation Function
# ---------------------------
def generate_figure():
    # Create a figure with three subplots (ULS, SLS, 3D Utilisation)
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.15)
    ax_ULS = fig.add_subplot(gs[0, 0])
    ax_SLS = fig.add_subplot(gs[0, 1])
    ax_3d  = fig.add_subplot(gs[0, 2], projection='3d')
    fig.suptitle(f"{plot_material} Design", fontsize=11)
    fig.subplots_adjust(left=0.2, bottom=0.33, right=0.95, top=0.89)

    # Retrieve slider values and compute basic forces/moments
    p_kPa = wind_pressure
    p = p_kPa * 0.001         # Convert kPa to N/mm²
    bay = bay_width           # mm
    L = mullion_length        # mm
    w = p * bay               # Uniform wind load in N/mm

    M_WL = (w * L**2) / 8     # Bending moment from wind load (N·mm)
    M_BL = ((selected_barrier_load * bay) * barrier_L) / 2  # Bending moment from barrier load (N·mm)

    # ULS moment based on selected load case:
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

    # For worst-case detection (not dynamically updated in Streamlit)
    M1 = 1.5 * M_WL + 0.75 * M_BL  # ULS 1
    M2 = 0.75 * M_WL + 1.5 * M_BL  # ULS 2
    M3 = 1.5 * M_WL                # ULS 3
    M4 = 1.5 * M_BL                # ULS 4
    uls_values = [M1, M2, M3, M4]
    worst_index = np.argmax(uls_values)
    # (ULS labels are fixed by the sidebar selection.)

    # Calculate required section modulus (ULS)
    fy = material_props[plot_material]["fy"]
    Z_req = M_ULS / fy          # in mm³
    Z_req_cm3 = Z_req / 1000     # in cm³

    # Deflection limit calculation:
    if L <= 3000:
        defl_limit = L / 200
    elif L < 7500:
        defl_limit = 5 + L / 300
    else:
        defl_limit = L / 250

    # Filter the dataframe by material and selected suppliers
    df_mat = df_selected[(df_selected["Material"] == plot_material) & 
                         (df_selected["Supplier"].isin(selected_suppliers))]
    if df_mat.empty:
        for ax in [ax_ULS, ax_SLS, ax_3d]:
            ax.clear()
            ax.set_title("No sections selected", **small_title)
        return fig

    # Extract section data
    depths   = df_mat["Depth"].values  # mm
    Wyy_vals = df_mat["Wyy"].values     # mm³
    Iyy_vals = df_mat["Iyy"].values     # mm⁴
    profiles = df_mat["Profile Name"].values
    reinf    = df_mat["Reinf"].values
    supps    = df_mat["Supplier"].values

    available_cm3 = Wyy_vals / 1000  # Convert to cm³

    # ------------------------------
    # ULS 2D Plot
    # ------------------------------
    ax_ULS.clear()
    x_min = np.min(depths) * 0.95
    x_max = np.max(depths) * 1.05
    ax_ULS.set_xlim(x_min, x_max)
    ax_ULS.set_ylim(0, 4 * Z_req_cm3)
    ax_ULS.axhspan(Z_req_cm3, ax_ULS.get_ylim()[1], facecolor=TT_LightBlue, alpha=0.2, zorder=0)
    ax_ULS.axhspan(0, Z_req_cm3, facecolor=TT_MidBlue, alpha=0.2, zorder=0)
    uls_passed = []
    for i in range(len(depths)):
        x = depths[i]
        avail = available_cm3[i]
        uls_pass = (avail >= Z_req_cm3)
        uls_passed.append(uls_pass)
        color = 'seagreen' if uls_pass else 'darkred'
        marker = 's' if reinf[i] else 'o'
        ax_ULS.scatter(x, avail, c=color, marker=marker, s=100, edgecolors='black', zorder=3)
        if show_data_labels:
            label_text = (f"{profiles[i]}\nSupplier: {supps[i]}\nDepth: {depths[i]} mm\n"
                          f"Z: {avail:.2f} cm³\nULS: {'Pass' if uls_pass else 'Fail'}")
            ax_ULS.annotate(label_text, (x, avail), fontsize=6)
    ax_ULS.set_xlabel("Section Depth (mm)", **small_label)
    ax_ULS.set_ylabel("Section Modulus (cm³)", **small_label)
    ax_ULS.set_title(f"{plot_material} ULS Design ({ULS_case})\n"
                     f"WL: {p_kPa:.2f} kPa, Bay: {bay:.0f} mm, L: {L:.0f} mm, BL: {selected_barrier_load:.2f} kN/m\n"
                     f"$\\mathbf{{Req.\\ Z:\\ {Z_req_cm3:.1f}\\ cm^3}}$", fontsize=9)

    # ------------------------------
    # SLS 2D Plot
    # ------------------------------
    ax_SLS.clear()
    E = material_props[plot_material]["E"]
    sls_defl_wl = []
    sls_defl_bl = []
    for i in range(len(depths)):
        d_wl = (5 * w * L**4) / (384 * E * Iyy_vals[i])
        sls_defl_wl.append(d_wl)
        F_BL = selected_barrier_load * bay
        d_bl = ((F_BL * barrier_L) / (12 * E * Iyy_vals[i])) * (0.75 * L**2 - barrier_L**2)
        sls_defl_bl.append(d_bl)
    max_defl_wl = max(sls_defl_wl) if sls_defl_wl else 0
    max_defl_bl = max(sls_defl_bl) if sls_defl_bl else 0
    worst_sls_index = 0 if max_defl_wl > max_defl_bl else 1
    sls_load_cases = ["SLS 1: WL", "SLS 2: BL"]
    defl_values = []
    for i in range(len(depths)):
        defl_wl = (5 * w * L**4) / (384 * E * Iyy_vals[i])
        F_BL = selected_barrier_load * bay
        defl_bl = ((F_BL * barrier_L) / (12 * E * Iyy_vals[i])) * (0.75 * L**2 - barrier_L**2)
        defl_total = defl_wl if SLS_case.startswith("SLS 1") else defl_bl
        defl_values.append(defl_total)
        sls_pass = (defl_total <= defl_limit)
        color = 'seagreen' if sls_pass else 'darkred'
        marker = 's' if reinf[i] else 'o'
        if uls_passed[i]:
            ax_SLS.scatter(depths[i], defl_total, c=color, marker=marker, s=100, edgecolors='black', zorder=3)
            if show_data_labels:
                label_text = (f"{profiles[i]}\nSupplier: {supps[i]}\nDepth: {depths[i]} mm\n"
                              f"Defl: {defl_total:.2f} mm\nSLS: {'Pass' if sls_pass else 'Fail'}")
                ax_SLS.annotate(label_text, (depths[i], defl_total), fontsize=6)
    y_slmin = 0
    y_slmax = 1.33 * defl_limit
    ax_SLS.set_ylim(y_slmin, y_slmax)
    ax_SLS.set_xlim(x_min, x_max)
    ax_SLS.axhspan(y_slmin, defl_limit, facecolor=TT_LightBlue, alpha=0.2, zorder=0)
    ax_SLS.axhspan(defl_limit, y_slmax, facecolor=TT_MidBlue, alpha=0.2, zorder=0)
    ax_SLS.set_xlabel("Section Depth (mm)", **small_label)
    ax_SLS.set_ylabel("Deflection (mm)", **small_label)
    ax_SLS.set_title(f"{plot_material} SLS Design ({SLS_case})\n"
                     f"WL: {p_kPa:.2f} kPa, Bay: {bay:.0f} mm, L: {L:.0f} mm, BL: {selected_barrier_load:.2f} kN/m\n"
                     f"$\\mathbf{{Defl\\ Limit:\\ {defl_limit:.1f}\\ mm}}$", fontsize=9)

    # ------------------------------
    # 3D Utilisation Plot
    # ------------------------------
    uls_util = []
    sls_util = []
    depths_3d = []
    safe_suppliers = []
    safe_profiles = []
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
        sizes = 20 + (d_arr / np.sqrt(2)) * 480
    else:
        sizes = 500

    if depths_3d:
        dmin = np.min(depths_3d)
        dmax = np.max(depths_3d)
        norm = plt.Normalize(dmin, dmax)
        colors = cm.RdYlGn_r(norm(depths_3d))
    else:
        colors = 'blue'
    
    recommended_text = "No suitable profile - choose a custom one!"
    if len(depths_3d) > 0:
        min_depth = min(depths_3d)
        indices = [i for i, d in enumerate(depths_3d) if d == min_depth]
        if indices:
            rec_index = indices[0] if len(indices) == 1 else indices[np.argmax(np.sqrt(np.array(uls_util)**2 + np.array(sls_util)[indices]))]
            recommended_text = f"Recommended Profile: {safe_suppliers[rec_index]}: {safe_profiles[rec_index]}"
    
    ax_3d.clear()
    ax_3d.scatter(uls_util, sls_util, depths_3d, s=sizes, c=colors, marker='o')
    ax_3d.set_xlabel("ULS Utilisation", fontsize=8)
    ax_3d.set_ylabel("SLS Utilisation", fontsize=8)
    ax_3d.set_zlabel("Section Depth (mm)", fontsize=8)
    ax_3d.set_title("3D Utilisation Plot\n" + r"$\mathbf{" + recommended_text.replace(' ', r'\ ') + "}$", fontsize=9)
    ax_3d.set_xlim(0, 1)
    ax_3d.set_ylim(0, 1)
    ax_3d.set_zlim(x_min, x_max)

    # Adjust 3D view based on selection
    if view_3d_option == "Isometric: Overview":
        ax_3d.view_init(elev=30, azim=-45)
    elif view_3d_option == "XY Plane: Utilisation":
        ax_3d.view_init(elev=90, azim=-90)
    elif view_3d_option == "XZ Plane: Section Depth":
        ax_3d.view_init(elev=0, azim=0)

    # Add legends for the 3D plot
    color_legend = [
        Line2D([0], [0], marker='o', color='w', label='Shallow Section',
               markerfacecolor='seagreen', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Deeper Section',
               markerfacecolor='red', markersize=10)
    ]
    size_legend = [
        Line2D([0], [0], marker='o', color='w', label='Lower Utilisation',
               markerfacecolor='gray', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='Higher Utilisation',
               markerfacecolor='gray', markersize=15)
    ]
    leg1 = ax_3d.legend(handles=color_legend, title="Section Depth", fontsize=8, title_fontsize=8, loc='upper left')
    leg2 = ax_3d.legend(handles=size_legend, title="Utilisation", fontsize=8, title_fontsize=8, loc='upper right')
    ax_3d.add_artist(leg1)
    
    return fig

# ---------------------------
# Generate and Display the Figure
# ---------------------------
fig = generate_figure()
st.pyplot(fig)
