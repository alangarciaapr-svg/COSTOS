import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="ForestCost Excel Pro",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Ajustado para parecer Excel
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    /* Ajustes para la tabla editable */
    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_excel_final_v12.json'

# --- 2. GESTIÓN DE PERSISTENCIA (CORREGIDA PARA DATAFRAMES) ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    # Convertimos DataFrames a diccionarios antes de guardar
    config_data = {}
    for k, v in st.session_state.items():
        if k in EXPECTED_KEYS:
            if isinstance(v, pd.DataFrame):
                config_data[k] = v.to_dict('records') # Serializar tabla
            elif isinstance(v, (int, float, str, bool, list, dict)):
                config_data[k] = v
                
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f)

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    # Tablas Completas
    "df_harvester_data", "df_forwarder_data", "df_indirect_data",
    # Simulador
    "sim_m3_h", "sim_m3_f", "alloc_method", "h_share_pct_manual"
]

# Carga Inicial Segura
if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            # Reconstruir DataFrames
            if key in ["df_harvester_data", "df_forwarder_data", "df_indirect_data"]:
                st.session_state[key] = pd.DataFrame(val)
            else:
                st.session_state[key] = val
    st.session_state['config_loaded'] = True

# --- 3. UTILIDADES ---
def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:,.0f}".replace(",", ".")
    return x

def card(title, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="font-size:12px; color:#888;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_uf_value():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['serie'][0]['valor'], data['serie'][0]['fecha'][:10]
    except:
        pass
    return None, None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Parámetros Base")
    
    st.markdown("### 1. Economía")
    uf_api, fecha_api = get_uf_value()
    use_auto_uf = st.checkbox("UF Automática", value=True, key="use_auto_uf", on_change=save_config)
    if use_auto_uf and uf_api:
        current_uf = uf_api
        st.success(f"UF Hoy: ${fmt(current_uf)}")
    else:
        current_uf = st.number_input("Valor UF", value=39704.93, step=100.0, key="uf_manual", on_change=save_config)

    fuel_price = st.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

    st.markdown("### 2. Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)
    
    st.markdown("### 3. Distribución Ingreso")
    h_rev_pct = st.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    f_rev_pct = 100 - h_rev_pct
    
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * (f_rev_pct / 100)
    
    st.info(f"H: ${fmt(h_income)} | F: ${fmt(f_income)}")

# --- 5. LOGICA PRINCIPAL ---

st.title("🌲 Planilla de Costos Forestales (Detalle Excel)")

# --- A. JORNADA ---
with st.expander("📅 Configuración de Jornada", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", 28, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", 10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
        st.caption(f"Total: {fmt(h_total_hours)} Hrs")
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", 25, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", 9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day
        st.caption(f"Total: {fmt(f_total_hours)} Hrs")

st.divider()

# --- B. TABLAS DE COSTOS (EDITABLES) ---

col_tab1, col_tab2 = st.columns(2)

def create_machine_table(prefix, col_obj, total_hours, fuel_p):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        
        # Estructura inicial basada en tus CSVs
        initial_data = [
            {"Categoría": "Fijos", "Ítem": "Arriendo Mensual", "Valor Input": 10900000 if prefix=="Harvester" else 8000000, "Unidad": "$/Mes"},
            {"Categoría": "Fijos", "Ítem": "Operador Turno 1", "Valor Input": 1923721, "Unidad": "$/Mes"},
            {"Categoría": "Fijos", "Ítem": "Operador Turno 2", "Valor Input": 1923721 if prefix=="Harvester" else 0, "Unidad": "$/Mes"}, # Forwarder a veces 1 turno
            {"Categoría": "Fijos", "Ítem": "Seguro / Otros", "Valor Input": 750000, "Unidad": "$/Mes"},
            
            {"Categoría": "Variable", "Ítem": "Petróleo (Consumo)", "Valor Input": 20.0 if prefix=="Harvester" else 15.0, "Unidad": "Litros/Hora"},
            
            {"Categoría": "Mantención", "Ítem": "Mant. Programada (600/1200h)", "Valor Input": 127000, "Unidad": "$/Mes (Prom)"},
            {"Categoría": "Mantención", "Ítem": "Mant. Correctiva/Repuestos", "Valor Input": 400000, "Unidad": "$/Mes"},
            {"Categoría": "Mantención", "Ítem": "Hidráulica / Mangueras", "Valor Input": 910000 if prefix=="Harvester" else 500000, "Unidad": "$/Mes"},
            {"Categoría": "Mantención", "Ítem": "Neumáticos / Rodado", "Valor Input": 280000, "Unidad": "$/Mes"},
            {"Categoría": "Mantención", "Ítem": "Grasas y Lubricantes", "Valor Input": 80000, "Unidad": "$/Mes"},
            
            {"Categoría": "Consumibles", "Ítem": "Cadenas / Elementos Corte", "Valor Input": 30000, "Unidad": "$/Mes"},
            {"Categoría": "Consumibles", "Ítem": "Espadas / Barras", "Valor Input": 130000, "Unidad": "$/Mes"},
            {"Categoría": "Consumibles", "Ítem": "Aceite Hidráulico", "Valor Input": 160000, "Unidad": "$/Mes"},
            
            {"Categoría": "Otros", "Ítem": "Reservas / Overhaul", "Valor Input": 330000, "Unidad": "$/Mes"},
        ]
        
        key_df = f"df_{prefix.lower()}_data"
        if key_df not in st.session_state:
            st.session_state[key_df] = pd.DataFrame(initial_data)
            
        # TABLA EDITABLE
        st.caption("Modifica la columna 'Valor Input'. Si la unidad es 'Litros/Hora', se multiplicará por el precio del petróleo.")
        edited_df = st.data_editor(
            st.session_state[key_df],
            key=f"editor_{prefix}",
            column_config={
                "Categoría": st.column_config.TextColumn(disabled=True),
                "Ítem": st.column_config.TextColumn(disabled=True),
                "Unidad": st.column_config.TextColumn(disabled=True),
                "Valor Input": st.column_config.NumberColumn("Valor Input", format="%d", required=True),
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Actualizar Session State
        st.session_state[key_df] = edited_df
        save_config() # Guardar cambios al archivo
        
        # --- CÁLCULOS INTERNOS ---
        total_month_sum = 0
        
        # Iteramos para calcular el costo mensual real de cada fila
        # Si es L/h -> Input * Horas * Precio
        # Si es $/Mes -> Input
        calc_rows = []
        
        for index, row in edited_df.iterrows():
            val = row["Valor Input"]
            unit = row["Unidad"]
            
            cost_month = 0
            if unit == "Litros/Hora":
                cost_month = val * total_hours * fuel_p
            else:
                cost_month = val
            
            total_month_sum += cost_month
            
        total_hr = total_month_sum / total_hours if total_hours else 0
        
        # Tarjeta Resumen Máquina
        st.markdown(f"""
        <div style="background-color:#e6fffa; padding:10px; border-radius:5px; border:1px solid #b2f5ea; text-align:center;">
            <div style="font-size:14px; color:#2c7a7b;">Costo Total {prefix}</div>
            <div style="font-size:20px; font-weight:bold; color:#234e52;">${fmt(total_hr)} /hora</div>
            <div style="font-size:12px; color:#285e61;">Mensual: ${fmt(total_month_sum)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        return total_month_sum, total_hr

h_total_m, h_total_hr = create_machine_table("Harvester", col_tab1, h_total_hours, fuel_price)
f_total_m, f_total_hr = create_machine_table("Forwarder", col_tab2, f_total_hours, fuel_price)

# --- C. INDIRECTOS Y FAENA (TABLA) ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos (Faena)")

key_ind = "df_indirect_data"
if key_ind not in st.session_state:
    st.session_state[key_ind] = pd.DataFrame([
        {"Ítem": "Arriendo Camionetas (UF)", "Valor": 38.0, "Tipo": "UF Mensual"},
        {"Ítem": "Combustible Camionetas (L/Día)", "Valor": 12.0, "Tipo": "Litros Diario"},
        {"Ítem": "Días Uso Camioneta", "Valor": 30.0, "Tipo": "Días Mes"},
        {"Ítem": "Personal Apoyo ($)", "Valor": 2164000.0, "Tipo": "$ Mensual"},
        {"Ítem": "Instalaciones/Gastos ($)", "Valor": 560000.0, "Tipo": "$ Mensual"},
        {"Ítem": "Pensión/Alojamiento ($)", "Valor": 1890000.0, "Tipo": "$ Mensual"},
        {"Ítem": "Otros Faena ($)", "Valor": 100000.0, "Tipo": "$ Mensual"},
    ])

with st.expander("📝 Editar Costos Indirectos", expanded=True):
    edited_ind = st.data_editor(
        st.session_state[key_ind],
        key="editor_indirect",
        hide_index=True,
        use_container_width=True
    )
    st.session_state[key_ind] = edited_ind
    save_config()

# Procesar Indirectos
def get_ind_val(name):
    row = edited_ind[edited_ind["Ítem"] == name]
    return float(row.iloc[0]["Valor"]) if not row.empty else 0.0

uf_rent_val = get_ind_val("Arriendo Camionetas (UF)")
fuel_day_val = get_ind_val("Combustible Camionetas (L/Día)")
days_use = get_ind_val("Días Uso Camioneta")
staff_val = get_ind_val("Personal Apoyo ($)")
fac_val = get_ind_val("Instalaciones/Gastos ($)")
pen_val = get_ind_val("Pensión/Alojamiento ($)")
oth_val = get_ind_val("Otros Faena ($)")

# Cálculo Faena
cost_rent_clp = uf_rent_val * current_uf
cost_fuel_clp = fuel_day_val * days_use * fuel_price
total_shared = cost_rent_clp + cost_fuel_clp + staff_val + fac_val + pen_val + oth_val

st.success(f"**TOTAL GASTO INDIRECTO: ${fmt(total_shared)} /Mes**")

# Asignación
st.markdown("**Distribución Indirectos**")
alloc_opts = ["Manual", "Proporcional Horas"]
idx = alloc_opts.index(st.session_state.get("alloc_method", "Manual")) if st.session_state.get("alloc_method") in alloc_opts else 0
alloc = st.radio("", alloc_opts, index=idx, key="alloc_method", horizontal=True, on_change=save_config)

if alloc == "Manual":
    h_pct = st.slider("% Asignado Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
else:
    tt = h_total_hours + f_total_hours
    h_pct = h_total_hours / tt if tt > 0 else 0.5

shared_h = total_shared * h_pct
shared_f = total_shared * (1 - h_pct)

# --- D. RESULTADOS FINALES ---
st.divider()
st.subheader("📊 Resultados Consolidados")

# Costos Finales Hora
final_h_hr = (h_total_m + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_m + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

c1, c2, c3 = st.columns(3)
with c1: card("Costo Harvester", f"${fmt(final_h_hr)}/hr", "Inc. Indirectos")
with c2: card("Costo Forwarder", f"${fmt(final_f_hr)}/hr", "Inc. Indirectos")
with c3: card("Costo Sistema", f"${fmt(sys_hr)}/hr", "Total Operacional")

# --- SIMULADOR ---
st.markdown("### 🧮 Simulador de Rentabilidad")

col_sim_in, col_sim_out = st.columns([1, 2])

with col_sim_in:
    st.markdown("**Producción Estimada**")
    m3_h = st.number_input("M3/Hr Harvester", value=25.0, step=0.5, key="sim_m3_h", on_change=save_config)
    m3_f = st.number_input("M3/Hr Forwarder", value=28.0, step=0.5, key="sim_m3_f", on_change=save_config)
    
    mr_h = m3_h / conversion_factor if conversion_factor else 0
    mr_f = m3_f / conversion_factor if conversion_factor else 0
    
    st.caption(f"H: {mr_h:.1f} MR | F: {mr_f:.1f} MR")

with col_sim_out:
    # Lógica de cálculo
    mr_sys = min(mr_h, mr_f) # Bottleneck
    
    cost_sys_unit = sys_hr / mr_sys if mr_sys else 0
    util_unit = sales_price_mr - cost_sys_unit
    margen_pct = (util_unit / sales_price_mr * 100) if sales_price_mr else 0
    
    # Tarjeta Principal
    color_m = "green" if util_unit > 0 else "red"
    st.markdown(f"""
    <div style="background-color:#f8f9fa; border:1px solid #ddd; border-radius:10px; padding:20px; text-align:center;">
        <h3 style="margin:0; color:#333;">Margen del Sistema Real</h3>
        <div style="font-size:36px; font-weight:bold; color:{color_m};">{margen_pct:.1f}%</div>
        <div style="color:#555;">Utilidad: <b>${fmt(util_unit)}</b> / MR</div>
        <hr>
        <div style="font-size:13px;">Basado en producción real (Limitada por el más lento): <b>{mr_sys:.1f} MR/hr</b></div>
    </div>
    """, unsafe_allow_html=True)

# Tabla de detalle por máquina
st.markdown("##### Detalle por Máquina")
res_data = pd.DataFrame([
    {"Máquina": "Harvester", "Prod (MR/h)": mr_h, "Ingreso": h_income, "Costo": (final_h_hr/mr_h) if mr_h else 0},
    {"Máquina": "Forwarder", "Prod (MR/h)": mr_f, "Ingreso": f_income, "Costo": (final_f_hr/mr_f) if mr_f else 0},
])
res_data["Utilidad"] = res_data["Ingreso"] - res_data["Costo"]
res_data["Margen %"] = (res_data["Utilidad"] / res_data["Ingreso"] * 100)

st.dataframe(res_data, column_config={
    "Ingreso": st.column_config.NumberColumn(format="$%d"),
    "Costo": st.column_config.NumberColumn(format="$%d"),
    "Utilidad": st.column_config.NumberColumn(format="$%d"),
    "Margen %": st.column_config.ProgressColumn(format="%.1f%%", min_value=-0.5, max_value=0.5)
}, hide_index=True, use_container_width=True)
