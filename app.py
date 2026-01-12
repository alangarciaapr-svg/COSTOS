import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="ForestCost Pro",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stDataEditor {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v19_ind_list.json'

# --- 2. PERSISTENCIA ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    config_data = {}
    for k, v in st.session_state.items():
        if k in EXPECTED_KEYS:
            if isinstance(v, pd.DataFrame):
                config_data[k] = v.to_dict('records')
            elif isinstance(v, (int, float, str, bool, list, dict)):
                config_data[k] = v
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, cls=NumpyEncoder)

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    "df_harvester_v17", "df_forwarder_v17", "df_indirect_list_v19",
    "alloc_method", "h_share_pct_manual", 
    "sim_m3_h_val", "sim_m3_f_val",
    "pickup_days_use" # Nuevo: Días uso camionetas
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            if key in ["df_harvester_v17", "df_forwarder_v17", "df_indirect_list_v19"]:
                st.session_state[key] = pd.DataFrame(val)
            else:
                st.session_state[key] = val
    st.session_state['config_loaded'] = True

# --- 3. UTILIDADES ---
def fmt(x):
    return f"{x:,.0f}".replace(",", ".")

def card(title, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="font-size:11px; color:#888;">{sub}</div>
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

st.title("🌲 Planilla de Costos Completa")

# --- A. JORNADA ---
with st.expander("📅 Configuración de Jornada", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", value=28, step=1, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
        st.caption(f"Divisor: {fmt(h_total_hours)} Hrs Mes")
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", value=25, step=1, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day
        st.caption(f"Divisor: {fmt(f_total_hours)} Hrs Mes")

st.divider()

# --- B. FUNCIONES MAQUINARIA ---

def render_machine_table(prefix, col_obj, machine_days, machine_hours_total, fuel_p):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        
        key_df = f"df_{prefix.lower()}_v17"
        
        if key_df not in st.session_state:
            if prefix == "Harvester":
                data = [
                    {"Categoría": "Fijos", "Ítem": "Arriendo Maquinaria", "Valor Input": 10900000, "Unidad": "$/Mes"},
                    {"Categoría": "Fijos", "Ítem": "Sueldos Operadores", "Valor Input": 3800000, "Unidad": "$/Mes"},
                    {"Categoría": "Variable", "Ítem": "Petróleo (Consumo)", "Valor Input": 200.0, "Unidad": "Litros/Día"},
                    {"Categoría": "Mantención", "Ítem": "Mantenciones Varias", "Valor Input": 1300000, "Unidad": "$/Mes"},
                    {"Categoría": "Consumibles", "Ítem": "Cadenas/Espadas", "Valor Input": 300000, "Unidad": "$/Mes"},
                    {"Categoría": "Otros", "Ítem": "Varios", "Valor Input": 0, "Unidad": "$/Mes"},
                ]
            else:
                data = [
                    {"Categoría": "Fijos", "Ítem": "Arriendo Forwarder", "Valor Input": 8000000, "Unidad": "$/Mes"},
                    {"Categoría": "Fijos", "Ítem": "Sueldo Operador", "Valor Input": 1900000, "Unidad": "$/Mes"},
                    {"Categoría": "Variable", "Ítem": "Petróleo (Diesel)", "Valor Input": 135.0, "Unidad": "Litros/Día"},
                    {"Categoría": "Mantención", "Ítem": "Mantención 600/1200 hrs", "Valor Input": 220000, "Unidad": "$/Mes"},
                    {"Categoría": "Mantención", "Ítem": "Mantención Garra/Grúa", "Valor Input": 150000, "Unidad": "$/Mes"},
                    {"Categoría": "Mantención", "Ítem": "Neumáticos", "Valor Input": 360000, "Unidad": "$/Mes"},
                    {"Categoría": "Consumibles", "Ítem": "Grasa y Lubricantes", "Valor Input": 80000, "Unidad": "$/Mes"},
                    {"Categoría": "Otros", "Ítem": "Otros Insumos", "Valor Input": 50000, "Unidad": "$/Mes"},
                ]
            st.session_state[key_df] = pd.DataFrame(data)
            
        st.info("Ingresa **Litros/Día** para Petróleo.")
        
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
            height=350
        )
        st.session_state[key_df] = edited_df
        save_config()
        
        total_month_sum = 0
        fuel_daily_liters = 0
        fuel_monthly_cost = 0
        
        for index, row in edited_df.iterrows():
            val = row["Valor Input"]
            unit = row["Unidad"]
            row_cost = 0
            
            if unit == "Litros/Día":
                fuel_daily_liters = val
                fuel_monthly_cost = val * machine_days * fuel_p
                row_cost = fuel_monthly_cost
            else:
                row_cost = val
            
            total_month_sum += row_cost
            
        total_hr = total_month_sum / machine_hours_total if machine_hours_total else 0
        
        st.success(f"**Costo Hora: ${fmt(total_hr)}**")
        return total_month_sum, total_hr

col_tab1, col_tab2 = st.columns(2)
h_total_m, h_total_hr = render_machine_table("Harvester", col_tab1, h_days, h_total_hours, fuel_price)
f_total_m, f_total_hr = render_machine_table("Forwarder", col_tab2, f_days, f_total_hours, fuel_price)

# --- D. INDIRECTOS (LISTADO NUEVO) ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos")

key_ind = "df_indirect_list_v19"

if key_ind not in st.session_state:
    # LISTADO EXACTO DE LA IMAGEN
    data_ind = [
        {"Ítem": "Instalación de faena", "Tipo": "$/Mes", "Valor Input": 0},
        {"Ítem": "Arriendo camioneta 1", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Arriendo camioneta 2", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Arriendo camioneta 3", "Tipo": "UF/Mes", "Valor Input": 0.0},
        {"Ítem": "Combustible camioneta 1", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Combustible camioneta 2", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Combustible camioneta 3", "Tipo": "Litros/Día", "Valor Input": 0.0},
        {"Ítem": "Prevencionista de riesgos", "Tipo": "$/Mes", "Valor Input": 800000},
        {"Ítem": "Otros / agregar mensual", "Tipo": "$/Mes", "Valor Input": 100000},
        {"Ítem": "Pensión personal", "Tipo": "$/Mes", "Valor Input": 1890000},
        {"Ítem": "Mecánico", "Tipo": "$/Mes", "Valor Input": 0},
        {"Ítem": "EPP Y Ropa de trabajo", "Tipo": "$/Mes", "Valor Input": 200000},
        {"Ítem": "Gastos Adm y Gerencia", "Tipo": "$/Mes", "Valor Input": 500000},
    ]
    st.session_state[key_ind] = pd.DataFrame(data_ind)

with st.expander("📝 Editar Costos Indirectos (Listado Detallado)", expanded=True):
    col_i1, col_i2 = st.columns([2, 1])
    
    with col_i2:
        pickup_days = st.number_input("Días Mes Camionetas", value=30, key="pickup_days_use", on_change=save_config)
        st.info("Este valor multiplica el consumo de litros diarios.")
        st.caption(f"UF Actual: ${fmt(current_uf)}")
        st.caption(f"Diesel: ${fmt(fuel_price)}")

    with col_i1:
        edited_ind = st.data_editor(
            st.session_state[key_ind],
            key="editor_indirect_list",
            column_config={
                "Ítem": st.column_config.TextColumn(disabled=True),
                "Tipo": st.column_config.TextColumn(disabled=True),
                "Valor Input": st.column_config.NumberColumn("Valor", format="%f")
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )
        st.session_state[key_ind] = edited_ind
        save_config()

# CÁLCULO DE INDIRECTOS
total_shared = 0
for idx, row in edited_ind.iterrows():
    val = row["Valor Input"]
    tipo = row["Tipo"]
    
    row_cost = 0
    if tipo == "UF/Mes":
        row_cost = val * current_uf
    elif tipo == "Litros/Día":
        row_cost = val * pickup_days * fuel_price
    else: # $/Mes
        row_cost = val
        
    total_shared += row_cost

st.success(f"**TOTAL INDIRECTOS: ${fmt(total_shared)} /Mes**")

# Asignación
alloc_opts = ["Manual", "Proporcional Horas"]
idx = alloc_opts.index(st.session_state.get("alloc_method", "Manual")) if st.session_state.get("alloc_method") in alloc_opts else 0
alloc = st.radio("Distribución Indirectos", alloc_opts, index=idx, horizontal=True, on_change=save_config, key="alloc_method")

if alloc == "Manual":
    h_pct = st.slider("% Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
else:
    tt = h_total_hours + f_total_hours
    h_pct = h_total_hours / tt if tt > 0 else 0.5

shared_h = total_shared * h_pct
shared_f = total_shared * (1 - h_pct)

# --- E. RESULTADOS FINALES ---
st.divider()
st.subheader("📊 Resultados Consolidados")

final_h_hr = (h_total_m + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_m + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

c1, c2, c3 = st.columns(3)
with c1: card("Costo Harvester", f"${fmt(final_h_hr)}/hr", "Inc. Indirectos")
with c2: card("Costo Forwarder", f"${fmt(final_f_hr)}/hr", "Inc. Indirectos")
with c3: card("Costo Sistema", f"${fmt(sys_hr)}/hr", "Total Operacional")

# Simulador
st.markdown("### 🧮 Simulador de Rentabilidad")
col_sim_in, col_sim_out = st.columns([1, 2])

with col_sim_in:
    st.markdown("**Producción Estimada ($m^3$/hr)**")
    m3_h = st.number_input("Harvester", value=25.0, step=0.5, key="sim_m3_h_val", on_change=save_config)
    m3_f = st.number_input("Forwarder", value=28.0, step=0.5, key="sim_m3_f_val", on_change=save_config)
    
    mr_h = m3_h / conversion_factor if conversion_factor else 0
    mr_f = m3_f / conversion_factor if conversion_factor else 0
    mr_sys = min(mr_h, mr_f)

with col_sim_out:
    cost_sys_unit = sys_hr / mr_sys if mr_sys else 0
    util_unit = sales_price_mr - cost_sys_unit
    margen_pct = (util_unit / sales_price_mr * 100) if sales_price_mr else 0
    
    color_m = "green" if util_unit > 0 else "red"
    st.markdown(f"""
    <div style="background-color:#f8f9fa; border:1px solid #ddd; border-radius:10px; padding:20px; text-align:center;">
        <h3 style="margin:0; color:#333;">Margen del Sistema Real</h3>
        <div style="font-size:36px; font-weight:bold; color:{color_m};">{margen_pct:.1f}%</div>
        <div style="color:#555;">Utilidad: <b>${fmt(util_unit)}</b> / MR</div>
        <hr>
        <div style="font-size:13px;">Prod. Real (Limitada): <b>{mr_sys:.1f} MR/hr</b></div>
    </div>
    """, unsafe_allow_html=True)
