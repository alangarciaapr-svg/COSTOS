import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="ForestCost Simple",
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
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .big-input .stNumberInput input {
        font-size: 20px;
        font-weight: bold;
        color: #15803d;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_simple_v1.json'

# --- 2. PERSISTENCIA ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
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
    "use_auto_uf", "uf_manual", "sales_price_mr", "conversion_factor", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    "total_cost_harvester", "total_cost_forwarder",
    "sim_m3_h", "sim_m3_f", "df_indirect_simple", "alloc_method", "h_share_pct_manual"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            if key == "df_indirect_simple":
                st.session_state[key] = pd.DataFrame(saved_config[key])
            else:
                st.session_state[key] = saved_config[key]
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
    st.title("Configuración")
    
    # Economía Base
    uf_api, fecha_api = get_uf_value()
    use_auto_uf = st.checkbox("UF Automática", value=True, key="use_auto_uf", on_change=save_config)
    if use_auto_uf and uf_api:
        current_uf = uf_api
        st.success(f"UF: ${fmt(current_uf)}")
    else:
        current_uf = st.number_input("UF Manual", value=38000.0, step=100.0, key="uf_manual", on_change=save_config)

    st.markdown("### Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)
    
    st.markdown("### Ingresos")
    h_rev_pct = st.slider("% Ingreso Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    f_rev_pct = 100 - h_rev_pct
    
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * (f_rev_pct / 100)
    st.info(f"H: ${fmt(h_income)} | F: ${fmt(f_income)}")

# --- 5. LOGICA PRINCIPAL ---

st.title("🌲 ForestCost (Entrada Directa)")

# --- A. JORNADA ---
with st.expander("📅 Configuración de Jornada", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", value=28, step=1, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
        st.caption(f"Total: {fmt(h_total_hours)} Hrs")
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", value=25, step=1, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day
        st.caption(f"Total: {fmt(f_total_hours)} Hrs")

st.divider()

# --- B. COSTOS TOTALES (SIMPLIFICADO) ---
st.markdown("### 💰 Costos Directos Totales")
st.markdown('<style>.big-input {font-size: 1.2rem;}</style>', unsafe_allow_html=True)

col_h, col_f = st.columns(2)

with col_h:
    st.subheader("🚜 Harvester")
    # INPUT UNICO: TOTAL COSTO MENSUAL
    h_total_month = st.number_input(
        "Costo Total Mensual ($)", 
        value=15000000, 
        step=500000, 
        key="total_cost_harvester", 
        on_change=save_config,
        help="Incluye arriendo, operador, petróleo, mantenciones, etc."
    )
    # Cálculo hora
    h_cost_hr_direct = h_total_month / h_total_hours if h_total_hours else 0
    st.info(f"Costo Hora Directo: **${fmt(h_cost_hr_direct)}**")

with col_f:
    st.subheader("🚜 Forwarder")
    # INPUT UNICO: TOTAL COSTO MENSUAL
    f_total_month = st.number_input(
        "Costo Total Mensual ($)", 
        value=10000000, 
        step=500000, 
        key="total_cost_forwarder", 
        on_change=save_config,
        help="Incluye arriendo, operador, petróleo, mantenciones, etc."
    )
    # Cálculo hora
    f_cost_hr_direct = f_total_month / f_total_hours if f_total_hours else 0
    st.info(f"Costo Hora Directo: **${fmt(f_cost_hr_direct)}**")

# --- C. INDIRECTOS ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos (Faena)")

key_ind = "df_indirect_simple"
if key_ind not in st.session_state:
    st.session_state[key_ind] = pd.DataFrame([
        {"Ítem": "Camionetas (Arr + Diesel)", "Total ($)": 1500000},
        {"Ítem": "Personal / Administración", "Total ($)": 2000000},
        {"Ítem": "Instalaciones / Varios", "Total ($)": 1000000},
    ])

with st.expander("📝 Editar Indirectos (Totales)", expanded=True):
    edited_ind = st.data_editor(
        st.session_state[key_ind],
        key="editor_indirect",
        column_config={
            "Total ($)": st.column_config.NumberColumn(format="$%d")
        },
        hide_index=True,
        use_container_width=True
    )
    st.session_state[key_ind] = edited_ind
    save_config()

total_shared = edited_ind["Total ($)"].sum()
st.success(f"**TOTAL INDIRECTOS: ${fmt(total_shared)} /Mes**")

# Asignación
alloc_opts = ["Manual", "Proporcional Horas"]
idx = alloc_opts.index(st.session_state.get("alloc_method", "Manual")) if st.session_state.get("alloc_method") in alloc_opts else 0
alloc = st.radio("Distribución Indirectos", alloc_opts, index=idx, horizontal=True, on_change=save_config, key="alloc_method")

if alloc == "Manual":
    h_pct = st.slider("% Asignado Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
else:
    tt = h_total_hours + f_total_hours
    h_pct = h_total_hours / tt if tt > 0 else 0.5

shared_h = total_shared * h_pct
shared_f = total_shared * (1 - h_pct)

# --- D. RESULTADOS ---
st.divider()
st.subheader("📊 Resultados Consolidados")

final_h_hr = (h_total_month + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_month + shared_f) / f_total_hours if f_total_hours else 0
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
    m3_h = st.number_input("Harvester", value=25.0, step=0.5, key="sim_m3_h", on_change=save_config)
    m3_f = st.number_input("Forwarder", value=28.0, step=0.5, key="sim_m3_f", on_change=save_config)
    
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
        <div style="font-size:13px;">Prod. Real: <b>{mr_sys:.1f} MR/hr</b></div>
    </div>
    """, unsafe_allow_html=True)

# Tabla detalle
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
