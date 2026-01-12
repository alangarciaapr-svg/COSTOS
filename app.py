import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="ForestCost Analytics",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #2c3e50;
    }
    .big-input .stNumberInput input {
        font-weight: bold;
        color: #15803d;
    }
    .stDataEditor {
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v21_fixed.json'

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
    "sim_m3_h", "sim_m3_f", "df_indirect_simple_v2", "alloc_method", "h_share_pct_manual"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            if key == "df_indirect_simple_v2":
                st.session_state[key] = pd.DataFrame(saved_config[key])
            else:
                st.session_state[key] = saved_config[key]
    st.session_state['config_loaded'] = True

# --- 3. UTILIDADES ---
def fmt(x):
    return f"{x:,.0f}".replace(",", ".")

def kpi_card(label, value, sub="", color="#2c3e50"):
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div style="font-size:12px; color:#888; margin-top:4px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Configuración")
    st.markdown("### 💰 Economía")
    uf_val = st.number_input("Valor UF ($)", value=38000.0, step=100.0, key="uf_manual", on_change=save_config)
    
    st.markdown("### 🌲 Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Precio Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)
    
    st.divider()
    st.markdown("### 📊 Ingresos")
    h_rev_pct = st.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * ((100 - h_rev_pct) / 100)
    
    st.success(f"Harvester: **${fmt(h_income)} /MR**")
    st.info(f"Forwarder: **${fmt(f_income)} /MR**")

# --- 5. CUERPO PRINCIPAL ---

st.title("🌲 Dashboard de Rentabilidad Forestal")

# --- SECCIÓN 1: DATOS CLAVE ---
with st.expander("📝 1. Configuración de Jornada y Costos Mensuales", expanded=True):
    col_h_in, col_f_in = st.columns(2)
    
    # HARVESTER
    with col_h_in:
        st.subheader("🚜 Harvester")
        c1, c2 = st.columns(2)
        h_days = c1.number_input("Días/Mes (H)", value=28, key="h_days_month", on_change=save_config)
        h_hours = c2.number_input("Hrs/Día (H)", value=10.0, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours
        
        st.markdown("**Costo Total Mensual ($)**")
        h_total_month = st.number_input("Monto Mes H", value=15000000, step=500000, key="total_cost_harvester", label_visibility="collapsed", on_change=save_config)
        
        h_cost_hr_direct = h_total_month / h_total_hours if h_total_hours else 0
        st.caption(f"Divisor: {h_days}d x {h_hours}h = **{h_total_hours} hrs/mes**")
        st.info(f"Valor Hora Directo: **${fmt(h_cost_hr_direct)}**")

    # FORWARDER
    with col_f_in:
        st.subheader("🚜 Forwarder")
        c3, c4 = st.columns(2)
        f_days = c3.number_input("Días/Mes (F)", value=25, key="f_days_month", on_change=save_config)
        f_hours = c4.number_input("Hrs/Día (F)", value=9.0, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours
        
        st.markdown("**Costo Total Mensual ($)**")
        f_total_month = st.number_input("Monto Mes F", value=10000000, step=500000, key="total_cost_forwarder", label_visibility="collapsed", on_change=save_config)
        
        f_cost_hr_direct = f_total_month / f_total_hours if f_total_hours else 0
        st.caption(f"Divisor: {f_days}d x {f_hours}h = **{f_total_hours} hrs/mes**")
        st.info(f"Valor Hora Directo: **${fmt(f_cost_hr_direct)}**")

# --- INDIRECTOS (TABLA CON CORRECCIÓN DE ERROR) ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos")

key_ind = "df_indirect_simple_v2" # Cambiamos nombre key para forzar reset limpio

# Estructura por defecto
default_ind = pd.DataFrame([
    {"Ítem": "Camionetas (Arr + Diesel)", "Monto ($)": 1500000},
    {"Ítem": "Personal / Administración", "Monto ($)": 2000000},
    {"Ítem": "Instalaciones / Varios", "Monto ($)": 1000000},
])

if key_ind not in st.session_state:
    st.session_state[key_ind] = default_ind

# VALIDACIÓN DE SEGURIDAD (FIX KEYERROR)
# Si por alguna razón carga datos viejos sin la columna correcta, reseteamos
if "Monto ($)" not in st.session_state[key_ind].columns:
    st.session_state[key_ind] = default_ind

# Editor
edited_ind = st.data_editor(
    st.session_state[key_ind],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Monto ($)": st.column_config.NumberColumn(format="$%d")
    },
    key="editor_indirect_v2"
)

# Guardar cambios
if not edited_ind.equals(st.session_state[key_ind]):
    st.session_state[key_ind] = edited_ind
    save_config()

# Cálculo Total Indirecto
total_shared = edited_ind["Monto ($)"].sum()
st.success(f"**Total Indirectos: ${fmt(total_shared)} /Mes**")

# Asignación
c_alloc1, c_alloc2 = st.columns(2)
with c_alloc1:
    alloc_method = st.radio("Método Distribución", ["Proporcional a Horas", "Manual"], index=0, horizontal=True, key="alloc_method", on_change=save_config)

h_share = 0
f_share = 0

if alloc_method == "Manual":
    with c_alloc2:
        h_pct = st.slider("% Asignado a Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config)
    h_share = total_shared * (h_pct/100)
else:
    tt = h_total_hours + f_total_hours
    if tt > 0:
        h_share = total_shared * (h_total_hours/tt)
    else:
        h_share = 0

f_share = total_shared - h_share

# --- COSTOS FINALES ---
final_h_hr = (h_total_month + h_share) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_month + f_share) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

st.divider()

# --- SECCIÓN 2: SIMULADOR Y KPIs VISUALES ---
st.markdown("### 🚀 2. Simulador de Producción y Rentabilidad")

col_sim, col_kpi = st.columns([1, 2])

with col_sim:
    st.markdown("##### ⚙️ Producción Real ($m^3/hr$)")
    m3_h = st.number_input("Harvester", value=25.0, step=0.5, key="sim_m3_h", on_change=save_config)
    m3_f = st.number_input("Forwarder", value=28.0, step=0.5, key="sim_m3_f", on_change=save_config)
    
    mr_h = m3_h / conversion_factor if conversion_factor else 0
    mr_f = m3_f / conversion_factor if conversion_factor else 0
    mr_sys = min(mr_h, mr_f)
    
    st.warning(f"**Sistema Limitado a:** {mr_sys:.1f} MR/hr")

# Cálculos Rentabilidad
cost_sys_unit = sys_hr / mr_sys if mr_sys else 0
util_unit = sales_price_mr - cost_sys_unit
margen_pct = (util_unit / sales_price_mr * 100) if sales_price_mr else 0

with col_kpi:
    # Tarjetas Superiores
    k1, k2, k3 = st.columns(3)
    with k1: kpi_card("Costo Unitario", f"${fmt(cost_sys_unit)}", "por MR producido")
    with k2: kpi_card("Utilidad Neta", f"${fmt(util_unit)}", "por MR producido", color="#16a34a" if util_unit>0 else "#dc2626")
    with k3: kpi_card("Margen", f"{margen_pct:.1f}%", "Rentabilidad", color="#16a34a" if margen_pct>0 else "#dc2626")

# --- GRÁFICOS DINÁMICOS ---
st.markdown("---")
g1, g2 = st.columns(2)

with g1:
    st.subheader("🌡️ Termómetro de Rentabilidad")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = margen_pct,
        number = {'suffix': "%"},
        title = {'text': "Margen Operacional"},
        gauge = {
            'axis': {'range': [-20, 50]},
            'bar': {'color': "#1f2937"},
            'steps': [
                {'range': [-20, 0], 'color': "#fee2e2"},
                {'range': [0, 15], 'color': "#fef3c7"},
                {'range': [15, 50], 'color': "#dcfce7"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:
    st.subheader("⚖️ Comparativa: Costo vs Ingreso")
    
    # Datos por maquina
    c_h_unit = final_h_hr / mr_h if mr_h else 0
    c_f_unit = final_f_hr / mr_f if mr_f else 0
    
    df_chart = pd.DataFrame({
        "Equipo": ["Harvester", "Forwarder", "SISTEMA"],
        "Costo Unitario": [c_h_unit, c_f_unit, cost_sys_unit],
        "Ingreso Unitario": [h_income, f_income, sales_price_mr]
    })
    
    fig_bar = go.Figure(data=[
        go.Bar(name='Costo ($/MR)', x=df_chart['Equipo'], y=df_chart['Costo Unitario'], marker_color='#ef4444'),
        go.Bar(name='Ingreso ($/MR)', x=df_chart['Equipo'], y=df_chart['Ingreso Unitario'], marker_color='#22c55e')
    ])
    fig_bar.update_layout(barmode='group', height=300, margin=dict(t=30, b=10, l=30, r=30))
    st.plotly_chart(fig_bar, use_container_width=True)
