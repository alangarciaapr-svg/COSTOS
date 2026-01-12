import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import requests

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="ForestCost Pro Advanced",
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
    .highlight-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v24_complete.json'

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
    "df_harvester_v24", "df_forwarder_v24", "df_indirect_v24",
    "sim_m3_h_val", "sim_m3_f_val",
    "pickup_days_use", "target_margin_pct"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            if key in ["df_harvester_v24", "df_forwarder_v24", "df_indirect_v24"]:
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
    
    st.markdown("### 3. Distribución")
    st.caption("Esta distribución define tanto los Ingresos como la asignación de Costos Indirectos.")
    h_rev_pct = st.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    f_rev_pct = 100 - h_rev_pct
    
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * (f_rev_pct / 100)
    
    st.info(f"Ingreso H: ${fmt(h_income)} | Ingreso F: ${fmt(f_income)}")

# --- 5. LOGICA PRINCIPAL ---

st.title("🌲 ForestCost Pro: Gestión Avanzada")

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

# --- B. TABLA MAESTRA AVANZADA (PARA AMBAS MÁQUINAS) ---

def render_advanced_table(prefix, col_obj, machine_days, machine_hours_total, fuel_p, uf_val):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        
        # KEY única por máquina y versión
        key_df = f"df_{prefix.lower()}_v24"
        
        if key_df not in st.session_state:
            # ESTRUCTURA DETALLADA PARA AMBAS MÁQUINAS
            if prefix == "Harvester":
                data = [
                    {"Categoría": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 10900000},
                    {"Categoría": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 1923721},
                    {"Categoría": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 1923721},
                    {"Categoría": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frecuencia (Hrs)": 1, "Valor Input": 200.0},
                    {"Categoría": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frecuencia (Hrs)": 1, "Valor Input": 200.0},
                    {"Categoría": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 600, "Valor Input": 181840},
                    {"Categoría": "Mantención", "Ítem": "Mant. 1200h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 1200, "Valor Input": 181840},
                    {"Categoría": "Mantención", "Ítem": "Mant. 1800h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 1800, "Valor Input": 1990848},
                    {"Categoría": "Mantención", "Ítem": "Hidráulica 6000h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 6000, "Valor Input": 19500000},
                    {"Categoría": "Mantención", "Ítem": "Cabezal/Grúa 10kh", "Tipo": "$/Evento", "Frecuencia (Hrs)": 10000, "Valor Input": 15000000},
                    {"Categoría": "Mantención", "Ítem": "Electrónica Mensual", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 100000},
                    {"Categoría": "Consumibles", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 300000},
                    {"Categoría": "Consumibles", "Ítem": "Grasa/Aceites", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 250000},
                    {"Categoría": "Otros", "Ítem": "Seguro (UF)", "Tipo": "UF/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 19},
                    {"Categoría": "Reserva", "Ítem": "Overhaul/Motor", "Tipo": "$/Evento", "Frecuencia (Hrs)": 20000, "Valor Input": 40000000},
                ]
            else:
                # FORWARDER CON LA MISMA ESTRUCTURA DETALLADA
                data = [
                    {"Categoría": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 8000000},
                    {"Categoría": "Fijos", "Ítem": "Operador", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 1900000},
                    {"Categoría": "Variable", "Ítem": "Petróleo", "Tipo": "Litros/Día", "Frecuencia (Hrs)": 1, "Valor Input": 135.0},
                    {"Categoría": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 600, "Valor Input": 150000},
                    {"Categoría": "Mantención", "Ítem": "Mant. 1200h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 1200, "Valor Input": 200000},
                    {"Categoría": "Mantención", "Ítem": "Mant. Grúa/Garra 6000h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 6000, "Valor Input": 7000000},
                    {"Categoría": "Mantención", "Ítem": "Electrónica/Varios", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 50000},
                    {"Categoría": "Consumibles", "Ítem": "Neumáticos (Juego)", "Tipo": "$/Evento", "Frecuencia (Hrs)": 10000, "Valor Input": 20000000},
                    {"Categoría": "Consumibles", "Ítem": "Grasa y Lubricantes", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 80000},
                    {"Categoría": "Otros", "Ítem": "Seguro (UF)", "Tipo": "UF/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 18},
                    {"Categoría": "Reserva", "Ítem": "Overhaul Tren/Motor", "Tipo": "$/Evento", "Frecuencia (Hrs)": 18000, "Valor Input": 44000000},
                ]
            st.session_state[key_df] = pd.DataFrame(data)
            
        st.info("Ingresa **Litros/Día** (Diesel) y **Costo Evento** (Mantenciones).")
        
        edited_df = st.data_editor(
            st.session_state[key_df],
            key=f"editor_{prefix}_adv",
            column_config={
                "Categoría": st.column_config.TextColumn(disabled=True),
                "Ítem": st.column_config.TextColumn(disabled=True),
                "Tipo": st.column_config.TextColumn(disabled=True),
                "Frecuencia (Hrs)": st.column_config.NumberColumn(format="%d"),
                "Valor Input": st.column_config.NumberColumn("Valor Input", format="%d", required=True),
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
        st.session_state[key_df] = edited_df
        save_config()
        
        # --- CÁLCULOS ---
        total_month = 0
        
        for index, row in edited_df.iterrows():
            val = row["Valor Input"]
            tipo = row["Tipo"]
            freq = row.get("Frecuencia (Hrs)", 1)
            
            item_month = 0
            
            if tipo == "$/Mes":
                item_month = val
            elif tipo == "UF/Mes":
                item_month = val * uf_val
            elif tipo == "Litros/Día":
                item_month = val * machine_days * fuel_p
            elif tipo == "$/Evento":
                if freq > 0 and machine_hours_total > 0:
                    cost_per_hour = val / freq
                    item_month = cost_per_hour * machine_hours_total
            
            total_month += item_month
            
        total_hr = total_month / machine_hours_total if machine_hours_total else 0
        
        st.success(f"**Costo Hora: ${fmt(total_hr)}**")
        return total_month, total_hr

col_tab1, col_tab2 = st.columns(2)
h_total_m, h_total_hr = render_advanced_table("Harvester", col_tab1, h_days, h_total_hours, fuel_price, current_uf)
f_total_m, f_total_hr = render_advanced_table("Forwarder", col_tab2, f_days, f_total_hours, fuel_price, current_uf)

# --- D. INDIRECTOS ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos")

key_ind = "df_indirect_v24"

if key_ind not in st.session_state:
    data_ind = [
        {"Ítem": "Instalación de faena", "Tipo": "$/Mes", "Valor Input": 0},
        {"Ítem": "Arriendo camioneta 1", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Arriendo camioneta 2", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Combustible camioneta 1", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Combustible camioneta 2", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Prevencionista de riesgos", "Tipo": "$/Mes", "Valor Input": 800000},
        {"Ítem": "Otros / agregar mensual", "Tipo": "$/Mes", "Valor Input": 100000},
        {"Ítem": "Pensión personal", "Tipo": "$/Mes", "Valor Input": 1890000},
        {"Ítem": "EPP Y Ropa de trabajo", "Tipo": "$/Mes", "Valor Input": 200000},
        {"Ítem": "Gastos Adm y Gerencia", "Tipo": "$/Mes", "Valor Input": 500000},
    ]
    st.session_state[key_ind] = pd.DataFrame(data_ind)

with st.expander("📝 Editar Costos Indirectos", expanded=True):
    col_i1, col_i2 = st.columns([2, 1])
    with col_i2:
        pickup_days = st.number_input("Días Mes Camionetas", value=30, key="pickup_days_use", on_change=save_config)
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
            height=350
        )
        st.session_state[key_ind] = edited_ind
        save_config()

total_shared = 0
for idx, row in edited_ind.iterrows():
    val = row["Valor Input"]
    tipo = row["Tipo"]
    row_cost = 0
    if tipo == "UF/Mes": row_cost = val * current_uf
    elif tipo == "Litros/Día": row_cost = val * pickup_days * fuel_price
    else: row_cost = val
    total_shared += row_cost

st.success(f"**TOTAL INDIRECTOS: ${fmt(total_shared)} /Mes**")

# ASIGNACIÓN AUTOMÁTICA BASADA EN GANANCIAS
# Según requerimiento: "mismo porcentaje en los costos fijos... que los porcentajes de ganancia"
h_share_pct = h_rev_pct / 100.0
f_share_pct = f_rev_pct / 100.0

shared_h = total_shared * h_share_pct
shared_f = total_shared * f_share_pct

st.caption(f"Distribución Indirectos (Automática por % Ingreso): Harvester {h_rev_pct}% | Forwarder {f_rev_pct}%")

# --- E. RESULTADOS CONSOLIDADOS ---
final_h_hr = (h_total_m + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_m + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

st.divider()

# --- F. SIMULADOR Y TARIFAS ---
st.subheader("🚀 Simuladores de Rentabilidad")

tab_sim1, tab_sim2 = st.tabs(["📉 Rentabilidad Actual", "🎯 Calculadora de Tarifa Objetiva"])

# 1. SIMULADOR ACTUAL
with tab_sim1:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Producción ($m^3$/hr)**")
        m3_h = st.number_input("Harvester", value=25.0, step=0.5, key="sim_m3_h_val", on_change=save_config)
        m3_f = st.number_input("Forwarder", value=28.0, step=0.5, key="sim_m3_f_val", on_change=save_config)
        
        # Conversión a MR
        mr_h = m3_h / conversion_factor if conversion_factor else 0
        mr_f = m3_f / conversion_factor if conversion_factor else 0
        mr_sys = min(mr_h, mr_f)
        
        st.markdown(f"**Producción MR:** H: {mr_h:.1f} | F: {mr_f:.1f}")
        st.warning(f"**Cuello de Botella (Sistema): {mr_sys:.1f} MR/hr**")
    
    with col_s2:
        # Costos Unitarios
        cost_unit_h = final_h_hr / mr_h if mr_h else 0
        cost_unit_f = final_f_hr / mr_f if mr_f else 0
        cost_unit_sys = sys_hr / mr_sys if mr_sys else 0
        
        # Utilidad
        util_h = h_income - cost_unit_h
        util_f = f_income - cost_unit_f
        util_sys = sales_price_mr - cost_unit_sys
        
        # Márgenes
        margin_h = (util_h / h_income * 100) if h_income else 0
        margin_f = (util_f / f_income * 100) if f_income else 0
        margin_sys = (util_sys / sales_price_mr * 100) if sales_price_mr else 0
        
        st.markdown(f"""
        <div class="highlight-box">
            <h4>Rentabilidad del Sistema</h4>
            <div style="font-size: 2em; font-weight: bold; color: {'green' if margin_sys > 0 else 'red'}">{margin_sys:.1f}%</div>
            <div>Utilidad: ${fmt(util_sys)} / MR</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Harvester:** Costo ${fmt(cost_unit_h)} | Margen {margin_h:.1f}%")
        st.write(f"**Forwarder:** Costo ${fmt(cost_unit_f)} | Margen {margin_f:.1f}%")

# 2. SIMULADOR INVERSO (PRECIO OBJETIVO)
with tab_sim2:
    st.markdown("##### ¿Cuánto debo cobrar para ganar X%?")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        target_margin = st.slider("Rentabilidad Deseada (%)", 0, 60, 20, key="target_margin_pct", on_change=save_config) / 100.0
        
    with col_inv2:
        # Precio = Costo / (1 - Margen)
        if mr_sys > 0:
            target_price_mr = cost_unit_sys / (1 - target_margin)
            target_util = target_price_mr - cost_unit_sys
            
            st.markdown(f"""
            <div style="background-color: #e0f2fe; padding: 15px; border-radius: 8px; border: 1px solid #7dd3fc; text-align:center;">
                <div style="color: #0c4a6e; font-weight:bold;">PRECIO SUGERIDO</div>
                <div style="font-size: 28px; font-weight:bold; color: #0284c7;">${fmt(target_price_mr)} / MR</div>
                <div style="font-size: 12px; color: #0369a1;">Para ganar {target_margin*100:.0f}% (${fmt(target_util)}/MR)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Ingresa producción en el simulador para calcular.")
