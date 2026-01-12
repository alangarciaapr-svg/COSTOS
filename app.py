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

CONFIG_FILE = 'forest_config_v16_final.json'

# --- 2. GESTIÓN DE PERSISTENCIA ---
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
    "df_harvester_v16", "df_forwarder_v16", "df_indirect_v16",
    "sim_m3_h", "sim_m3_f", "alloc_method", "h_share_pct_manual"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            if key in ["df_harvester_v16", "df_forwarder_v16", "df_indirect_v16"]:
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

st.title("🌲 Planilla de Costos (Inputs Totales)")

# --- A. JORNADA ---
with st.expander("📅 Configuración de Jornada (Variables Clave)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", value=28, step=1, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
        st.caption(f"Total: {fmt(h_total_hours)} Hrs Mes")
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", value=25, step=1, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day
        st.caption(f"Total: {fmt(f_total_hours)} Hrs Mes")

st.divider()

# --- B. TABLAS DE COSTOS (LÓGICA MEJORADA) ---

col_tab1, col_tab2 = st.columns(2)

def create_machine_table(prefix, col_obj, machine_days, machine_hours_total, fuel_p):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        
        # Estructura Inicial (Si no existe)
        initial_data = [
            {"Categoría": "Fijos", "Ítem": "Arriendo Maquinaria", "Valor Input": 10900000 if prefix=="Harvester" else 8000000, "Unidad": "$/Mes"},
            {"Categoría": "Fijos", "Ítem": "Sueldos Operadores", "Valor Input": 3800000 if prefix=="Harvester" else 1900000, "Unidad": "$/Mes"},
            {"Categoría": "Fijos", "Ítem": "Seguros / Otros", "Valor Input": 750000, "Unidad": "$/Mes"},
            
            # FILA CLAVE: PETROLEO EN LITROS DIARIOS
            {"Categoría": "Variable", "Ítem": "Consumo Petróleo", "Valor Input": 200.0 if prefix=="Harvester" else 135.0, "Unidad": "Litros/Día"},
            
            {"Categoría": "Mantención", "Ítem": "Mant. Preventiva", "Valor Input": 800000 if prefix=="Harvester" else 500000, "Unidad": "$/Mes"},
            {"Categoría": "Mantención", "Ítem": "Mant. Correctiva", "Valor Input": 500000 if prefix=="Harvester" else 300000, "Unidad": "$/Mes"},
            {"Categoría": "Mantención", "Ítem": "Neumáticos/Rodado", "Valor Input": 280000, "Unidad": "$/Mes"},
            
            {"Categoría": "Consumibles", "Ítem": "Elementos Corte", "Valor Input": 200000, "Unidad": "$/Mes"},
            {"Categoría": "Consumibles", "Ítem": "Aceite Hidráulico", "Valor Input": 160000, "Unidad": "$/Mes"},
            {"Categoría": "Consumibles", "Ítem": "Lubricantes/Filtros", "Valor Input": 60000, "Unidad": "$/Mes"},
            
            {"Categoría": "Otros", "Ítem": "Reservas/Varios", "Valor Input": 0, "Unidad": "$/Mes"},
        ]
        
        key_df = f"df_{prefix.lower()}_v16"
        if key_df not in st.session_state:
            st.session_state[key_df] = pd.DataFrame(initial_data)
            
        st.info("Edita 'Valor Input'. Si la unidad es **Litros/Día**, se multiplicará por Días y Petróleo.")
        
        # EDITOR
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
            height=450
        )
        # Guardar inmediatamente en state
        st.session_state[key_df] = edited_df
        save_config()
        
        # --- CÁLCULOS ROBUSTOS ---
        total_month_sum = 0
        fuel_daily_liters = 0
        fuel_monthly_cost = 0
        
        for index, row in edited_df.iterrows():
            val = row["Valor Input"]
            unit = row["Unidad"]
            
            row_cost = 0
            
            if unit == "Litros/Día":
                # Lógica Específica: LitrosDia * DiasMes * Precio
                fuel_daily_liters = val
                fuel_monthly_cost = val * machine_days * fuel_p
                row_cost = fuel_monthly_cost
            else:
                # Asumimos $/Mes
                row_cost = val
            
            total_month_sum += row_cost
            
        total_hr = total_month_sum / machine_hours_total if machine_hours_total else 0
        
        # Tarjeta Resumen
        st.success(f"**Costo Hora: ${fmt(total_hr)}**")
        with st.expander(f"📊 Detalles Cálculo {prefix}"):
            st.write(f"Total Mensual: **${fmt(total_month_sum)}**")
            st.write(f"Días Trabajados: **{machine_days}**")
            if fuel_daily_liters > 0:
                st.write(f"---")
                st.write(f"⛽ Petróleo: **{fuel_daily_liters} L/día**")
                st.write(f"= {fuel_daily_liters} x {machine_days} días x ${fuel_p}")
                st.write(f"= **${fmt(fuel_monthly_cost)}** (Mes)")
            
        return total_month_sum, total_hr

# Llamada a funciones pasando los días específicos
h_total_m, h_total_hr = create_machine_table("Harvester", col_tab1, h_days, h_total_hours, fuel_price)
f_total_m, f_total_hr = create_machine_table("Forwarder", col_tab2, f_days, f_total_hours, fuel_price)

# --- C. INDIRECTOS Y FAENA ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos (Faena)")

key_ind = "df_indirect_v16"
if key_ind not in st.session_state:
    st.session_state[key_ind] = pd.DataFrame([
        {"Ítem": "Arriendo Camionetas", "Detalle": "Total Mes (UF)", "Valor": 38.0},
        {"Ítem": "Combustible Camionetas", "Detalle": "Total Mes ($)", "Valor": 535000.0},
        {"Ítem": "Personal Apoyo", "Detalle": "Total Mes ($)", "Valor": 2164000.0},
        {"Ítem": "Instalaciones", "Detalle": "Total Mes ($)", "Valor": 560000.0},
        {"Ítem": "Pensión/Alojamiento", "Detalle": "Total Mes ($)", "Valor": 1890000.0},
        {"Ítem": "Otros Faena", "Detalle": "Total Mes ($)", "Valor": 100000.0},
    ])

with st.expander("📝 Editar Totales Indirectos", expanded=True):
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        edited_ind = st.data_editor(
            st.session_state[key_ind],
            key="editor_indirect",
            column_config={
                "Ítem": st.column_config.TextColumn(disabled=True),
                "Detalle": st.column_config.TextColumn(disabled=True),
                "Valor": st.column_config.NumberColumn("Valor Input")
            },
            hide_index=True,
            use_container_width=True
        )
        st.session_state[key_ind] = edited_ind
        save_config()
        
    with col_i2:
        st.info("Camionetas: Ingrese UF en 'Arriendo' y $ en 'Combustible' (Total Mes).")

# Procesar Indirectos
def get_ind(name):
    r = edited_ind[edited_ind["Ítem"] == name]
    return float(r.iloc[0]["Valor"]) if not r.empty else 0

total_shared = 0
uf_val = get_ind("Arriendo Camionetas")
total_shared += uf_val * current_uf
total_shared += get_ind("Combustible Camionetas")
total_shared += get_ind("Personal Apoyo")
total_shared += get_ind("Instalaciones")
total_shared += get_ind("Pensión/Alojamiento")
total_shared += get_ind("Otros Faena")

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

# --- D. RESULTADOS FINALES ---
st.divider()
st.subheader("📊 Resultados Consolidados")

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
    st.markdown("**Producción ($m^3$/hr)**")
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
        <div style="font-size:13px;">Prod. Real (Limitada): <b>{mr_sys:.1f} MR/hr</b></div>
    </div>
    """, unsafe_allow_html=True)

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
