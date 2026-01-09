import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import os
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Calculadora de Costos Forestales", layout="wide")

# Nombre del archivo de configuración
CONFIG_FILE = 'config.json'

# --- GESTIÓN DE PERSISTENCIA (GUARDAR/CARGAR) ---

def load_config():
    """Carga la configuración desde el archivo JSON si existe."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    """Guarda el estado actual de la sesión en el archivo JSON."""
    # Filtramos solo las claves que nos interesa guardar (evitamos guardar datos temporales de Streamlit)
    # Guardamos todo lo que esté en session_state
    config_data = {k: v for k, v in st.session_state.items() if k in EXPECTED_KEYS}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f)

# Lista de claves que queremos persistir (para filtrar y mantener orden)
EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    # Harvester
    "rent_Harvester", "fuel_consump_Harvester", "num_operators_Harvester", 
    "salary_Harvester", "maint_Harvester", "consumables_Harvester",
    # Forwarder
    "rent_Forwarder", "fuel_consump_Forwarder", "num_operators_Forwarder", 
    "salary_Forwarder", "maint_Forwarder", "consumables_Forwarder",
    # Shared
    "pickup_rent", "pickup_fuel", "support_staff", 
    "facilities", "pension", "others", "alloc_method", "h_share_pct_manual"
]

# Inicialización de Session State con valores del archivo (SOLO AL INICIO)
if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            st.session_state[key] = saved_config[key]
    st.session_state['config_loaded'] = True

# --- FUNCIONES DE FORMATO Y UTILIDADES ---

def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:,.0f}".replace(",", ".")
    return x

@st.cache_data(ttl=3600)
def get_uf_value():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            valor = data['serie'][0]['valor']
            fecha = data['serie'][0]['fecha'][:10]
            return valor, fecha
    except:
        return None, None
    return None, None

# --- INTERFAZ GRÁFICA ---

st.title("🌲 Calculadora de Costos Forestales (Autoguardado)")
st.caption("Los cambios que realices se guardarán automáticamente para tu próxima sesión.")

# --- BARRA LATERAL ---
st.sidebar.header("1. Parámetros Económicos")

# UF
uf_api, fecha_api = get_uf_value()

# Checkbox con autoguardado (key vinculada a session_state)
use_auto_uf = st.sidebar.checkbox("Usar UF del Banco Central", value=True, key="use_auto_uf", on_change=save_config)

current_uf_val = 39704.93 # Valor fallback
if use_auto_uf and uf_api:
    # Mostramos el valor de la API pero no permitimos editar
    st.sidebar.number_input(f"Valor UF ({fecha_api})", value=uf_api, disabled=True, format="%.2f")
    current_uf_val = uf_api
else:
    # Si no usa API, usa el valor manual guardado (key="uf_manual")
    current_uf_val = st.sidebar.number_input("Valor UF (Manual)", value=39704.93, step=100.0, format="%.2f", key="uf_manual", on_change=save_config)

fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

# COMERCIAL
st.sidebar.header("2. Comercial y Conversión")
conversion_factor = st.sidebar.number_input("Factor Conversión (M3 / Factor = MR)", value=0.65, step=0.01, format="%.2f", key="conversion_factor", on_change=save_config)
sales_price_mr = st.sidebar.number_input("Valor Venta Total ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)

st.sidebar.markdown("**Distribución de la Venta**")
h_rev_pct = st.sidebar.slider("% Venta para Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
f_rev_pct = 100 - h_rev_pct

h_price_mr = sales_price_mr * (h_rev_pct / 100)
f_price_mr = sales_price_mr * (f_rev_pct / 100)
st.sidebar.info(f"Harvester: ${fmt(h_price_mr)}/MR | Forwarder: ${fmt(f_price_mr)}/MR")

# JORNADA
st.sidebar.header("3. Configuración de Jornada")

st.sidebar.subheader("🚜 Harvester")
h_days_month = st.sidebar.number_input("Días/Mes (Harvester)", value=28, step=1, key="h_days_month", on_change=save_config)
h_hours_day = st.sidebar.number_input("Horas/Día (Harvester)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
h_monthly_hours = h_days_month * h_hours_day
st.sidebar.caption(f"⏱ Horas: **{fmt(h_monthly_hours)}**")

st.sidebar.subheader("🚜 Forwarder")
f_days_month = st.sidebar.number_input("Días/Mes (Forwarder)", value=25, step=1, key="f_days_month", on_change=save_config)
f_hours_day = st.sidebar.number_input("Horas/Día (Forwarder)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
f_monthly_hours = f_days_month * f_hours_day
st.sidebar.caption(f"⏱ Horas: **{fmt(f_monthly_hours)}**")

# --- ENTRADA DE DATOS DE MAQUINARIA ---

def get_machine_inputs(prefix, hours_month):
    with st.expander(f"⚙️ Costos Operacionales: {prefix}", expanded=False):
        col1, col2 = st.columns(2)
        # Definimos keys únicas usando el prefijo
        k_rent = f"rent_{prefix}"
        k_fuel = f"fuel_consump_{prefix}"
        k_op_num = f"num_operators_{prefix}"
        k_salary = f"salary_{prefix}"
        k_maint = f"maint_{prefix}"
        k_consum = f"consumables_{prefix}"

        # Valores por defecto según máquina si no existen en config
        def_rent = 10900000 if prefix=="Harvester" else 8000000
        def_fuel = 20.0 if prefix=="Harvester" else 15.0
        def_maint = 5500 if prefix=="Harvester" else 3500
        def_consum = 410000 if prefix=="Harvester" else 200000
        def_op_num = 2 if prefix=="Harvester" else 1

        with col1:
            rent = st.number_input(f"Arriendo Mensual {prefix} ($)", value=def_rent, step=100000, key=k_rent, on_change=save_config)
        with col2:
            fuel_consump = st.number_input(f"Consumo Petróleo (L/hr) {prefix}", value=def_fuel, step=1.0, key=k_fuel, on_change=save_config)
            
        st.markdown(f"##### 👷‍♂️ Operadores {prefix}")
        col_op1, col_op2, col_op3 = st.columns([1, 1, 1])
        with col_op1:
            num_operators = st.number_input(f"N° Operadores {prefix}", value=def_op_num, step=1, min_value=0, key=k_op_num, on_change=save_config)
        with col_op2:
            salary_per_op = st.number_input(f"Sueldo Unitario {prefix} ($)", value=1923721, step=50000, key=k_salary, on_change=save_config)
        with col_op3:
            total_salary = num_operators * salary_per_op
            st.metric(f"Total Sueldos {prefix}", f"${fmt(total_salary)}")

        st.markdown("##### 🔧 Mantención")
        col3, col4 = st.columns(2)
        with col3:
            maint_hourly = st.number_input(f"Costo Mantención ($/hr) {prefix}", value=def_maint, step=100, key=k_maint, on_change=save_config)
        with col4:
            consumables = st.number_input(f"Consumibles Mes {prefix} ($)", value=def_consum, step=10000, key=k_consum, on_change=save_config)
        
        return {"rent": rent, "salary": total_salary, "fuel_l_hr": fuel_consump, 
                "maintenance_hr": maint_hourly, "consumables_month": consumables, "hours_month": hours_month}

# --- COSTOS COMPARTIDOS ---
def get_shared_inputs(h_hours, f_hours):
    with st.expander("🏢 Costos Fijos Compartidos (Faena)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            pickup_rent = st.number_input("Arriendo Camionetas ($)", value=1504816, step=10000, key="pickup_rent", on_change=save_config)
            pickup_fuel = st.number_input("Combustible Camionetas ($)", value=535104, step=10000, key="pickup_fuel", on_change=save_config)
            support_staff = st.number_input("Personal Apoyo ($)", value=2164000, step=50000, key="support_staff", on_change=save_config)
        with col2:
            facilities = st.number_input("Instalación/Gastos Adm ($)", value=560000, step=10000, key="facilities", on_change=save_config)
            pension = st.number_input("Pensión/Alojamiento ($)", value=1890000, step=50000, key="pension", on_change=save_config)
            others = st.number_input("Otros Costos Fijos ($)", value=100000, step=10000, key="others", on_change=save_config)
        
        total_shared = pickup_rent + pickup_fuel + support_staff + facilities + pension + others
        
        st.markdown("---")
        # Radio button también con persistencia
        # Nota: Los radio buttons devuelven strings, lo mapearemos si es necesario, pero st guarda el string
        alloc_options = ["Porcentaje Manual", "Proporcional a Horas"]
        # Buscar índice por defecto
        def_idx = 0
        if "alloc_method" in st.session_state and st.session_state["alloc_method"] in alloc_options:
             def_idx = alloc_options.index(st.session_state["alloc_method"])

        alloc_method = st.radio("Método de Asignación", alloc_options, index=def_idx, key="alloc_method", horizontal=True, on_change=save_config)
        
        if alloc_method == "Porcentaje Manual":
            h_share_pct = st.slider("% Asignado a Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
            f_share_pct = 1.0 - h_share_pct
        else:
            total_h = h_hours + f_hours
            h_share_pct = h_hours / total_h if total_h > 0 else 0.5
            f_share_pct = f_hours / total_h if total_h > 0 else 0.5
            st.info(f"Asignación Auto: Harvester {h_share_pct*100:.1f}% | Forwarder {f_share_pct*100:.1f}%")
            
        return total_shared, h_share_pct, f_share_pct

# --- OBTENCIÓN DE DATOS Y CÁLCULOS ---
harvester_data = get_machine_inputs("Harvester", h_monthly_hours)
forwarder_data = get_machine_inputs("Forwarder", f_monthly_hours)
shared_total, h_share, f_share = get_shared_inputs(h_monthly_hours, f_monthly_hours)

def calculate_costs(data, fuel_price, shared_alloc):
    if data['hours_month'] == 0: return 0
    fixed = (data['rent'] + data['salary'] + data['consumables_month'] + shared_alloc) / data['hours_month']
    variable = (data['fuel_l_hr'] * fuel_price) + data['maintenance_hr']
    return fixed + variable

h_cost_hr = calculate_costs(harvester_data, fuel_price, shared_total * h_share)
f_cost_hr = calculate_costs(forwarder_data, fuel_price, shared_total * f_share)
system_cost_hr = h_cost_hr + f_cost_hr

# --- RESULTADOS ---
st.divider()
st.subheader("📊 Resultados Generales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo Sistema Total ($/Hora)", f"${fmt(system_cost_hr)}")
col2.metric("Valor UF", f"${fmt(current_uf_val)}")
col3.metric("Factor (M3/Factor=MR)", f"{conversion_factor:.2f}")
col4.metric("Precio Venta (MR)", f"${fmt(sales_price_mr)}")

st.subheader("📈 Rentabilidad por Máquina")
st.markdown(f"**Distribución de Ingresos:** Harvester **{h_rev_pct}%** (${fmt(h_price_mr)}) | Forwarder **{f_rev_pct}%** (${fmt(f_price_mr)})")

prod_m3_range = np.arange(10, 51, 1)
data_rows = []

for p_m3 in prod_m3_range:
    # Conversión M3 / Factor = MR
    p_mr = p_m3 / conversion_factor if conversion_factor != 0 else 0
    
    # Costos Unitarios
    cost_h_mr = h_cost_hr / p_mr if p_mr > 0 else 0
    cost_f_mr = f_cost_hr / p_mr if p_mr > 0 else 0
    cost_sys_mr = cost_h_mr + cost_f_mr
    
    # Utilidad
    margin_h = h_price_mr - cost_h_mr
    margin_f = f_price_mr - cost_f_mr
    margin_sys = sales_price_mr - cost_sys_mr
    
    # Margen %
    pct_h = (margin_h / h_price_mr * 100) if h_price_mr > 0 else 0
    pct_f = (margin_f / f_price_mr * 100) if f_price_mr > 0 else 0
    pct_sys = (margin_sys / sales_price_mr * 100) if sales_price_mr > 0 else 0
    
    data_rows.append({
        "Prod. M3/hr": p_m3,
        "Prod. MR/hr": p_mr,
        "Costo H ($/MR)": cost_h_mr,
        "Utilidad H ($/MR)": margin_h,
        "Margen H (%)": pct_h,
        "Costo F ($/MR)": cost_f_mr,
        "Utilidad F ($/MR)": margin_f,
        "Margen F (%)": pct_f,
        "Utilidad Total ($/MR)": margin_sys,
        "Margen Total (%)": pct_sys
    })

df_sens = pd.DataFrame(data_rows)

tab1, tab2 = st.tabs(["📊 Gráfico", "📋 Tabla Detallada"])

with tab1:
    fig = px.line(df_sens, x="Prod. MR/hr", y=["Margen H (%)", "Margen F (%)", "Margen Total (%)"],
                  title="Margen de Ganancia (%)", markers=True)
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(separators=",.", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    def color_margin(val):
        color = '#d63031' if val < 0 else '#00b894'
        return f'color: {color}; font-weight: bold;'
        
    cols = ["Prod. MR/hr", "Costo H ($/MR)", "Utilidad H ($/MR)", "Margen H (%)", 
            "Costo F ($/MR)", "Utilidad F ($/MR)", "Margen F (%)", "Margen Total (%)"]
            
    st.dataframe(df_sens[cols].style.format({
        "Prod. MR/hr": "{:.1f}",
        "Costo H ($/MR)": lambda x: f"${fmt(x)}",
        "Utilidad H ($/MR)": lambda x: f"${fmt(x)}",
        "Margen H (%)": "{:.1f}%",
        "Costo F ($/MR)": lambda x: f"${fmt(x)}",
        "Utilidad F ($/MR)": lambda x: f"${fmt(x)}",
        "Margen F (%)": "{:.1f}%",
        "Margen Total (%)": "{:.1f}%"
    }).applymap(color_margin, subset=["Utilidad H ($/MR)", "Margen H (%)", "Utilidad F ($/MR)", "Margen F (%)", "Margen Total (%)"]),
    use_container_width=True)

csv = df_sens.to_csv(index=False).encode('utf-8')
st.download_button("💾 Descargar CSV", data=csv, file_name='costos_guardados.csv', mime='text/csv')
