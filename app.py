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

CONFIG_FILE = 'config_totales_v2.json'

# --- GESTIÓN DE PERSISTENCIA ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    config_data = {k: v for k, v in st.session_state.items() if k in EXPECTED_KEYS}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f)

# Claves a guardar
EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    # Harvester (Totales)
    "rent_total_Harvester", "fuel_liters_total_Harvester", "salary_total_Harvester", 
    "maint_total_Harvester", "consumables_total_Harvester", "others_total_Harvester",
    # Forwarder (Totales)
    "rent_total_Forwarder", "fuel_liters_total_Forwarder", "salary_total_Forwarder", 
    "maint_total_Forwarder", "consumables_total_Forwarder", "others_total_Forwarder",
    # Shared
    "pickup_rent_uf", "pickup_fuel", "support_staff", 
    "facilities", "pension", "others_shared", "alloc_method", "h_share_pct_manual"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            st.session_state[key] = saved_config[key]
    st.session_state['config_loaded'] = True

# --- UTILIDADES ---
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
            return data['serie'][0]['valor'], data['serie'][0]['fecha'][:10]
    except:
        pass
    return None, None

# --- TÍTULO ---
st.title("🌲 Calculadora Forestal: Detalle por Ítem")
st.markdown("Ingresa los datos solicitados. El **Combustible** se pide en Litros y las **Camionetas** en UF.")

# --- SIDEBAR ---
st.sidebar.header("1. Parámetros Económicos")
uf_api, fecha_api = get_uf_value()
use_auto_uf = st.sidebar.checkbox("Usar UF Banco Central", value=True, key="use_auto_uf", on_change=save_config)

if use_auto_uf and uf_api:
    st.sidebar.number_input(f"Valor UF ({fecha_api})", value=uf_api, disabled=True, format="%.2f")
    current_uf = uf_api
else:
    current_uf = st.sidebar.number_input("Valor UF (Manual)", value=39704.93, step=100.0, format="%.2f", key="uf_manual", on_change=save_config)

fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

st.sidebar.header("2. Comercial")
conversion_factor = st.sidebar.number_input("Factor (M3 / Factor = MR)", value=0.65, step=0.01, format="%.2f", key="conversion_factor", on_change=save_config)
sales_price_mr = st.sidebar.number_input("Valor Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)

st.sidebar.markdown("**Distribución Venta**")
h_rev_pct = st.sidebar.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
h_price_mr = sales_price_mr * (h_rev_pct / 100)
f_price_mr = sales_price_mr * ((100 - h_rev_pct) / 100)

st.sidebar.header("3. Jornada Laboral")
h_days = st.sidebar.number_input("Días/Mes (H)", value=28, step=1, key="h_days_month", on_change=save_config)
h_hours_day = st.sidebar.number_input("Horas/Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
h_total_hours = h_days * h_hours_day
st.sidebar.caption(f"Total Horas Harvester: {fmt(h_total_hours)}")

f_days = st.sidebar.number_input("Días/Mes (F)", value=25, step=1, key="f_days_month", on_change=save_config)
f_hours_day = st.sidebar.number_input("Horas/Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
f_total_hours = f_days * f_hours_day
st.sidebar.caption(f"Total Horas Forwarder: {fmt(f_total_hours)}")

# --- ENTRADA DE DATOS MAQUINARIA ---

def get_machine_totals(prefix, hours_month, fuel_price_val):
    with st.expander(f"🚜 Costos Mensuales: {prefix}", expanded=True):
        
        # Valores por defecto
        def_rent = 10900000 if prefix=="Harvester" else 8000000
        def_sal = 3800000 if prefix=="Harvester" else 1900000
        def_liters = 5600.0 if prefix=="Harvester" else 3000.0 # Litros aprox
        def_maint = 1500000 if prefix=="Harvester" else 900000
        def_consum = 410000 if prefix=="Harvester" else 200000
        
        col1, col2 = st.columns(2)
        
        # 1. ARRIENDO
        with col1:
            rent_total = st.number_input(f"Arriendo ($/Mes) {prefix}", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
        with col2:
            st.caption(f"Costo Hora: ${fmt(rent_total/hours_month if hours_month else 0)}")

        # 2. SUELDOS
        with col1:
            salary_total = st.number_input(f"Sueldos Operadores ($/Mes) {prefix}", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)
        with col2:
            st.caption(f"Costo Hora: ${fmt(salary_total/hours_month if hours_month else 0)}")
        
        # 3. COMBUSTIBLE (EN LITROS)
        with col1:
            # CAMBIO: Ingreso en Litros
            liters_total = st.number_input(f"Petróleo (Litros/Mes) {prefix}", value=def_liters, step=100.0, key=f"fuel_liters_total_{prefix}", on_change=save_config)
        with col2:
            # Cálculo a pesos
            fuel_cost_total = liters_total * fuel_price_val
            st.markdown(f"Costo Petróleo: **${fmt(fuel_cost_total)}**")
            st.caption(f"Consumo: {liters_total/hours_month:.1f} L/hr aprox.")

        # 4. MANTENCIÓN
        with col1:
            maint_total = st.number_input(f"Mantención ($/Mes) {prefix}", value=def_maint, step=100000, key=f"maint_total_{prefix}", on_change=save_config)
        with col2:
             st.caption(f"Costo Hora: ${fmt(maint_total/hours_month if hours_month else 0)}")

        # 5. CONSUMIBLES
        with col1:
            consumables_total = st.number_input(f"Consumibles ($/Mes) {prefix}", value=def_consum, step=50000, key=f"consumables_total_{prefix}", on_change=save_config)
        
        # 6. OTROS
        with col1:
            others_total = st.number_input(f"Otros ($/Mes) {prefix}", value=0, step=10000, key=f"others_total_{prefix}", on_change=save_config)

        total_month = rent_total + salary_total + fuel_cost_total + maint_total + consumables_total + others_total
        
        st.info(f"Total Mensual {prefix}: ${fmt(total_month)}")
        
        return {
            "total_month": total_month,
            "hours_month": hours_month
        }

# --- COSTOS COMPARTIDOS (FAENA) ---
def get_shared_totals(h_hours, f_hours, current_uf_val):
    with st.expander("🏢 Costos Totales Faena (Compartidos)", expanded=False):
        
        col1, col2 = st.columns(2)
        
        # 1. CAMIONETAS (EN UF)
        with col1:
            # CAMBIO: Ingreso en UF
            p_rent_uf = st.number_input("Arriendo Camionetas (UF/Mes)", value=38.0, step=0.5, key="pickup_rent_uf", on_change=save_config)
            p_rent_clp = p_rent_uf * current_uf_val
        with col2:
            st.markdown(f"Costo Arriendo: **${fmt(p_rent_clp)}**")
            st.caption(f"Calculado a UF: ${fmt(current_uf_val)}")

        # 2. OTROS ÍTEMS
        with col1:
            p_fuel = st.number_input("Combustible Camionetas ($/Mes)", value=535104, step=50000, key="pickup_fuel", on_change=save_config)
            staff = st.number_input("Personal Apoyo ($/Mes)", value=2164000, step=100000, help="Mecánico, Prevencionista, Jefe Faena", key="support_staff", on_change=save_config)
        with col2:
            fac = st.number_input("Instalación/Gastos Adm ($/Mes)", value=560000, step=50000, key="facilities", on_change=save_config)
            pen = st.number_input("Pensión/Alojamiento ($/Mes)", value=1890000, step=50000, key="pension", on_change=save_config)
            oth = st.number_input("Otros Faena ($/Mes)", value=100000, step=10000, key="others_shared", on_change=save_config)
        
        total_shared = p_rent_clp + p_fuel + staff + fac + pen + oth
        st.success(f"Gran Total Faena: ${fmt(total_shared)}")

        # Asignación
        st.markdown("---")
        def_idx = 0
        options = ["Porcentaje Manual", "Proporcional a Horas"]
        if "alloc_method" in st.session_state and st.session_state["alloc_method"] in options:
            def_idx = options.index(st.session_state["alloc_method"])
            
        alloc_method = st.radio("Método Distribución Costos Fijos", options, index=def_idx, horizontal=True, key="alloc_method", on_change=save_config)
        
        if alloc_method == "Porcentaje Manual":
            h_pct = st.slider("% Asignado a Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
            f_pct = 1.0 - h_pct
        else:
            tot_h = h_hours + f_hours
            h_pct = h_hours / tot_h if tot_h > 0 else 0.5
            f_pct = f_hours / tot_h if tot_h > 0 else 0.5
            st.caption(f"Asignación Automática: H {h_pct*100:.1f}% - F {f_pct*100:.1f}%")
            
        return total_shared, h_pct, f_pct

# --- PROCESAMIENTO ---
h_data = get_machine_totals("Harvester", h_total_hours, fuel_price)
f_data = get_machine_totals("Forwarder", f_total_hours, fuel_price)
shared_total, h_share_pct, f_share_pct = get_shared_totals(h_total_hours, f_total_hours, current_uf)

# Cálculo Final Hora
def calc_final_hourly(machine_data, shared_total, share_pct):
    if machine_data["hours_month"] == 0: return 0
    direct = machine_data["total_month"]
    allocated = shared_total * share_pct
    return (direct + allocated) / machine_data["hours_month"]

h_final_cost_hr = calc_final_hourly(h_data, shared_total, h_share_pct)
f_final_cost_hr = calc_final_hourly(f_data, shared_total, f_share_pct)
system_cost_hr = h_final_cost_hr + f_final_cost_hr

# --- RESULTADOS ---
st.divider()
st.subheader("📊 Resultados Consolidados")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Costo Hora Harvester", f"${fmt(h_final_cost_hr)}")
c2.metric("Costo Hora Forwarder", f"${fmt(f_final_cost_hr)}")
c3.metric("Costo Hora Sistema", f"${fmt(system_cost_hr)}")
c4.metric("Gasto Mensual Total Faena", f"${fmt(h_data['total_month'] + f_data['total_month'] + shared_total)}")

st.subheader("📈 Análisis de Rentabilidad")
st.caption(f"Fórmula: M3 / {conversion_factor} = MR")

prod_m3 = np.arange(10, 51, 1)
rows = []
for m3 in prod_m3:
    mr = m3 / conversion_factor if conversion_factor else 0
    
    c_h_mr = h_final_cost_hr / mr if mr > 0 else 0
    c_f_mr = f_final_cost_hr / mr if mr > 0 else 0
    c_sys_mr = c_h_mr + c_f_mr
    
    m_h = h_price_mr - c_h_mr
    m_f = f_price_mr - c_f_mr
    m_sys = sales_price_mr - c_sys_mr
    
    rows.append({
        "Prod M3": m3,
        "Prod MR": mr,
        "Costo H ($/MR)": c_h_mr,
        "Utilidad H ($)": m_h,
        "Margen H %": (m_h/h_price_mr*100) if h_price_mr else 0,
        "Costo F ($/MR)": c_f_mr,
        "Utilidad F ($)": m_f,
        "Margen F %": (m_f/f_price_mr*100) if f_price_mr else 0,
        "Utilidad Total ($)": m_sys,
        "Margen Total %": (m_sys/sales_price_mr*100) if sales_price_mr else 0
    })

df = pd.DataFrame(rows)

tab1, tab2 = st.tabs(["Gráfico", "Tabla Detallada"])

with tab1:
    fig = px.line(df, x="Prod MR", y=["Margen H %", "Margen F %", "Margen Total %"], 
                  title="Margen (%) vs Productividad (MR/hr)", markers=True)
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(separators=",.", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    def style_neg(v):
        color = '#d63031' if v < 0 else '#00b894'
        return f'color: {color}; font-weight: bold;'

    cols_show = ["Prod MR", "Costo H ($/MR)", "Utilidad H ($)", "Margen H %", 
                 "Costo F ($/MR)", "Utilidad F ($)", "Margen F %", "Margen Total %"]
            
    st.dataframe(df[cols_show].style.format({
        "Prod MR": "{:.1f}",
        "Costo H ($/MR)": lambda x: f"${fmt(x)}",
        "Utilidad H ($)": lambda x: f"${fmt(x)}",
        "Margen H %": "{:.1f}%",
        "Costo F ($/MR)": lambda x: f"${fmt(x)}",
        "Utilidad F ($)": lambda x: f"${fmt(x)}",
        "Margen F %": "{:.1f}%",
        "Margen Total %": "{:.1f}%"
    }).applymap(style_neg, subset=["Utilidad H ($)", "Margen H %", "Utilidad F ($)", "Margen F %", "Margen Total %"]), use_container_width=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("💾 Guardar CSV", csv, "analisis_faena.csv", "text/csv")
