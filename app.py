import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Calculadora de Costos Forestales", layout="wide")

# Función de formato con separador de miles (punto)
def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:,.0f}".replace(",", ".")
    return x

# --- FUNCIÓN PARA OBTENER UF ---
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
    except Exception as e:
        return None, None
    return None, None

# --- TÍTULO ---
st.title("🌲 Calculadora de Costos Harvester y Forwarder (M3 y MR)")

# --- BARRA LATERAL ---
st.sidebar.header("1. Parámetros Económicos")

# UF
uf_api, fecha_api = get_uf_value()
use_auto_uf = st.sidebar.checkbox("Usar UF del Banco Central", value=True)

if use_auto_uf and uf_api:
    uf_value = st.sidebar.number_input(f"Valor UF ({fecha_api})", value=uf_api, disabled=True, format="%.2f")
else:
    uf_value = st.sidebar.number_input("Valor UF (Manual)", value=39704.93, step=100.0, format="%.2f")

fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", value=774, step=10)

# SECCIÓN COMERCIAL (MODIFICADA)
st.sidebar.header("2. Comercial y Conversión")
st.sidebar.markdown("Configura la transformación de M3 a MR y la venta.")

# Fórmula corregida M3 / Factor = MR
conversion_factor = st.sidebar.number_input("Factor Conversión (M3 / Factor = MR)", value=0.65, step=0.01, format="%.2f", help="Fórmula: M3 Sólido dividido por este factor da los Metros Ruma. Ej: Si el factor es 0.65 (densidad), entonces 1 / 0.65 = 1.54 MR.")
sales_price_mr = st.sidebar.number_input("Valor Venta Total ($/MR)", value=4500, step=100, help="Precio de venta total por Metro Ruma")

st.sidebar.markdown("**Distribución de la Venta**")
st.sidebar.caption("Define qué % del precio de venta se asigna a cada máquina.")
h_rev_pct = st.sidebar.slider("% Venta para Harvester", 0, 100, 70, help="El porcentaje restante será para el Forwarder")
f_rev_pct = 100 - h_rev_pct

# Precios asignados
h_price_mr = sales_price_mr * (h_rev_pct / 100)
f_price_mr = sales_price_mr * (f_rev_pct / 100)

st.sidebar.info(f"Harvester: ${fmt(h_price_mr)}/MR | Forwarder: ${fmt(f_price_mr)}/MR")

st.sidebar.header("3. Configuración de Jornada")

# HARVESTER
st.sidebar.subheader("🚜 Harvester")
h_days_month = st.sidebar.number_input("Días/Mes (Harvester)", value=28, step=1)
h_hours_day = st.sidebar.number_input("Horas/Día (Harvester)", value=10.0, step=0.5)
h_monthly_hours = h_days_month * h_hours_day
st.sidebar.caption(f"⏱ Horas Mensuales Harvester: **{fmt(h_monthly_hours)}**")

# FORWARDER
st.sidebar.subheader("🚜 Forwarder")
f_days_month = st.sidebar.number_input("Días/Mes (Forwarder)", value=25, step=1)
f_hours_day = st.sidebar.number_input("Horas/Día (Forwarder)", value=9.0, step=0.5)
f_monthly_hours = f_days_month * f_hours_day
st.sidebar.caption(f"⏱ Horas Mensuales Forwarder: **{fmt(f_monthly_hours)}**")

# --- INPUTS DE COSTOS ---
def get_machine_inputs(prefix, hours_month):
    with st.expander(f"⚙️ Costos Operacionales: {prefix}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rent = st.number_input(f"Arriendo Mensual {prefix} ($)", value=10900000 if prefix=="Harvester" else 8000000, step=100000)
        with col2:
            fuel_consump = st.number_input(f"Consumo Petróleo (L/hr) {prefix}", value=20.0 if prefix=="Harvester" else 15.0, step=1.0)
            
        st.markdown(f"##### 👷‍♂️ Operadores {prefix}")
        col_op1, col_op2, col_op3 = st.columns([1, 1, 1])
        with col_op1:
            num_operators = st.number_input(f"N° Operadores {prefix}", value=2 if prefix=="Harvester" else 1, step=1, min_value=0)
        with col_op2:
            salary_per_op = st.number_input(f"Sueldo Unitario {prefix} ($)", value=1923721, step=50000, key=f"salary_{prefix}")
        with col_op3:
            total_salary = num_operators * salary_per_op
            st.metric(f"Total Sueldos {prefix}", f"${fmt(total_salary)}")

        st.markdown("##### 🔧 Mantención")
        col3, col4 = st.columns(2)
        with col3:
            maint_hourly = st.number_input(f"Costo Mantención ($/hr) {prefix}", value=5500 if prefix=="Harvester" else 3500, step=100)
        with col4:
            consumables = st.number_input(f"Consumibles Mes {prefix} ($)", value=410000 if prefix=="Harvester" else 200000, step=10000)
        
        return {"rent": rent, "salary": total_salary, "fuel_l_hr": fuel_consump, 
                "maintenance_hr": maint_hourly, "consumables_month": consumables, "hours_month": hours_month}

def get_shared_inputs(h_hours, f_hours):
    with st.expander("🏢 Costos Fijos Compartidos (Faena)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            pickup_rent = st.number_input("Arriendo Camionetas ($)", value=1504816, step=10000)
            pickup_fuel = st.number_input("Combustible Camionetas ($)", value=535104, step=10000)
            support_staff = st.number_input("Personal Apoyo ($)", value=2164000, step=50000)
        with col2:
            facilities = st.number_input("Instalación/Gastos Adm ($)", value=560000, step=10000)
            pension = st.number_input("Pensión/Alojamiento ($)", value=1890000, step=50000)
            others = st.number_input("Otros Costos Fijos ($)", value=100000, step=10000)
        
        total_shared = pickup_rent + pickup_fuel + support_staff + facilities + pension + others
        
        st.markdown("---")
        alloc_method = st.radio("Método de Asignación", ["Porcentaje Manual", "Proporcional a Horas"], horizontal=True)
        if alloc_method == "Porcentaje Manual":
            h_share_pct = st.slider("% Asignado a Harvester", 0, 100, 60) / 100.0
            f_share_pct = 1.0 - h_share_pct
        else:
            total_h = h_hours + f_hours
            h_share_pct = h_hours / total_h if total_h > 0 else 0.5
            f_share_pct = f_hours / total_h if total_h > 0 else 0.5
            st.info(f"Asignación Auto: Harvester {h_share_pct*100:.1f}% | Forwarder {f_share_pct*100:.1f}%")
            
        return total_shared, h_share_pct, f_share_pct

harvester_data = get_machine_inputs("Harvester", h_monthly_hours)
forwarder_data = get_machine_inputs("Forwarder", f_monthly_hours)
shared_total, h_share, f_share = get_shared_inputs(h_monthly_hours, f_monthly_hours)

# --- CÁLCULOS ---
def calculate_costs(data, fuel_price, shared_alloc):
    if data['hours_month'] == 0: return 0
    fixed = (data['rent'] + data['salary'] + data['consumables_month'] + shared_alloc) / data['hours_month']
    variable = (data['fuel_l_hr'] * fuel_price) + data['maintenance_hr']
    return fixed + variable

h_cost_hr = calculate_costs(harvester_data, fuel_price, shared_total * h_share)
f_cost_hr = calculate_costs(forwarder_data, fuel_price, shared_total * f_share)
system_cost_hr = h_cost_hr + f_cost_hr

# --- RESULTADOS PRINCIPALES ---
st.divider()
st.subheader("📊 Resultados Generales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo Sistema Total ($/Hora)", f"${fmt(system_cost_hr)}")
col2.metric("Valor UF", f"${fmt(uf_value)}")
col3.metric("Factor (M3/Factor=MR)", f"{conversion_factor:.2f}")
col4.metric("Precio Venta (MR)", f"${fmt(sales_price_mr)}")

# --- TABLA DE SENSIBILIDAD ---
st.subheader("📈 Rentabilidad por Máquina")
st.markdown(f"**Distribución de Ingresos:** Harvester **{h_rev_pct}%** (${fmt(h_price_mr)}) | Forwarder **{f_rev_pct}%** (${fmt(f_price_mr)})")

prod_m3_range = np.arange(10, 51, 1)

data_rows = []
for p_m3 in prod_m3_range:
    # Conversión Corregida: M3 / Factor = MR
    p_mr = p_m3 / conversion_factor if conversion_factor != 0 else 0
    
    # 1. Costos Unitarios por MR
    cost_h_mr = h_cost_hr / p_mr if p_mr > 0 else 0
    cost_f_mr = f_cost_hr / p_mr if p_mr > 0 else 0
    cost_sys_mr = cost_h_mr + cost_f_mr
    
    # 2. Utilidad por MR (Precio Asignado - Costo)
    margin_h = h_price_mr - cost_h_mr
    margin_f = f_price_mr - cost_f_mr
    margin_sys = sales_price_mr - cost_sys_mr
    
    # 3. Margen %
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

# Pestañas
tab1, tab2 = st.tabs(["📊 Gráfico de Márgenes", "📋 Tabla Detallada"])

with tab1:
    fig = px.line(df_sens, x="Prod. MR/hr", y=["Margen H (%)", "Margen F (%)", "Margen Total (%)"],
                  title="Margen de Ganancia (%) según Productividad",
                  labels={"value": "Margen (%)", "Prod. MR/hr": "Productividad (MR/Hora)"},
                  markers=True)
    fig.add_hline(y=0, line_dash="solid", line_color="black") # Línea base 0%
    fig.update_layout(separators=",.", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    def color_margin(val):
        color = '#d63031' if val < 0 else '#00b894'
        return f'color: {color}; font-weight: bold;'

    # Seleccionamos columnas clave para mostrar
    cols_to_show = ["Prod. MR/hr", "Costo H ($/MR)", "Utilidad H ($/MR)", "Margen H (%)", 
                    "Costo F ($/MR)", "Utilidad F ($/MR)", "Margen F (%)", "Margen Total (%)"]
    
    st.dataframe(df_sens[cols_to_show].style
                 .format({
                     "Prod. MR/hr": "{:.1f}",
                     "Costo H ($/MR)": lambda x: f"${fmt(x)}",
                     "Utilidad H ($/MR)": lambda x: f"${fmt(x)}",
                     "Margen H (%)": "{:.1f}%",
                     "Costo F ($/MR)": lambda x: f"${fmt(x)}",
                     "Utilidad F ($/MR)": lambda x: f"${fmt(x)}",
                     "Margen F (%)": "{:.1f}%",
                     "Margen Total (%)": "{:.1f}%"
                 })
                 .applymap(color_margin, subset=["Utilidad H ($/MR)", "Margen H (%)", "Utilidad F ($/MR)", "Margen F (%)", "Margen Total (%)"]), 
                 use_container_width=True)

# Descarga
csv = df_sens.to_csv(index=False).encode('utf-8')
st.download_button("💾 Descargar Reporte (CSV)", data=csv, file_name='margenes_maquinaria.csv', mime='text/csv')
