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
@st.cache_data(ttl=3600) # Guardar en caché por 1 hora para no llamar a la API a cada rato
def get_uf_value():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # El valor más reciente está en la primera posición de la serie
            valor = data['serie'][0]['valor']
            fecha = data['serie'][0]['fecha'][:10] # Formato YYYY-MM-DD
            return valor, fecha
    except Exception as e:
        return None, None
    return None, None

# --- TÍTULO ---
st.title("🌲 Calculadora de Costos Harvester y Forwarder (M3 y MR)")

# --- BARRA LATERAL ---
st.sidebar.header("1. Parámetros Económicos")

# Lógica de UF Automática
uf_api, fecha_api = get_uf_value()

use_auto_uf = st.sidebar.checkbox("Usar UF del Banco Central", value=True, help="Intenta descargar el valor actual de la UF desde mindicador.cl")

if use_auto_uf and uf_api:
    uf_value = st.sidebar.number_input(f"Valor UF ({fecha_api})", value=uf_api, disabled=True, format="%.2f")
    st.sidebar.success(f"✅ UF Actualizada: ${uf_api:,.2f}")
else:
    if use_auto_uf and not uf_api:
        st.sidebar.warning("⚠️ No se pudo conectar a la API. Usando modo manual.")
    uf_value = st.sidebar.number_input("Valor UF (Manual)", value=39704.93, step=100.0, format="%.2f")

fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", value=774, step=10)

# SECCIÓN COMERCIAL (NUEVA)
st.sidebar.header("2. Comercial y Conversión")
st.sidebar.markdown("Configura la transformación de **M3 Sólido** a **Metro Ruma (MR)**.")
conversion_factor = st.sidebar.number_input("Factor Conversión (M3 a MR)", value=1.60, step=0.01, help="Fórmula: M3 * Factor = MR. Ej: 1 m3 sólido equivale a 1.6 m3 estéreos (MR).")
sales_price_mr = st.sidebar.number_input("Valor Venta ($/MR)", value=4500, step=100, help="Precio de venta por Metro Ruma")

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

# Métricas Principales con Cálculo de MR
col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo Sistema ($/Hora)", f"${fmt(system_cost_hr)}")

# Cálculo base para margen: Supongamos una productividad promedio de referencia (ej: 25 m3/hr)
# Esto es solo para mostrar un valor "snapshot", el detalle está en la tabla
ref_prod_m3 = 25 
ref_prod_mr = ref_prod_m3 * conversion_factor
ref_cost_mr = system_cost_hr / ref_prod_mr if ref_prod_mr > 0 else 0
ref_margin = (sales_price_mr - ref_cost_mr) / sales_price_mr if sales_price_mr > 0 else 0

col2.metric("Valor UF Hoy", f"${fmt(uf_value)}")
col3.metric("Factor Conversión", f"{conversion_factor:.2f}")
col4.metric("Precio Venta (MR)", f"${fmt(sales_price_mr)}")

# --- TABLA DE SENSIBILIDAD CON MARGEN ---
st.subheader("📈 Análisis de Rentabilidad y Conversión")
st.markdown(f"**Fórmula:** M3 Sólido * {conversion_factor} = Metro Ruma (MR)")

prod_m3_range = np.arange(10, 51, 1)

data_rows = []
for p_m3 in prod_m3_range:
    # Conversiones
    p_mr = p_m3 * conversion_factor
    
    # Costos Unitarios
    cost_sys_m3 = system_cost_hr / p_m3 if p_m3 > 0 else 0
    cost_sys_mr = system_cost_hr / p_mr if p_mr > 0 else 0 # Costo por MR es menor porque hay más unidades MR
    
    # Margen
    margin_value = sales_price_mr - cost_sys_mr
    margin_pct = (margin_value / sales_price_mr * 100) if sales_price_mr > 0 else 0
    
    data_rows.append({
        "Prod. M3/hr": p_m3,
        "Prod. MR/hr": p_mr,
        "Costo Sistema ($/M3)": cost_sys_m3,
        "Costo Sistema ($/MR)": cost_sys_mr,
        "Utilidad ($/MR)": margin_value,
        "Margen (%)": margin_pct
    })

df_sens = pd.DataFrame(data_rows)

# Pestañas
tab1, tab2 = st.tabs(["📊 Gráfico de Rentabilidad", "📋 Tabla Detallada"])

with tab1:
    fig = px.line(df_sens, x="Prod. MR/hr", y=["Costo Sistema ($/MR)", "Utilidad ($/MR)"],
                  title="Costo vs Utilidad por Metro Ruma (MR)",
                  labels={"value": "Pesos ($)", "Prod. MR/hr": "Productividad (MR/Hora)"},
                  markers=True)
    # Línea de Precio de Venta
    fig.add_hline(y=sales_price_mr, line_dash="dash", line_color="green", annotation_text="Precio Venta")
    fig.update_layout(separators=",.", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Formato condicional para el margen
    def color_margin(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'

    st.dataframe(df_sens.style
                 .format({
                     "Prod. M3/hr": "{:.0f}",
                     "Prod. MR/hr": "{:.1f}",
                     "Costo Sistema ($/M3)": lambda x: f"${fmt(x)}",
                     "Costo Sistema ($/MR)": lambda x: f"${fmt(x)}",
                     "Utilidad ($/MR)": lambda x: f"${fmt(x)}",
                     "Margen (%)": "{:.1f}%"
                 })
                 .applymap(color_margin, subset=["Margen (%)", "Utilidad ($/MR)"]), 
                 use_container_width=True)

# Descarga
csv = df_sens.to_csv(index=False).encode('utf-8')
st.download_button("💾 Descargar Tabla (Excel/CSV)", data=csv, file_name='rentabilidad_forestal.csv', mime='text/csv')
