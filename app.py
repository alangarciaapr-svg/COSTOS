import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Calculadora de Costos Forestales", layout="wide")

st.title("🌲 Calculadora de Costos Harvester y Forwarder")
st.markdown("""
Esta aplicación permite estimar los costos horarios y unitarios ($/m^3$) para operaciones forestales.
Modifique los parámetros en el panel lateral y vea los resultados actualizados en tiempo real.
""")

# --- SIDEBAR: Configuration Parameters ---
st.sidebar.header("1. Parámetros Económicos")
uf_value = st.sidebar.number_input("Valor UF ($)", value=39704.93, step=100.0)
fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", value=774, step=10)

st.sidebar.header("2. Configuración de Jornada")
days_per_month = st.sidebar.number_input("Días por Mes", value=30, step=1)
# Harvester Shift
st.sidebar.subheader("Harvester")
harvester_daily_hours = st.sidebar.number_input("Horas/Día (Harvester)", value=10.0, step=0.5)
harvester_monthly_hours = days_per_month * harvester_daily_hours
st.sidebar.write(f"⏱ Horas Mensuales: **{harvester_monthly_hours}**")

# Forwarder Shift
st.sidebar.subheader("Forwarder")
forwarder_daily_hours = st.sidebar.number_input("Horas/Día (Forwarder)", value=9.0, step=0.5)
forwarder_monthly_hours = days_per_month * forwarder_daily_hours
st.sidebar.write(f"⏱ Horas Mensuales: **{forwarder_monthly_hours}**")

# --- DATA INPUTS ---

def get_machine_inputs(prefix, hours_month):
    with st.expander(f"⚙️ Costos Operacionales: {prefix}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rent = st.number_input(f"Arriendo Mensual {prefix} ($)", value=10900000 if prefix=="Harvester" else 8000000, step=100000)
            salary = st.number_input(f"Sueldo Operadores (Total Mes) {prefix} ($)", value=3847442 if prefix=="Harvester" else 1923721, step=50000, help="Suma de todos los operadores del equipo")
        with col2:
            fuel_consump = st.number_input(f"Consumo Petróleo (L/hr) {prefix}", value=20.0 if prefix=="Harvester" else 15.0, step=1.0)
            maint_hourly = st.number_input(f"Costo Mantención Promedio ($/hr) {prefix}", value=5500 if prefix=="Harvester" else 3500, step=100, help="Promedio ponderado de mantenciones 600h, 1200h, reparaciones, etc.")
        
        consumables = st.number_input(f"Consumibles Mensuales (Cadenas, Espadas, Aceite) {prefix} ($)", value=410000 if prefix=="Harvester" else 200000, step=10000)
        
        return {
            "rent": rent,
            "salary": salary,
            "fuel_l_hr": fuel_consump,
            "maintenance_hr": maint_hourly,
            "consumables_month": consumables,
            "hours_month": hours_month
        }

# Shared Costs
def get_shared_inputs(h_hours, f_hours):
    with st.expander("🏢 Costos Fijos Compartidos (Faena)", expanded=False):
        st.info("Estos costos se distribuyen entre las máquinas.")
        col1, col2 = st.columns(2)
        with col1:
            pickup_rent = st.number_input("Arriendo Camionetas (Total Mes) ($)", value=1504816, step=10000)
            pickup_fuel = st.number_input("Combustible Camionetas (Total Mes) ($)", value=535104, step=10000)
            support_staff = st.number_input("Personal Apoyo (Mecánico, Prevencionista, Jefe) ($)", value=2164000, step=50000)
        with col2:
            facilities = st.number_input("Instalación de Faena / Gastos Adm ($)", value=560000, step=10000)
            pension = st.number_input("Pensión / Alojamiento ($)", value=1890000, step=50000)
            others = st.number_input("Otros Costos Fijos ($)", value=100000, step=10000)
        
        total_shared = pickup_rent + pickup_fuel + support_staff + facilities + pension + others
        
        # Allocation Logic
        alloc_method = st.radio("Método de Asignación", ["Porcentaje Fijo", "Proporcional a Horas"], horizontal=True)
        if alloc_method == "Porcentaje Fijo":
            h_share_pct = st.slider("% Asignado a Harvester", 0, 100, 66) / 100.0
            f_share_pct = 1.0 - h_share_pct
        else:
            total_h = h_hours + f_hours
            h_share_pct = h_hours / total_h
            f_share_pct = f_hours / total_h
            
        return total_shared, h_share_pct, f_share_pct

# Get Inputs
harvester_data = get_machine_inputs("Harvester", harvester_monthly_hours)
forwarder_data = get_machine_inputs("Forwarder", forwarder_monthly_hours)
shared_total, h_share, f_share = get_shared_inputs(harvester_monthly_hours, forwarder_monthly_hours)

# --- CALCULATIONS ---

def calculate_hourly_rate(data, fuel_price, shared_cost_allocation):
    # Fixed Costs per Hour
    rent_hr = data['rent'] / data['hours_month']
    salary_hr = data['salary'] / data['hours_month']
    consumables_hr = data['consumables_month'] / data['hours_month']
    
    # Variable Costs per Hour
    fuel_hr = data['fuel_l_hr'] * fuel_price
    maint_hr = data['maintenance_hr']
    
    # Shared Overhead per Hour
    shared_hr = shared_cost_allocation / data['hours_month']
    
    total_hr = rent_hr + salary_hr + consumables_hr + fuel_hr + maint_hr + shared_hr
    
    return {
        "Arriendo": rent_hr,
        "Operadores": salary_hr,
        "Combustible": fuel_hr,
        "Mantención": maint_hr,
        "Consumibles": consumables_hr,
        "Costos Fijos Asig.": shared_hr,
        "Total Hora": total_hr
    }

h_costs = calculate_hourly_rate(harvester_data, fuel_price, shared_total * h_share)
f_costs = calculate_hourly_rate(forwarder_data, fuel_price, shared_total * f_share)

# --- DISPLAY RESULTS ---

st.divider()

# 1. Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Costo Hora Harvester", f"${h_costs['Total Hora']:,.0f}")
col2.metric("Costo Hora Forwarder", f"${f_costs['Total Hora']:,.0f}")
col3.metric("Costo Fijo Mensual Faena", f"${shared_total:,.0f}")

# 2. Detailed Breakdown Table
st.subheader("📊 Desglose de Costos Horarios")
breakdown_df = pd.DataFrame([h_costs, f_costs], index=["Harvester", "Forwarder"]).T
breakdown_df = breakdown_df.style.format("${:,.0f}")
st.dataframe(breakdown_df, use_container_width=True)

# 3. Sensitivity Analysis (Cost per m3)
st.subheader("📈 Análisis de Sensibilidad (Costo Unitario vs Productividad)")

prod_range = np.arange(10, 45, 1) # Productivity from 10 to 44 m3/hr

def generate_sensitivity(hourly_cost, prod_range):
    costs = [hourly_cost / p for p in prod_range]
    return costs

h_sens = generate_sensitivity(h_costs['Total Hora'], prod_range)
f_sens = generate_sensitivity(f_costs['Total Hora'], prod_range)

sens_df = pd.DataFrame({
    "Productividad (m3/hr)": prod_range,
    "Costo Harvester ($/m3)": h_sens,
    "Costo Forwarder ($/m3)": f_sens,
    "Costo Sistema ($/m3)": [h+f for h,f in zip(h_sens, f_sens)]
})

# Display Charts
tab1, tab2 = st.tabs(["Gráfico de Costos", "Tabla de Datos"])

with tab1:
    fig = px.line(sens_df, x="Productividad (m3/hr)", y=["Costo Harvester ($/m3)", "Costo Forwarder ($/m3)", "Costo Sistema ($/m3)"],
                  title="Costo Unitario ($/m3) según Productividad",
                  labels={"value": "Costo ($/m3)", "variable": "Máquina"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(sens_df.style.format("{:.1f}"), use_container_width=True)

# Download Button
csv = sens_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 Descargar Tabla de Costos (CSV)",
    data=csv,
    file_name='costos_forestales_faena.csv',
    mime='text/csv',
)
