import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Calculadora de Costos Forestales", layout="wide")

# Función para formatear números con punto de miles (Ej: 1.000.000)
def fmt(x):
    """Convierte un número a string con separador de miles '.'"""
    if isinstance(x, (int, float)):
        return f"{x:,.0f}".replace(",", ".")
    return x

st.title("🌲 Calculadora de Costos Harvester y Forwarder")
st.markdown("""
Esta aplicación permite estimar los costos horarios y unitarios ($/m³) para operaciones forestales.
Modifique los parámetros y vea los resultados con formato de miles (ej. 1.000).
""")

# --- BARRA LATERAL: Parámetros ---
st.sidebar.header("1. Parámetros Económicos")
uf_value = st.sidebar.number_input("Valor UF ($)", value=39704.93, step=100.0, format="%.2f")
fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", value=774, step=10)

st.sidebar.header("2. Configuración de Jornada")

# --- SECCIÓN HARVESTER ---
st.sidebar.subheader("🚜 Harvester")
h_days_month = st.sidebar.number_input("Días/Mes (Harvester)", value=28, step=1)
h_hours_day = st.sidebar.number_input("Horas/Día (Harvester)", value=10.0, step=0.5)
h_monthly_hours = h_days_month * h_hours_day
st.sidebar.caption(f"⏱ Horas Mensuales Harvester: **{fmt(h_monthly_hours)}**")

# --- SECCIÓN FORWARDER ---
st.sidebar.subheader("🚜 Forwarder")
f_days_month = st.sidebar.number_input("Días/Mes (Forwarder)", value=25, step=1)
f_hours_day = st.sidebar.number_input("Horas/Día (Forwarder)", value=9.0, step=0.5)
f_monthly_hours = f_days_month * f_hours_day
st.sidebar.caption(f"⏱ Horas Mensuales Forwarder: **{fmt(f_monthly_hours)}**")

# --- ENTRADA DE DATOS DE COSTOS ---

def get_machine_inputs(prefix, hours_month):
    """Genera los inputs para una máquina específica"""
    with st.expander(f"⚙️ Costos Operacionales: {prefix}", expanded=True):
        
        # 1. ARRIENDO Y COMBUSTIBLE
        col1, col2 = st.columns(2)
        with col1:
            rent = st.number_input(f"Arriendo Mensual {prefix} ($)", value=10900000 if prefix=="Harvester" else 8000000, step=100000)
        with col2:
            fuel_consump = st.number_input(f"Consumo Petróleo (L/hr) {prefix}", value=20.0 if prefix=="Harvester" else 15.0, step=1.0)
            
        # 2. OPERADORES
        st.markdown(f"##### 👷‍♂️ Operadores {prefix}")
        col_op1, col_op2, col_op3 = st.columns([1, 1, 1])
        with col_op1:
            num_operators = st.number_input(f"N° Operadores {prefix}", value=2 if prefix=="Harvester" else 1, step=1, min_value=0)
        with col_op2:
            # Clave única para evitar error de duplicados
            salary_per_op = st.number_input(f"Sueldo por Operador {prefix} ($)", value=1923721, step=50000, key=f"salary_{prefix}")
        with col_op3:
            total_salary = num_operators * salary_per_op
            st.metric(label=f"Costo Total Operadores {prefix}", value=f"${fmt(total_salary)}")

        # 3. MANTENCIÓN Y CONSUMIBLES
        st.markdown("##### 🔧 Mantención y Otros")
        col3, col4 = st.columns(2)
        with col3:
            maint_hourly = st.number_input(f"Costo Mantención Promedio ($/hr) {prefix}", value=5500 if prefix=="Harvester" else 3500, step=100)
        with col4:
            consumables = st.number_input(f"Consumibles Mensuales {prefix} ($)", value=410000 if prefix=="Harvester" else 200000, step=10000, help="Cadenas, espadas, aceite, etc.")
        
        return {
            "rent": rent,
            "salary": total_salary,
            "fuel_l_hr": fuel_consump,
            "maintenance_hr": maint_hourly,
            "consumables_month": consumables,
            "hours_month": hours_month
        }

# Costos Compartidos (Faena)
def get_shared_inputs(h_hours, f_hours):
    with st.expander("🏢 Costos Fijos Compartidos (Faena)", expanded=False):
        st.info("Estos costos se pagan mensualmente por la faena completa y se distribuyen entre las máquinas.")
        col1, col2 = st.columns(2)
        with col1:
            pickup_rent = st.number_input("Arriendo Camionetas ($)", value=1504816, step=10000)
            pickup_fuel = st.number_input("Combustible Camionetas ($)", value=535104, step=10000)
            support_staff = st.number_input("Personal Apoyo ($)", value=2164000, step=50000, help="Mecánico, Prevencionista, Jefe, etc.")
        with col2:
            facilities = st.number_input("Instalación/Gastos Adm ($)", value=560000, step=10000)
            pension = st.number_input("Pensión/Alojamiento ($)", value=1890000, step=50000)
            others = st.number_input("Otros Costos Fijos ($)", value=100000, step=10000)
        
        total_shared = pickup_rent + pickup_fuel + support_staff + facilities + pension + others
        
        st.write(f"**Total Costos Fijos Faena:** ${fmt(total_shared)}")

        # Lógica de Asignación
        st.markdown("---")
        st.write(" **Distribución de Costos Fijos:**")
        alloc_method = st.radio("Método de Asignación", ["Porcentaje Manual", "Proporcional a Horas Trabajadas"], horizontal=True)
        
        if alloc_method == "Porcentaje Manual":
            h_share_pct = st.slider("% Asignado a Harvester", 0, 100, 60) / 100.0
            f_share_pct = 1.0 - h_share_pct
        else:
            total_h = h_hours + f_hours
            if total_h > 0:
                h_share_pct = h_hours / total_h
                f_share_pct = f_hours / total_h
            else:
                h_share_pct = 0.5
                f_share_pct = 0.5
            st.info(f"Asignación automática: Harvester {h_share_pct*100:.1f}% | Forwarder {f_share_pct*100:.1f}%")
            
        return total_shared, h_share_pct, f_share_pct

# Obtener Inputs
harvester_data = get_machine_inputs("Harvester", h_monthly_hours)
forwarder_data = get_machine_inputs("Forwarder", f_monthly_hours)
shared_total, h_share, f_share = get_shared_inputs(h_monthly_hours, f_monthly_hours)

# --- CÁLCULOS ---

def calculate_hourly_rate(data, fuel_price, shared_cost_allocation):
    if data['hours_month'] == 0:
        return {k:0 for k in ["Arriendo", "Operadores", "Combustible", "Mantención", "Consumibles", "Costos Fijos Asig.", "Total Hora"]}

    # Costos Fijos llevados a Hora
    rent_hr = data['rent'] / data['hours_month']
    salary_hr = data['salary'] / data['hours_month']
    consumables_hr = data['consumables_month'] / data['hours_month']
    
    # Costos Variables directos
    fuel_hr = data['fuel_l_hr'] * fuel_price
    maint_hr = data['maintenance_hr']
    
    # Costo Fijo Faena asignado por hora
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

# --- VISUALIZACIÓN DE RESULTADOS ---

st.divider()

# 1. Métricas Resumen
col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo Hora Harvester", f"${fmt(h_costs['Total Hora'])}")
col2.metric("Costo Hora Forwarder", f"${fmt(f_costs['Total Hora'])}")
col3.metric("Costo Sistema (H+F)", f"${fmt(h_costs['Total Hora'] + f_costs['Total Hora'])}")
col4.metric("Costo Fijo Mensual Faena", f"${fmt(shared_total)}")

# 2. Tabla Detallada
st.subheader("📊 Desglose de Costos Horarios")
breakdown_df = pd.DataFrame([h_costs, f_costs], index=["Harvester", "Forwarder"]).T
breakdown_df["TOTAL SISTEMA"] = breakdown_df["Harvester"] + breakdown_df["Forwarder"]

# Aplicar formato "1.000" a la tabla
st.dataframe(breakdown_df.style.format(lambda x: f"${fmt(x)}"), use_container_width=True)

# 3. Análisis de Sensibilidad
st.subheader("📈 Análisis de Sensibilidad (Costo Unitario vs Productividad)")
st.caption("Variación del costo por metro cúbico ($/m³) según la productividad.")

prod_range = np.arange(10, 51, 1) # Productividad de 10 a 50 m3/hr

def generate_sensitivity(hourly_cost, prod_range):
    if hourly_cost == 0: return [0]*len(prod_range)
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

# Pestañas para Gráfico y Datos
tab1, tab2 = st.tabs(["Gráfico de Costos", "Tabla de Datos"])

with tab1:
    fig = px.line(sens_df, x="Productividad (m3/hr)", y=["Costo Harvester ($/m3)", "Costo Forwarder ($/m3)", "Costo Sistema ($/m3)"],
                  title="Curva de Costo Unitario ($/m3)",
                  labels={"value": "Costo ($/m3)", "variable": "Ítem"},
                  markers=True)
    
    # Configurar separadores de miles con punto para el gráfico (formato local)
    fig.update_layout(separators=",.") # Define: decimal=, miles=.
    fig.update_yaxes(tickformat=",.0f") # Formato numérico
    fig.update_layout(hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Formatear la tabla de datos también
    st.dataframe(sens_df.style.format({
        "Productividad (m3/hr)": "{:.0f}",
        "Costo Harvester ($/m3)": lambda x: f"${fmt(x)}",
        "Costo Forwarder ($/m3)": lambda x: f"${fmt(x)}",
        "Costo Sistema ($/m3)": lambda x: f"${fmt(x)}"
    }), use_container_width=True)

# Botón de Descarga
csv = sens_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 Descargar Tabla de Sensibilidad (CSV)",
    data=csv,
    file_name='costos_forestales_sensibilidad.csv',
    mime='text/csv',
)
