# app_actualizada_horas_dias.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Calculadora de Costos Forestales", layout="wide")

st.title("🌲 Calculadora de Costos Harvester y Forwarder")
st.markdown(
    """
Esta aplicación estima costos horarios, mensuales y unitarios ($/m³) para operaciones forestales.
Ahora el cálculo considera por separado los **días trabajados** y las **horas por día** de cada máquina.
"""
)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# --- SIDEBAR: Configuration Parameters ---
st.sidebar.header("1. Parámetros Económicos")
uf_value = st.sidebar.number_input("Valor UF ($)", min_value=1.0, value=39704.93, step=100.0)
fuel_price = st.sidebar.number_input("Precio Petróleo ($/L)", min_value=0, value=774, step=10)

st.sidebar.header("2. Configuración de Jornada por Máquina")

st.sidebar.subheader("Harvester")
harvester_days_month = st.sidebar.number_input("Días/Mes (Harvester)", min_value=1, value=30, step=1)
harvester_daily_hours = st.sidebar.number_input("Horas/Día (Harvester)", min_value=0.5, value=10.0, step=0.5)
harvester_monthly_hours = harvester_days_month * harvester_daily_hours
st.sidebar.write(f"⏱ Horas Mensuales Harvester: **{harvester_monthly_hours:,.1f}**")

st.sidebar.subheader("Forwarder")
forwarder_days_month = st.sidebar.number_input("Días/Mes (Forwarder)", min_value=1, value=30, step=1)
forwarder_daily_hours = st.sidebar.number_input("Horas/Día (Forwarder)", min_value=0.5, value=9.0, step=0.5)
forwarder_monthly_hours = forwarder_days_month * forwarder_daily_hours
st.sidebar.write(f"⏱ Horas Mensuales Forwarder: **{forwarder_monthly_hours:,.1f}**")

st.sidebar.header("3. Productividad de Referencia")
harvester_prod = st.sidebar.number_input("Productividad Harvester (m³/hr)", min_value=0.1, value=28.0, step=0.5)
forwarder_prod = st.sidebar.number_input("Productividad Forwarder (m³/hr)", min_value=0.1, value=24.0, step=0.5)


# --- DATA INPUTS ---
def get_machine_inputs(prefix, days_month, hours_day, hours_month):
    with st.expander(f"⚙️ Costos Operacionales: {prefix}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rent = st.number_input(
                f"Arriendo Mensual {prefix} ($)",
                min_value=0,
                value=10900000 if prefix == "Harvester" else 8000000,
                step=100000,
            )
            salary = st.number_input(
                f"Sueldo Operadores (Total Mes) {prefix} ($)",
                min_value=0,
                value=3847442 if prefix == "Harvester" else 1923721,
                step=50000,
                help="Suma de todos los operadores del equipo",
            )
        with col2:
            fuel_consump = st.number_input(
                f"Consumo Petróleo (L/hr) {prefix}",
                min_value=0.0,
                value=20.0 if prefix == "Harvester" else 15.0,
                step=1.0,
            )
            maint_hourly = st.number_input(
                f"Costo Mantención Promedio ($/hr) {prefix}",
                min_value=0,
                value=5500 if prefix == "Harvester" else 3500,
                step=100,
                help="Promedio ponderado de mantenciones 600h, 1200h, reparaciones, etc.",
            )

        consumables = st.number_input(
            f"Consumibles Mensuales (Cadenas, Espadas, Aceite) {prefix} ($)",
            min_value=0,
            value=410000 if prefix == "Harvester" else 200000,
            step=10000,
        )

        return {
            "days_month": days_month,
            "hours_day": hours_day,
            "hours_month": hours_month,
            "rent": rent,
            "salary": salary,
            "fuel_l_hr": fuel_consump,
            "maintenance_hr": maint_hourly,
            "consumables_month": consumables,
        }


# Shared Costs
def get_shared_inputs(h_hours, f_hours, h_days, f_days):
    with st.expander("🏢 Costos Fijos Compartidos (Faena)", expanded=False):
        st.info("Estos costos se distribuyen entre las máquinas.")
        col1, col2 = st.columns(2)
        with col1:
            pickup_rent = st.number_input("Arriendo Camionetas (Total Mes) ($)", min_value=0, value=1504816, step=10000)
            pickup_fuel = st.number_input("Combustible Camionetas (Total Mes) ($)", min_value=0, value=535104, step=10000)
            support_staff = st.number_input("Personal Apoyo (Mecánico, Prevencionista, Jefe) ($)", min_value=0, value=2164000, step=50000)
        with col2:
            facilities = st.number_input("Instalación de Faena / Gastos Adm ($)", min_value=0, value=560000, step=10000)
            pension = st.number_input("Pensión / Alojamiento ($)", min_value=0, value=1890000, step=50000)
            others = st.number_input("Otros Costos Fijos ($)", min_value=0, value=100000, step=10000)

        total_shared = pickup_rent + pickup_fuel + support_staff + facilities + pension + others

        alloc_method = st.radio(
            "Método de Asignación",
            ["Porcentaje Fijo", "Proporcional a Horas", "Proporcional a Días"],
            horizontal=True,
        )
        if alloc_method == "Porcentaje Fijo":
            h_share_pct = st.slider("% Asignado a Harvester", 0, 100, 66) / 100.0
            f_share_pct = 1.0 - h_share_pct
        elif alloc_method == "Proporcional a Días":
            total_days = h_days + f_days
            h_share_pct = safe_div(h_days, total_days)
            f_share_pct = safe_div(f_days, total_days)
        else:
            total_hours = h_hours + f_hours
            h_share_pct = safe_div(h_hours, total_hours)
            f_share_pct = safe_div(f_hours, total_hours)

        return total_shared, h_share_pct, f_share_pct, alloc_method


# Get Inputs
harvester_data = get_machine_inputs(
    "Harvester", harvester_days_month, harvester_daily_hours, harvester_monthly_hours
)
forwarder_data = get_machine_inputs(
    "Forwarder", forwarder_days_month, forwarder_daily_hours, forwarder_monthly_hours
)
shared_total, h_share, f_share, alloc_method = get_shared_inputs(
    harvester_monthly_hours,
    forwarder_monthly_hours,
    harvester_days_month,
    forwarder_days_month,
)


# --- CALCULATIONS ---
def calculate_machine_costs(data, fuel_price, shared_cost_allocation, productivity_m3_hr, uf_value):
    hours_month = data["hours_month"]

    # Fixed monthly costs
    rent_month = data["rent"]
    salary_month = data["salary"]
    consumables_month = data["consumables_month"]
    shared_month = shared_cost_allocation

    # Variable costs
    fuel_hr = data["fuel_l_hr"] * fuel_price
    maint_hr = data["maintenance_hr"]
    variable_hr = fuel_hr + maint_hr
    variable_month = variable_hr * hours_month

    total_month = rent_month + salary_month + consumables_month + shared_month + variable_month

    # Hourly breakdown based on the specific days/hours worked by the machine
    rent_hr = safe_div(rent_month, hours_month)
    salary_hr = safe_div(salary_month, hours_month)
    consumables_hr = safe_div(consumables_month, hours_month)
    shared_hr = safe_div(shared_month, hours_month)
    total_hr = safe_div(total_month, hours_month)

    production_month = productivity_m3_hr * hours_month
    cost_m3 = safe_div(total_month, production_month)

    return {
        "Días/Mes": data["days_month"],
        "Horas/Día": data["hours_day"],
        "Horas/Mes": hours_month,
        "Productividad (m3/hr)": productivity_m3_hr,
        "Producción Mes (m3)": production_month,
        "Arriendo": rent_hr,
        "Operadores": salary_hr,
        "Combustible": fuel_hr,
        "Mantención": maint_hr,
        "Consumibles": consumables_hr,
        "Costos Fijos Asig.": shared_hr,
        "Total Hora": total_hr,
        "Costo Total Mes": total_month,
        "Costo Unitario ($/m3)": cost_m3,
        "Costo Hora (UF/hr)": safe_div(total_hr, uf_value),
    }


h_costs = calculate_machine_costs(harvester_data, fuel_price, shared_total * h_share, harvester_prod, uf_value)
f_costs = calculate_machine_costs(forwarder_data, fuel_price, shared_total * f_share, forwarder_prod, uf_value)

system_monthly_production = min(h_costs["Producción Mes (m3)"], f_costs["Producción Mes (m3)"])
system_monthly_cost = h_costs["Costo Total Mes"] + f_costs["Costo Total Mes"]
system_unit_cost = safe_div(system_monthly_cost, system_monthly_production)

# --- DISPLAY RESULTS ---
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Horas/Mes Harvester", f"{h_costs['Horas/Mes']:,.1f}")
col2.metric("Horas/Mes Forwarder", f"{f_costs['Horas/Mes']:,.1f}")
col3.metric("Costo Hora Harvester", f"${h_costs['Total Hora']:,.0f}")
col4.metric("Costo Hora Forwarder", f"${f_costs['Total Hora']:,.0f}")

col5, col6, col7 = st.columns(3)
col5.metric("Costo Fijo Mensual Faena", f"${shared_total:,.0f}")
col6.metric("Producción Sistema Mes", f"{system_monthly_production:,.1f} m³")
col7.metric("Costo Sistema", f"${system_unit_cost:,.0f}/m³")

st.caption(
    f"Asignación de costos compartidos: {alloc_method} | Harvester {h_share:.1%} | Forwarder {f_share:.1%}"
)

# Detailed summary
st.subheader("📋 Resumen Operacional y Económico")
summary_df = pd.DataFrame(
    [h_costs, f_costs],
    index=["Harvester", "Forwarder"],
)[
    [
        "Días/Mes",
        "Horas/Día",
        "Horas/Mes",
        "Productividad (m3/hr)",
        "Producción Mes (m3)",
        "Costo Total Mes",
        "Total Hora",
        "Costo Unitario ($/m3)",
        "Costo Hora (UF/hr)",
    ]
]
st.dataframe(
    summary_df.style.format(
        {
            "Días/Mes": "{:.0f}",
            "Horas/Día": "{:.1f}",
            "Horas/Mes": "{:.1f}",
            "Productividad (m3/hr)": "{:.1f}",
            "Producción Mes (m3)": "{:.1f}",
            "Costo Total Mes": "$ {:,.0f}",
            "Total Hora": "$ {:,.0f}",
            "Costo Unitario ($/m3)": "$ {:,.1f}",
            "Costo Hora (UF/hr)": "{:.3f}",
        }
    ),
    use_container_width=True,
)

# Hourly cost breakdown
st.subheader("📊 Desglose de Costos Horarios")
breakdown_df = pd.DataFrame([h_costs, f_costs], index=["Harvester", "Forwarder"]).T.loc[
    ["Arriendo", "Operadores", "Combustible", "Mantención", "Consumibles", "Costos Fijos Asig.", "Total Hora"]
]
st.dataframe(breakdown_df.style.format("$ {:,.0f}"), use_container_width=True)

# Sensitivity Analysis
st.subheader("📈 Análisis de Sensibilidad")
st.caption(
    "En la curva se usa una productividad común de referencia para comparar el costo unitario del sistema con las horas mensuales reales de cada máquina."
)
prod_range = np.arange(10, 45, 1)

sens_rows = []
for prod in prod_range:
    h_month_prod = h_costs["Horas/Mes"] * prod
    f_month_prod = f_costs["Horas/Mes"] * prod
    system_prod = min(h_month_prod, f_month_prod)
    sens_rows.append(
        {
            "Productividad Referencia (m3/hr)": prod,
            "Costo Harvester ($/m3)": safe_div(h_costs["Costo Total Mes"], h_month_prod),
            "Costo Forwarder ($/m3)": safe_div(f_costs["Costo Total Mes"], f_month_prod),
            "Costo Sistema ($/m3)": safe_div(system_monthly_cost, system_prod),
        }
    )

sens_df = pd.DataFrame(sens_rows)

tab1, tab2 = st.tabs(["Gráfico de Costos", "Tabla de Datos"])

with tab1:
    fig = px.line(
        sens_df,
        x="Productividad Referencia (m3/hr)",
        y=["Costo Harvester ($/m3)", "Costo Forwarder ($/m3)", "Costo Sistema ($/m3)"],
        title="Costo Unitario ($/m3) según Productividad de Referencia",
        labels={"value": "Costo ($/m3)", "variable": "Concepto"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(
        sens_df.style.format(
            {
                "Productividad Referencia (m3/hr)": "{:.0f}",
                "Costo Harvester ($/m3)": "$ {:,.1f}",
                "Costo Forwarder ($/m3)": "$ {:,.1f}",
                "Costo Sistema ($/m3)": "$ {:,.1f}",
            }
        ),
        use_container_width=True,
    )

# Download Button
csv = sens_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Descargar Tabla de Costos (CSV)",
    data=csv,
    file_name="costos_forestales_faena.csv",
    mime="text/csv",
)
