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
)
