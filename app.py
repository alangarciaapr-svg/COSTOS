import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(
    page_title="ForestCost Excel View",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo para tablas tipo Excel
st.markdown("""
<style>
    .excel-table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .excel-table th {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        padding: 8px;
        text-align: right;
        font-weight: bold;
        color: #333;
    }
    .excel-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: right;
    }
    .excel-table td:first-child, .excel-table th:first-child {
        text-align: left;
    }
    .excel-total-row {
        background-color: #e8f5e9;
        font-weight: bold;
    }
    .metric-card {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_excel_v4.json'

# --- 2. PERSISTENCIA ---
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

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    # Harvester
    "rent_total_Harvester", "fuel_l_hr_Harvester", "salary_total_Harvester", 
    "maint_prev_Harvester", "maint_corr_Harvester", "maint_tires_Harvester",
    "consum_cut_Harvester", "consum_hyd_Harvester", "consum_lub_Harvester",
    "others_total_Harvester",
    # Forwarder
    "rent_total_Forwarder", "fuel_l_hr_Forwarder", "salary_total_Forwarder", 
    "maint_prev_Forwarder", "maint_corr_Forwarder", "maint_tires_Forwarder",
    "consum_cut_Forwarder", "consum_hyd_Forwarder", "consum_lub_Forwarder",
    "others_total_Forwarder",
    # Shared
    "pickup_rent_uf", "pickup_liters_day", "pickup_days_month",
    "support_staff", "facilities", "pension", "others_shared", "alloc_method", "h_share_pct_manual"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            st.session_state[key] = saved_config[key]
    st.session_state['config_loaded'] = True

# --- 3. UTILIDADES ---
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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Configuración")
    
    st.markdown("### 1. Economía")
    uf_api, fecha_api = get_uf_value()
    use_auto_uf = st.checkbox("UF Automática", value=True, key="use_auto_uf", on_change=save_config)
    if use_auto_uf and uf_api:
        current_uf = uf_api
        st.success(f"UF: ${fmt(current_uf)}")
    else:
        current_uf = st.number_input("UF Manual", value=39704.93, step=100.0, key="uf_manual", on_change=save_config)

    fuel_price = st.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

    st.markdown("### 2. Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Precio Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)
    
    st.markdown("### 3. Asignación Ingresos")
    h_rev_pct = st.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)

# --- 5. LOGICA PRINCIPAL ---

st.title("🌲 Planilla de Costos Tipo Excel")

# --- BLOQUE 1: JORNADA ---
with st.expander("📅 Configuración de Jornada (Días y Horas)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", 28, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", 10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", 25, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", 9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day

# --- BLOQUE 2: INPUTS Y TABLA ---
st.markdown("---")

col_h, col_f = st.columns(2)

def machine_block(prefix, hours_month, days_month, hours_day, fuel_p, col_obj):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        
        # --- INPUTS (Acordeón para editar, cerrado por defecto para ver la tabla limpia) ---
        with st.expander("✏️ Editar Costos Unitarios y Totales", expanded=False):
            st.markdown("**(1) Fijos y Operación**")
            def_rent = 10900000 if prefix=="Harvester" else 8000000
            def_sal = 3800000 if prefix=="Harvester" else 1900000
            def_l_hr = 20.0 if prefix=="Harvester" else 15.0
            
            rent = st.number_input(f"Arriendo Mensual ($)", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
            sal = st.number_input(f"Sueldos Totales ($)", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)
            l_hr = st.number_input(f"Consumo L/Hora", value=def_l_hr, step=1.0, key=f"fuel_l_hr_{prefix}", on_change=save_config)
            
            st.markdown("**(2) Mantención (Desglose)**")
            m_prev = st.number_input("Preventiva ($)", value=800000 if prefix=="H" else 500000, step=50000, key=f"maint_prev_{prefix}", on_change=save_config)
            m_corr = st.number_input("Correctiva ($)", value=500000 if prefix=="H" else 300000, step=50000, key=f"maint_corr_{prefix}", on_change=save_config)
            m_tires = st.number_input("Neumát/Orugas ($)", value=200000, step=50000, key=f"maint_tires_{prefix}", on_change=save_config)
            
            st.markdown("**(3) Consumibles (Desglose)**")
            c_cut = st.number_input("Corte (Esp/Cad) ($)", value=200000, step=20000, key=f"consum_cut_{prefix}", on_change=save_config)
            c_hyd = st.number_input("Aceite Hid. ($)", value=150000, step=20000, key=f"consum_hyd_{prefix}", on_change=save_config)
            c_lub = st.number_input("Lubric/Filtros ($)", value=60000, step=10000, key=f"consum_lub_{prefix}", on_change=save_config)
            
            others = st.number_input(f"Otros ($)", value=0, step=10000, key=f"others_total_{prefix}", on_change=save_config)

        # --- CÁLCULOS ---
        fuel_month = l_hr * hours_month * fuel_p
        maint_total = m_prev + m_corr + m_tires
        consum_total = c_cut + c_hyd + c_lub
        total_direct = rent + sal + fuel_month + maint_total + consum_total + others
        
        # --- CONSTRUCCIÓN DE LA TABLA TIPO EXCEL ---
        # Lista de filas: [Concepto, Detalle, Mensual, Hora]
        rows = [
            ["Arriendo Maquinaria", "Fijo Mes", rent, rent/hours_month if hours_month else 0],
            ["Operadores (Sueldos)", "Fijo Mes", sal, sal/hours_month if hours_month else 0],
            ["Combustible Diesel", f"{l_hr} L/hr * ${fuel_p}", fuel_month, l_hr * fuel_p],
            ["Mantención Preventiva", "Variable/Fijo", m_prev, m_prev/hours_month if hours_month else 0],
            ["Mantención Correctiva", "Variable", m_corr, m_corr/hours_month if hours_month else 0],
            ["Neumáticos / Orugas", "Variable", m_tires, m_tires/hours_month if hours_month else 0],
            ["El. Corte (Cadenas/Espadas)", "Consumible", c_cut, c_cut/hours_month if hours_month else 0],
            ["Aceites Hidráulicos", "Consumible", c_hyd, c_hyd/hours_month if hours_month else 0],
            ["Lubricantes y Filtros", "Consumible", c_lub, c_lub/hours_month if hours_month else 0],
            ["Otros Costos", "Varios", others, others/hours_month if hours_month else 0],
        ]
        
        # Renderizar HTML Table
        html = f"""
        <table class="excel-table">
            <thead>
                <tr>
                    <th>Ítem de Costo</th>
                    <th>Detalle / Base</th>
                    <th>Total Mensual</th>
                    <th>Valor Hora</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for r in rows:
            html += f"""
            <tr>
                <td>{r[0]}</td>
                <td>{r[1]}</td>
                <td>${fmt(r[2])}</td>
                <td>${fmt(r[3])}</td>
            </tr>
            """
        
        # Fila Total
        tot_hr = total_direct / hours_month if hours_month else 0
        html += f"""
            <tr class="excel-total-row">
                <td>TOTAL COSTO DIRECTO</td>
                <td>{fmt(hours_month)} Hrs/Mes</td>
                <td>${fmt(total_direct)}</td>
                <td>${fmt(tot_hr)}</td>
            </tr>
            </tbody>
        </table>
        """
        
        st.markdown(html, unsafe_allow_html=True)
        return total_direct, tot_hr

# Ejecutar bloques
h_total_month, h_total_hr = machine_block("Harvester", h_total_hours, h_days, h_hours_day, fuel_price, col_h)
f_total_month, f_total_hr = machine_block("Forwarder", f_total_hours, f_days, f_hours_day, fuel_price, col_f)

# --- BLOQUE 3: INDIRECTOS Y FAENA (TIPO EXCEL) ---
st.markdown("---")
st.subheader("🏢 Costos Indirectos y Faena")

with st.expander("✏️ Editar Costos Indirectos", expanded=False):
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        pickup_l_day = st.number_input("Litros/Día Camionetas", 12.0, step=1.0, key="pickup_liters_day", on_change=save_config)
        pickup_days = st.number_input("Días Uso Camionetas", 30, key="pickup_days_month", on_change=save_config)
        p_rent_uf = st.number_input("Arriendo Camionetas (UF)", 38.0, step=0.5, key="pickup_rent_uf", on_change=save_config)
    with col_ind2:
        staff = st.number_input("Personal Apoyo ($)", 2164000, step=100000, key="support_staff", on_change=save_config)
        fac = st.number_input("Instalaciones ($)", 560000, step=50000, key="facilities", on_change=save_config)
        pen = st.number_input("Pensión/Alojamiento ($)", 1890000, step=50000, key="pension", on_change=save_config)
        oth = st.number_input("Otros Faena ($)", 100000, step=10000, key="others_shared", on_change=save_config)
    
    # Asignación
    st.caption("Asignación")
    alloc_opts = ["Porcentaje Manual", "Proporcional Horas"]
    idx = alloc_opts.index(st.session_state.get("alloc_method", "Porcentaje Manual")) if st.session_state.get("alloc_method") in alloc_opts else 0
    alloc = st.radio("", alloc_opts, index=idx, key="alloc_method", horizontal=True, on_change=save_config)
    if alloc == "Porcentaje Manual":
        h_pct = st.slider("% Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
    else:
        tot_h = h_total_hours + f_total_hours
        h_pct = h_total_hours / tot_h if tot_h > 0 else 0.5

# Cálculos Indirectos
p_fuel_month = pickup_l_day * pickup_days * fuel_price
p_rent_clp = p_rent_uf * current_uf
total_shared = p_rent_clp + p_fuel_month + staff + fac + pen + oth

# Asignación
shared_h = total_shared * h_pct
shared_f = total_shared * (1 - h_pct)

# Tabla Resumen Faena
st.markdown(f"""
<table class="excel-table">
    <thead>
        <tr>
            <th>Ítem Indirecto</th>
            <th>Detalle</th>
            <th>Total Mensual</th>
            <th>Asig. Harvester ({(h_pct*100):.0f}%)</th>
            <th>Asig. Forwarder ({(100-h_pct*100):.0f}%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Arriendo Camionetas</td>
            <td>{p_rent_uf} UF * ${fmt(current_uf)}</td>
            <td>${fmt(p_rent_clp)}</td>
            <td>${fmt(p_rent_clp*h_pct)}</td>
            <td>${fmt(p_rent_clp*(1-h_pct))}</td>
        </tr>
        <tr>
            <td>Diesel Camionetas</td>
            <td>{pickup_l_day} L/día * {pickup_days} días</td>
            <td>${fmt(p_fuel_month)}</td>
            <td>${fmt(p_fuel_month*h_pct)}</td>
            <td>${fmt(p_fuel_month*(1-h_pct))}</td>
        </tr>
        <tr>
            <td>Gastos Generales (Personal, Pensión, etc)</td>
            <td>Suma fijos</td>
            <td>${fmt(staff+fac+pen+oth)}</td>
            <td>${fmt((staff+fac+pen+oth)*h_pct)}</td>
            <td>${fmt((staff+fac+pen+oth)*(1-h_pct))}</td>
        </tr>
        <tr class="excel-total-row">
            <td>TOTAL INDIRECTOS</td>
            <td>---</td>
            <td>${fmt(total_shared)}</td>
            <td>${fmt(shared_h)}</td>
            <td>${fmt(shared_f)}</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

# --- RESUMEN FINAL ---
st.markdown("### 📊 Resultado Final Sistema")

final_h_hr = (h_total_month + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_month + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

c_res1, c_res2, c_res3 = st.columns(3)
with c_res1:
    st.info(f"Harvester Final: **${fmt(final_h_hr)} /hr**")
with c_res2:
    st.info(f"Forwarder Final: **${fmt(final_f_hr)} /hr**")
with c_res3:
    st.success(f"Sistema Total: **${fmt(sys_hr)} /hr**")

# Sensibilidad Rápida
st.caption("Rentabilidad Rápida")
m3_val = st.slider("Productividad (M3/Hr)", 10, 50, 25)
mr_val = m3_val / conversion_factor
cost_mr = sys_hr / mr_val if mr_val else 0
margen = sales_price_mr - cost_mr
margen_pct = (margen / sales_price_mr * 100) if sales_price_mr else 0

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Producción MR/Hr", f"{mr_val:.1f}")
col_kpi2.metric("Costo Unitario", f"${fmt(cost_mr)} /MR")
col_kpi3.metric("Margen", f"{margen_pct:.1f}%", f"${fmt(margen)}/MR")
