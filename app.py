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
    page_title="ForestCost Analytics",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS (Solo para tarjetas métricas, sin tablas complejas)
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_clean.json'

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

def card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
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
    h_rev_pct = st.slider("% Ingresos Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    # Calculamos ingresos
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * ((100 - h_rev_pct) / 100)
    st.info(f"H: ${fmt(h_income)} | F: ${fmt(f_income)}")

# --- 5. LOGICA PRINCIPAL ---

st.title("🌲 Planilla de Costos Forestales")

# --- BLOQUE 1: JORNADA ---
with st.expander("📅 Configuración de Jornada (Días y Horas)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", 28, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", 10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
        st.caption(f"Total: {fmt(h_total_hours)} Hrs/Mes")
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", 25, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", 9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day
        st.caption(f"Total: {fmt(f_total_hours)} Hrs/Mes")

st.divider()

# --- BLOQUE 2: MAQUINARIA (TABLAS NATIVAS) ---

col_h, col_f = st.columns(2)

def machine_block(prefix, hours_month, days_month, hours_day, fuel_p, col_obj):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        
        # --- INPUTS ---
        with st.expander("✏️ Editar Costos", expanded=False):
            st.markdown("**(1) Fijos y Operación**")
            def_rent = 10900000 if prefix=="Harvester" else 8000000
            def_sal = 3800000 if prefix=="Harvester" else 1900000
            def_l_hr = 20.0 if prefix=="Harvester" else 15.0
            
            rent = st.number_input(f"Arriendo Mensual ($)", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
            sal = st.number_input(f"Sueldos Totales ($)", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)
            l_hr = st.number_input(f"Consumo L/Hora", value=def_l_hr, step=1.0, key=f"fuel_l_hr_{prefix}", on_change=save_config)
            
            st.markdown("**(2) Mantención (Desglose)**")
            m_prev = st.number_input("Preventiva ($)", value=800000 if prefix=="Harvester" else 500000, step=50000, key=f"maint_prev_{prefix}", on_change=save_config)
            m_corr = st.number_input("Correctiva ($)", value=500000 if prefix=="Harvester" else 300000, step=50000, key=f"maint_corr_{prefix}", on_change=save_config)
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
        
        # --- TABLA DE DATOS (DataFrame) ---
        # Creamos los datos estructurados
        data_rows = [
            {"Ítem": "Arriendo Maquinaria", "Detalle": "Fijo Mes", "Mensual": rent},
            {"Ítem": "Operadores (Sueldos)", "Detalle": "Fijo Mes", "Mensual": sal},
            {"Ítem": "Combustible Diesel", "Detalle": f"{l_hr} L/hr", "Mensual": fuel_month},
            {"Ítem": "Mant. Preventiva", "Detalle": "Var", "Mensual": m_prev},
            {"Ítem": "Mant. Correctiva", "Detalle": "Var", "Mensual": m_corr},
            {"Ítem": "Neumáticos/Orugas", "Detalle": "Var", "Mensual": m_tires},
            {"Ítem": "El. Corte", "Detalle": "Consum", "Mensual": c_cut},
            {"Ítem": "Aceite Hidráulico", "Detalle": "Consum", "Mensual": c_hyd},
            {"Ítem": "Lubricantes", "Detalle": "Consum", "Mensual": c_lub},
            {"Ítem": "Otros", "Detalle": "Var", "Mensual": others},
        ]
        
        df = pd.DataFrame(data_rows)
        # Calculamos columna Hora
        df["Hora"] = df["Mensual"] / hours_month if hours_month else 0
        
        # Fila de TOTALES
        total_hr = total_direct / hours_month if hours_month else 0
        
        # Mostrar Tabla Nativa (Sin HTML)
        st.dataframe(
            df,
            column_config={
                "Ítem": st.column_config.TextColumn("Concepto Costo", width="medium"),
                "Detalle": st.column_config.TextColumn("Base", width="small"),
                "Mensual": st.column_config.NumberColumn("Total Mensual", format="$%d"),
                "Hora": st.column_config.NumberColumn("Valor Hora", format="$%d"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Tarjeta de resumen
        st.info(f"**TOTAL {prefix}: ${fmt(total_direct)}** (Mensual)  |  **${fmt(total_hr)}** (Hora)")
        
        return total_direct, total_hr

# Ejecutar bloques
h_total_month, h_total_hr = machine_block("Harvester", h_total_hours, h_days, h_hours_day, fuel_price, col_h)
f_total_month, f_total_hr = machine_block("Forwarder", f_total_hours, f_days, f_hours_day, fuel_price, col_f)

st.divider()

# --- BLOQUE 3: INDIRECTOS ---
st.subheader("🏢 Costos Indirectos y Faena")

with st.expander("✏️ Editar Indirectos", expanded=False):
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        st.markdown("**Camionetas**")
        pickup_l_day = st.number_input("Litros/Día Camionetas", 12.0, step=1.0, key="pickup_liters_day", on_change=save_config)
        pickup_days = st.number_input("Días Uso Camionetas", 30, key="pickup_days_month", on_change=save_config)
        p_rent_uf = st.number_input("Arriendo Camionetas (UF)", 38.0, step=0.5, key="pickup_rent_uf", on_change=save_config)
    with col_ind2:
        st.markdown("**Generales**")
        staff = st.number_input("Personal Apoyo ($)", 2164000, step=100000, key="support_staff", on_change=save_config)
        fac = st.number_input("Instalaciones ($)", 560000, step=50000, key="facilities", on_change=save_config)
        pen = st.number_input("Pensión/Alojamiento ($)", 1890000, step=50000, key="pension", on_change=save_config)
        oth = st.number_input("Otros Faena ($)", 100000, step=10000, key="others_shared", on_change=save_config)
    
    st.markdown("---")
    st.caption("Asignación de Indirectos")
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

shared_h = total_shared * h_pct
shared_f = total_shared * (1 - h_pct)

# Tabla Resumen Faena (DataFrame Nativo)
shared_data = [
    {"Ítem": "Arriendo Camionetas", "Detalle": f"{p_rent_uf} UF", "Total": p_rent_clp, "Asig H": p_rent_clp*h_pct, "Asig F": p_rent_clp*(1-h_pct)},
    {"Ítem": "Diesel Camionetas", "Detalle": f"{pickup_l_day} L/dia", "Total": p_fuel_month, "Asig H": p_fuel_month*h_pct, "Asig F": p_fuel_month*(1-h_pct)},
    {"Ítem": "Personal y Alojamiento", "Detalle": "Fijo", "Total": staff+pen, "Asig H": (staff+pen)*h_pct, "Asig F": (staff+pen)*(1-h_pct)},
    {"Ítem": "Instalaciones y Varios", "Detalle": "Fijo", "Total": fac+oth, "Asig H": (fac+oth)*h_pct, "Asig F": (fac+oth)*(1-h_pct)},
]
df_shared = pd.DataFrame(shared_data)

st.dataframe(
    df_shared,
    column_config={
        "Ítem": "Concepto Indirecto",
        "Total": st.column_config.NumberColumn("Total Faena", format="$%d"),
        "Asig H": st.column_config.NumberColumn(f"Harvester ({(h_pct*100):.0f}%)", format="$%d"),
        "Asig F": st.column_config.NumberColumn(f"Forwarder ({(100-h_pct*100):.0f}%)", format="$%d"),
    },
    hide_index=True,
    use_container_width=True
)

st.success(f"**TOTAL INDIRECTOS: ${fmt(total_shared)}**")

# --- RESUMEN FINAL Y KPIs ---
st.divider()
st.subheader("📊 Resultado Final Sistema (Costos + Margen)")

final_h_hr = (h_total_month + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_month + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

c1, c2, c3, c4 = st.columns(4)
with c1: card("Costo Hora Harvester", f"${fmt(final_h_hr)}")
with c2: card("Costo Hora Forwarder", f"${fmt(final_f_hr)}")
with c3: card("Costo Hora Sistema", f"${fmt(sys_hr)}")
with c4: card("Costo Total Mensual", f"${fmt(h_total_month + f_total_month + total_shared)}")

# Simulador
st.markdown("### 📉 Simulador de Margen")
m3_val = st.slider("Productividad Esperada (M3/Hr)", 10, 50, 25)
mr_val = m3_val / conversion_factor if conversion_factor else 0

# Costos unitarios
cost_h_mr = final_h_hr / mr_val if mr_val else 0
cost_f_mr = final_f_hr / mr_val if mr_val else 0
cost_sys_mr = sys_hr / mr_val if mr_val else 0

# Utilidades
util_h = h_income - cost_h_mr
util_f = f_income - cost_f_mr
margen_total = sales_price_mr - cost_sys_mr
margen_pct = (margen_total / sales_price_mr * 100) if sales_price_mr else 0

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Prod MR/Hr", f"{mr_val:.1f}")
col_s2.metric("Costo Unitario Total", f"${fmt(cost_sys_mr)} /MR")
col_s3.metric("Margen Global", f"{margen_pct:.1f}%", f"${fmt(margen_total)}/MR")
