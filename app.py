import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Estilos CSS Limpios
st.markdown("""
<style>
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
    }
    .stNumberInput label {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v5_final.json'

# --- 2. PERSISTENCIA DE DATOS ---
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

# --- 4. BARRA LATERAL (ECONOMÍA) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=50)
    st.title("Parámetros")
    
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
    conversion_factor = st.number_input("Factor (M3/F = MR)", value=0.65, step=0.01, key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Precio Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)
    
    st.markdown("### 3. Distribución Ingreso")
    h_rev_pct = st.slider("% Ingreso Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    
    # Ingresos proyectados
    h_income_unit = sales_price_mr * (h_rev_pct/100)
    f_income_unit = sales_price_mr * ((100-h_rev_pct)/100)

# --- 5. CUERPO PRINCIPAL ---
st.title("🌲 ForestCost Analytics")

# TABS PARA ORGANIZACIÓN
tab1, tab2, tab3 = st.tabs(["📝 Configuración Operacional", "📊 Dashboard Dinámico", "📋 Planilla Detallada"])

# ==========================================
# TAB 1: CONFIGURACIÓN OPERACIONAL
# ==========================================
with tab1:
    # --- BLOQUE JORNADA (SOLUCIÓN FORWARDER) ---
    st.subheader("📅 1. Jornada Laboral")
    with st.container(border=True):
        col_j1, col_j2 = st.columns(2)
        
        # HARVESTER
        with col_j1:
            st.markdown("##### 🚜 Harvester")
            c1, c2 = st.columns(2)
            h_days = c1.number_input("Días al Mes (H)", value=28, step=1, key="h_days_month", on_change=save_config)
            h_hours = c2.number_input("Horas al Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
            h_total_hours = h_days * h_hours
            st.info(f"Total: **{h_total_hours}** Horas/Mes")

        # FORWARDER (Corrección aquí: inputs independientes)
        with col_j2:
            st.markdown("##### 🚜 Forwarder")
            c3, c4 = st.columns(2)
            f_days = c3.number_input("Días al Mes (F)", value=25, step=1, key="f_days_month", on_change=save_config)
            f_hours = c4.number_input("Horas al Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
            f_total_hours = f_days * f_hours
            st.info(f"Total: **{f_total_hours}** Horas/Mes")

    st.divider()

    # --- BLOQUE COSTOS UNITARIOS ---
    st.subheader("💰 2. Costos Directos")
    
    col_h_in, col_f_in = st.columns(2)

    def machine_inputs(prefix, hours_month, fuel_p, col_obj):
        with col_obj:
            with st.expander(f"✏️ Editar Costos {prefix}", expanded=False):
                st.caption("Ingresa los montos TOTALES MENSUALES")
                
                # Defaults
                def_rent = 10900000 if prefix=="Harvester" else 8000000
                def_sal = 3800000 if prefix=="Harvester" else 1900000
                def_l_hr = 20.0 if prefix=="Harvester" else 15.0
                
                # Inputs
                l_hr = st.number_input(f"Consumo L/Hr", value=def_l_hr, step=1.0, key=f"fuel_l_hr_{prefix}", on_change=save_config)
                fuel_cost = l_hr * hours_month * fuel_p
                
                rent = st.number_input(f"Arriendo ($)", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
                sal = st.number_input(f"Sueldos ($)", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)
                
                st.markdown("**Mantención**")
                m1 = st.number_input("Preventiva", value=800000 if prefix=="Harvester" else 500000, key=f"maint_prev_{prefix}", on_change=save_config)
                m2 = st.number_input("Correctiva", value=500000 if prefix=="Harvester" else 300000, key=f"maint_corr_{prefix}", on_change=save_config)
                m3 = st.number_input("Neumát/Orugas", value=200000, key=f"maint_tires_{prefix}", on_change=save_config)
                
                st.markdown("**Consumibles**")
                c1 = st.number_input("Corte", value=200000, key=f"consum_cut_{prefix}", on_change=save_config)
                c2 = st.number_input("Hidráulico", value=150000, key=f"consum_hyd_{prefix}", on_change=save_config)
                c3 = st.number_input("Lubricantes", value=60000, key=f"consum_lub_{prefix}", on_change=save_config)
                
                oth = st.number_input("Otros", value=0, key=f"others_total_{prefix}", on_change=save_config)
                
                # Total
                total = rent + sal + fuel_cost + m1 + m2 + m3 + c1 + c2 + c3 + oth
                st.success(f"**Total Mes: ${fmt(total)}**")
                return total, total/hours_month if hours_month else 0

    h_total_m, h_total_hr = machine_inputs("Harvester", h_total_hours, fuel_price, col_h_in)
    f_total_m, f_total_hr = machine_inputs("Forwarder", f_total_hours, fuel_price, col_f_in)

    # --- BLOQUE INDIRECTOS ---
    st.subheader("🏢 3. Costos Faena (Indirectos)")
    with st.expander("✏️ Editar Indirectos", expanded=False):
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            pk_l_day = st.number_input("L/Día Camionetas", 12.0, key="pickup_liters_day", on_change=save_config)
            pk_days = st.number_input("Días Uso", 30, key="pickup_days_month", on_change=save_config)
            pk_rent_uf = st.number_input("Arr. Camionetas (UF)", 38.0, key="pickup_rent_uf", on_change=save_config)
        with c_i2:
            staff = st.number_input("Personal ($)", 2164000, key="support_staff", on_change=save_config)
            fac = st.number_input("Instalaciones ($)", 560000, key="facilities", on_change=save_config)
            pen = st.number_input("Pensión ($)", 1890000, key="pension", on_change=save_config)
            oth_sh = st.number_input("Otros Faena ($)", 100000, key="others_shared", on_change=save_config)

        # Cálculo
        pk_fuel = pk_l_day * pk_days * fuel_price
        pk_rent = pk_rent_uf * current_uf
        shared_tot = pk_fuel + pk_rent + staff + fac + pen + oth_sh
        
        st.markdown("---")
        # Asignación
        opt_alloc = ["Porcentaje Manual", "Proporcional Horas"]
        idx = opt_alloc.index(st.session_state.get("alloc_method", "Porcentaje Manual")) if st.session_state.get("alloc_method") in opt_alloc else 0
        method = st.radio("Asignación", opt_alloc, index=idx, key="alloc_method", horizontal=True, on_change=save_config)
        
        if method == "Porcentaje Manual":
            h_pct = st.slider("% a Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
        else:
            tt = h_total_hours + f_total_hours
            h_pct = h_total_hours / tt if tt > 0 else 0.5
        
        shared_h = shared_tot * h_pct
        shared_f = shared_tot * (1 - h_pct)

# ==========================================
# CÁLCULOS FINALES
# ==========================================
final_h_hr = (h_total_m + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_m + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

# ==========================================
# TAB 2: DASHBOARD DINÁMICO
# ==========================================
with tab2:
    st.markdown("### 🚀 Simulación de Rentabilidad")
    
    # Slider principal para dinamismo
    prod_target = st.slider("🎯 Productividad Objetivo (M3/Hr)", 10, 60, 25, step=1)
    mr_target = prod_target / conversion_factor if conversion_factor else 0
    
    # Cálculos al vuelo
    cost_unit_sys = sys_hr / mr_target if mr_target else 0
    margin_sys = sales_price_mr - cost_unit_sys
    margin_pct_sys = (margin_sys / sales_price_mr * 100) if sales_price_mr else 0
    
    # 1. TARJETAS PRINCIPALES
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    def kpi_box(label, value, color="black"):
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi1: kpi_box("Producción MR/Hr", f"{mr_val:.1f}" if 'mr_val' in locals() else f"{mr_target:.1f}")
    with col_kpi2: kpi_box("Costo Unitario", f"${fmt(cost_unit_sys)}")
    with col_kpi3: kpi_box("Utilidad / MR", f"${fmt(margin_sys)}", color="green" if margin_sys > 0 else "red")
    with col_kpi4: kpi_box("Margen %", f"{margin_pct_sys:.1f}%", color="green" if margin_sys > 0 else "red")

    st.markdown("---")

    # 2. GRÁFICOS VISUALES
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.subheader("⚖️ Costo vs Ingreso (Por Máquina)")
        # Costo unitario por maquina
        c_h_u = final_h_hr / mr_target if mr_target else 0
        c_f_u = final_f_hr / mr_target if mr_target else 0
        
        # Datos para gráfico de barras
        data_bar = pd.DataFrame({
            "Máquina": ["Harvester", "Forwarder", "Sistema Total"],
            "Costo Unitario": [c_h_u, c_f_u, cost_unit_sys],
            "Ingreso Unitario": [h_income_unit, f_income_unit, sales_price_mr]
        })
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Costo', x=data_bar['Máquina'], y=data_bar['Costo Unitario'], marker_color='#e74c3c'),
            go.Bar(name='Ingreso', x=data_bar['Máquina'], y=data_bar['Ingreso Unitario'], marker_color='#2ecc71')
        ])
        fig_bar.update_layout(barmode='group', height=400, title="Comparativa $/MR")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g2:
        st.subheader("⚡ Margen Operacional")
        # Gauge Chart (Velocímetro)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = margin_pct_sys,
            title = {'text': "Margen %"},
            gauge = {
                'axis': {'range': [-20, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-20, 0], 'color': "#ffcccc"},
                    {'range': [0, 20], 'color': "#ffffcc"},
                    {'range': [20, 50], 'color': "#ccffcc"}],
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 3: PLANILLA DETALLADA
# ==========================================
with tab3:
    st.subheader("📋 Planilla de Costos (Estilo Excel)")
    
    # Crear DataFrame consolidado
    rows_excel = [
        # Harvester
        ["HARVESTER", "Arriendo", h_total_m * (st.session_state.get(f"rent_total_Harvester",0)/h_total_m if h_total_m else 0), 0],
        ["HARVESTER", "Operadores", st.session_state.get(f"salary_total_Harvester",0), 0],
        ["HARVESTER", "Combustible", h_total_m - (st.session_state.get(f"rent_total_Harvester",0)+st.session_state.get(f"salary_total_Harvester",0)+st.session_state.get(f"maint_prev_Harvester",0)+st.session_state.get(f"maint_corr_Harvester",0)+st.session_state.get(f"maint_tires_Harvester",0)+st.session_state.get(f"consum_cut_Harvester",0)+st.session_state.get(f"consum_hyd_Harvester",0)+st.session_state.get(f"consum_lub_Harvester",0)+st.session_state.get(f"others_total_Harvester",0)), 0], # Aprox logic simplify
        ["HARVESTER", "Mantención", st.session_state.get(f"maint_prev_Harvester",0)+st.session_state.get(f"maint_corr_Harvester",0)+st.session_state.get(f"maint_tires_Harvester",0), 0],
        ["HARVESTER", "Consumibles", st.session_state.get(f"consum_cut_Harvester",0)+st.session_state.get(f"consum_hyd_Harvester",0)+st.session_state.get(f"consum_lub_Harvester",0), 0],
        ["HARVESTER", "Asig. Indirectos", shared_h, 0],
        # Forwarder
        ["FORWARDER", "Arriendo", st.session_state.get(f"rent_total_Forwarder",0), 0],
        ["FORWARDER", "Operadores", st.session_state.get(f"salary_total_Forwarder",0), 0],
        ["FORWARDER", "Asig. Indirectos", shared_f, 0],
    ]
    
    # Nota: Para hacerlo perfecto visualmente usamos st.dataframe con datos planos
    
    data_final = pd.DataFrame([
        {"Máquina": "Harvester", "Ítem": "COSTO DIRECTO", "Mensual": h_total_m, "Hora": h_total_m/h_total_hours},
        {"Máquina": "Harvester", "Ítem": "COSTO INDIRECTO", "Mensual": shared_h, "Hora": shared_h/h_total_hours},
        {"Máquina": "Forwarder", "Ítem": "COSTO DIRECTO", "Mensual": f_total_m, "Hora": f_total_m/f_total_hours},
        {"Máquina": "Forwarder", "Ítem": "COSTO INDIRECTO", "Mensual": shared_f, "Hora": shared_f/f_total_hours},
    ])
    
    st.dataframe(data_final, column_config={
        "Mensual": st.column_config.NumberColumn(format="$%d"),
        "Hora": st.column_config.NumberColumn(format="$%d"),
    }, use_container_width=True)
    
    # Botón descarga
    st.download_button("Descargar Planilla", data_final.to_csv(index=False).encode('utf-8'), "costos.csv", "text/csv")
