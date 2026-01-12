import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # <--- Esta es la librería que faltaba o daba error
import requests
import json
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="ForestCost Analytics Pro",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESTILOS CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 13px;
        color: #6b7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #1f2937;
    }
    .metric-sub {
        font-size: 12px;
        color: #9ca3af;
    }
    .income-box {
        background-color: #f0fdf4;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #bbf7d0;
        margin-bottom: 15px;
        color: #166534;
    }
    .stNumberInput label {
        font-weight: 600;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_pro_final.json'

# --- 2. GESTIÓN DE PERSISTENCIA ---
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

def card(title, value, sub="", help_text=""):
    st.markdown(f"""
    <div class="metric-card" title="{help_text}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
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
    st.title("⚙️ Configuración")
    
    st.markdown("### 1. Economía")
    uf_api, fecha_api = get_uf_value()
    use_auto_uf = st.checkbox("UF Automática", value=True, key="use_auto_uf", on_change=save_config)

    if use_auto_uf and uf_api:
        st.success(f"UF ({fecha_api}): **${fmt(uf_api)}**")
        current_uf = uf_api
    else:
        current_uf = st.number_input("Valor UF Manual", value=39704.93, step=100.0, format="%.2f", key="uf_manual", on_change=save_config)

    fuel_price = st.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

    st.markdown("### 2. Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, format="%.2f", key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Precio Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)

    st.divider()
    st.markdown("### 3. Distribución Ingresos")
    h_rev_pct = st.slider("% Asignado a Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    f_rev_pct = 100 - h_rev_pct
    
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * (f_rev_pct / 100)

    st.markdown(f"""
    <div class="income-box">
        <b>💰 Ingreso Unitario (MR):</b><br>
        🚜 <b>H:</b> ${fmt(h_income)}<br>
        🚜 <b>F:</b> ${fmt(f_income)}
    </div>
    """, unsafe_allow_html=True)

# --- 5. APP PRINCIPAL ---

st.title("🌲 ForestCost Analytics Pro")
st.markdown("**Sistema de Gestión de Costos y Rentabilidad**")

tab_inputs, tab_dashboard, tab_details = st.tabs(["📝 Entrada de Datos Detallada", "📊 Dashboard Ejecutivo", "📉 Análisis Sensibilidad"])

with tab_inputs:
    
    # --- JORNADA OPERACIONAL ---
    with st.expander("📅 1. Jornada Operacional", expanded=True):
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.markdown("#### 🚜 Harvester")
            c1, c2, c3 = st.columns([1, 1, 1])
            h_days = c1.number_input("Días/Mes", value=28, step=1, key="h_days_month", on_change=save_config)
            h_hours_day = c2.number_input("Hrs/Día", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
            h_total_hours = h_days * h_hours_day
            c3.metric("Horas Totales", fmt(h_total_hours))
            
        with col_j2:
            st.markdown("#### 🚜 Forwarder")
            c4, c5, c6 = st.columns([1, 1, 1])
            f_days = c4.number_input("Días/Mes", value=25, step=1, key="f_days_month", on_change=save_config)
            f_hours_day = c5.number_input("Hrs/Día", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
            f_total_hours = f_days * f_hours_day
            c6.metric("Horas Totales", fmt(f_total_hours))

    st.divider()
    st.markdown("### 🚜 2. Estructura de Costos por Máquina")

    col_h_input, col_f_input = st.columns(2)

    def render_machine_section(prefix, hours_month, fuel_p, col_obj):
        with col_obj:
            with st.container(border=True):
                st.subheader(f"{prefix}")
                
                # COMBUSTIBLE
                st.markdown("##### ⛽ Combustible")
                col_f1, col_f2 = st.columns([1, 1.5])
                def_l_hr = 20.0 if prefix=="Harvester" else 15.0
                l_hr = col_f1.number_input(f"Consumo L/Hr", value=def_l_hr, step=1.0, key=f"fuel_l_hr_{prefix}", on_change=save_config)
                
                cost_fuel_hr = l_hr * fuel_p
                cost_fuel_month = cost_fuel_hr * hours_month
                col_f2.metric("Diesel Mensual", f"${fmt(cost_fuel_month)}", f"{l_hr} L/h")

                # FIJOS
                st.markdown("##### 💼 Fijos")
                c_op1, c_op2 = st.columns(2)
                def_rent = 10900000 if prefix=="Harvester" else 8000000
                def_sal = 3800000 if prefix=="Harvester" else 1900000
                rent = c_op1.number_input(f"Arriendo ($)", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
                sal = c_op2.number_input(f"Sueldos ($)", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)

                # MANTENCIÓN
                st.markdown("##### 🔧 Mantención")
                c_m1, c_m2, c_m3 = st.columns(3)
                m_prev = c_m1.number_input("Preventiva", value=800000 if prefix=="Harvester" else 500000, step=50000, key=f"maint_prev_{prefix}", on_change=save_config)
                m_corr = c_m2.number_input("Correctiva", value=500000 if prefix=="Harvester" else 300000, step=50000, key=f"maint_corr_{prefix}", on_change=save_config)
                m_tires = c_m3.number_input("Neum/Oruga", value=200000, step=50000, key=f"maint_tires_{prefix}", on_change=save_config)
                total_maint = m_prev + m_corr + m_tires
                
                # CONSUMIBLES
                st.markdown("##### ⚙️ Consumibles")
                c_c1, c_c2, c_c3 = st.columns(3)
                c_cut = c_c1.number_input("Corte", value=200000, step=20000, key=f"consum_cut_{prefix}", on_change=save_config)
                c_hyd = c_c2.number_input("Hidráulico", value=150000, step=20000, key=f"consum_hyd_{prefix}", on_change=save_config)
                c_lub = c_c3.number_input("Lubricantes", value=60000, step=10000, key=f"consum_lub_{prefix}", on_change=save_config)
                total_consum = c_cut + c_hyd + c_lub

                # OTROS
                others = st.number_input(f"Otros Costos ($)", value=0, step=10000, key=f"others_total_{prefix}", on_change=save_config)

                # Total
                total_month = rent + sal + cost_fuel_month + total_maint + total_consum + others
                tot_hr = total_month / hours_month if hours_month > 0 else 0
                
                st.markdown("---")
                st.info(f"**Total {prefix}: ${fmt(tot_hr)} /hora**")
                
                return {"total_month": total_month, "hours_month": hours_month}

    h_data = render_machine_section("Harvester", h_total_hours, fuel_price, col_h_input)
    f_data = render_machine_section("Forwarder", f_total_hours, fuel_price, col_f_input)

    st.divider()
    
    # --- COSTOS INDIRECTOS ---
    st.markdown("### 🏢 3. Costos Indirectos y Faena")
    with st.expander("Configuración Costos Indirectos", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🛻 Camionetas (Consumo Diario)")
            
            cp1, cp2 = st.columns(2)
            pickup_l_day = cp1.number_input("L/Día (Total)", value=12.0, step=1.0, key="pickup_liters_day", on_change=save_config)
            pickup_days = cp2.number_input("Días Uso/Mes", value=30, key="pickup_days_month", on_change=save_config)

            p_fuel_cost_month = pickup_l_day * pickup_days * fuel_price
            
            p_rent_uf = st.number_input("Arriendo UF/Mes", value=38.0, step=0.5, key="pickup_rent_uf", on_change=save_config)
            p_rent_clp = p_rent_uf * current_uf
            
            st.caption(f"Diesel: ${fmt(p_fuel_cost_month)} | Arriendo: ${fmt(p_rent_clp)}")
            
        with c2:
            st.markdown("##### 🛠️ Gastos Generales")
            staff = st.number_input("Personal Apoyo ($)", value=2164000, step=100000, key="support_staff", on_change=save_config)
            fac = st.number_input("Instalaciones ($)", value=560000, step=50000, key="facilities", on_change=save_config)
            pen = st.number_input("Pensión/Alojamiento ($)", value=1890000, step=50000, key="pension", on_change=save_config)
            oth = st.number_input("Otros Faena ($)", value=100000, step=10000, key="others_shared", on_change=save_config)
            
            total_shared = p_rent_clp + p_fuel_cost_month + staff + fac + pen + oth
            st.success(f"**Total Indirectos: ${fmt(total_shared)}**")
        
        st.markdown("---")
        opts = ["Porcentaje Manual", "Proporcional a Horas"]
        idx = opts.index(st.session_state.get("alloc_method", "Porcentaje Manual")) if st.session_state.get("alloc_method") in opts else 0
        alloc = st.radio("Método de Asignación", opts, index=idx, key="alloc_method", horizontal=True, on_change=save_config)
        
        if alloc == "Porcentaje Manual":
            h_pct = st.slider("% Asignado a Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
            f_pct = 1.0 - h_pct
        else:
            tot_h = h_total_hours + f_total_hours
            h_pct = h_total_hours / tot_h if tot_h > 0 else 0.5
            f_pct = f_total_hours / tot_h if tot_h > 0 else 0.5

# ==========================================
# CÁLCULOS CENTRALIZADOS
# ==========================================
def calc_final(machine_data, shared_tot, pct):
    if machine_data["hours_month"] == 0: return 0
    direct = machine_data["total_month"]
    allocated = shared_tot * pct
    return (direct + allocated) / machine_data["hours_month"]

h_cost_hr = calc_final(h_data, total_shared, h_pct)
f_cost_hr = calc_final(f_data, total_shared, f_pct)
sys_cost_hr = h_cost_hr + f_cost_hr

# Datos Sensibilidad
prod_m3 = np.arange(10, 51, 1)
rows = []
for m3 in prod_m3:
    mr = m3 / conversion_factor if conversion_factor else 0
    c_h_mr = h_cost_hr / mr if mr > 0 else 0
    c_f_mr = f_cost_hr / mr if mr > 0 else 0
    
    # Márgenes Individuales y Totales
    m_h = h_income - c_h_mr
    m_f = f_income - c_f_mr
    m_sys = sales_price_mr - (c_h_mr + c_f_mr)
    
    pct_h = (m_h / h_income * 100) if h_income else 0
    pct_f = (m_f / f_income * 100) if f_income else 0
    pct_sys = (m_sys / sales_price_mr * 100) if sales_price_mr else 0
    
    rows.append({
        "Prod M3": m3,
        "Prod MR": mr,
        "Margen H %": pct_h,
        "Margen F %": pct_f,
        "Margen Total %": pct_sys
    })
df_sens = pd.DataFrame(rows)

# ==========================================
# TAB 2: DASHBOARD
# ==========================================
with tab_dashboard:
    st.markdown("### 📊 Tablero de Control Ejecutivo")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: card("Costo Hora Harvester", f"${fmt(h_cost_hr)}", "Incl. Indirectos")
    with k2: card("Costo Hora Forwarder", f"${fmt(f_cost_hr)}", "Incl. Indirectos")
    with k3: card("Costo Hora Sistema", f"${fmt(sys_cost_hr)}", "H + F")
    with k4: card("Presupuesto Faena", f"${fmt((h_data['total_month']+f_data['total_month']+total_shared)/1e6)} M", "Total Mensual")

    st.markdown("---")

    g1, g2 = st.columns([2, 1])
    with g1:
        st.subheader("Curvas de Rentabilidad")
        # Gráfico Multilínea
        fig = px.line(df_sens, x="Prod MR", y=["Margen Total %", "Margen H %", "Margen F %"], 
                      title="Comparativa de Márgenes (%) vs Productividad",
                      markers=True,
                      color_discrete_map={
                          "Margen Total %": "#2e7d32",
                          "Margen H %": "#1e88e5",
                          "Margen F %": "#ff8f00"
                      })
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(yaxis_tickformat=".0%", legend_title_text="Rentabilidad")
        st.plotly_chart(fig, use_container_width=True)
    
    with g2:
        st.subheader("Costos Globales")
        # Grafico Torta
        labels = ["Harvester Directo", "Forwarder Directo", "Indirectos"]
        values = [h_data["total_month"], f_data["total_month"], total_shared]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 3: SENSIBILIDAD
# ==========================================
with tab_details:
    st.subheader("📉 Tabla de Sensibilidad Detallada")
    
    st.dataframe(df_sens, column_config={
        "Prod M3": st.column_config.NumberColumn("M3/hr", format="%d"),
        "Prod MR": st.column_config.NumberColumn("MR/hr", format="%.1f"),
        "Margen H %": st.column_config.ProgressColumn("Margen Harvester", format="%.1f%%", min_value=-0.5, max_value=0.5),
        "Margen F %": st.column_config.ProgressColumn("Margen Forwarder", format="%.1f%%", min_value=-0.5, max_value=0.5),
        "Margen Total %": st.column_config.ProgressColumn("Margen Global", format="%.1f%%", min_value=-0.5, max_value=0.5)
    }, hide_index=True, use_container_width=True)
    
    csv = df_sens.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Reporte (CSV)", csv, "reporte_forestal_pro.csv", "text/csv")
