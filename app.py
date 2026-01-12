import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="ForestCost Analytics",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
    }
    .metric-sub {
        font-size: 12px;
        color: #374151;
    }
    .fuel-card {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffe0b2;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_fuel_v3.json'

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
    "maint_total_Harvester", "consumables_total_Harvester", "others_total_Harvester",
    # Forwarder
    "rent_total_Forwarder", "fuel_l_hr_Forwarder", "salary_total_Forwarder", 
    "maint_total_Forwarder", "consumables_total_Forwarder", "others_total_Forwarder",
    # Shared
    "pickup_rent_uf", "pickup_fuel_l_hr", "pickup_hours_day", "pickup_days_month",
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
    use_auto_uf = st.checkbox("UF Automática (API)", value=True, key="use_auto_uf", on_change=save_config)

    if use_auto_uf and uf_api:
        st.success(f"UF ({fecha_api}): **${fmt(uf_api)}**")
        current_uf = uf_api
    else:
        current_uf = st.number_input("Valor UF Manual", value=39704.93, step=100.0, format="%.2f", key="uf_manual", on_change=save_config)

    fuel_price = st.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

    st.markdown("### 2. Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, format="%.2f", key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Precio Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)

    st.markdown("Distr. Ingresos")
    h_rev_pct = st.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    h_price_mr = sales_price_mr * (h_rev_pct / 100)
    f_price_mr = sales_price_mr * ((100 - h_rev_pct) / 100)

# --- 5. APP PRINCIPAL ---

st.title("🌲 ForestCost Analytics")
st.markdown("**Sistema Profesional de Costeo de Maquinaria**")

tab_inputs, tab_dashboard, tab_details = st.tabs(["📝 Entrada de Datos", "📊 Dashboard", "📉 Sensibilidad"])

with tab_inputs:
    
    # A. JORNADA
    with st.expander("📅 1. Jornada Operacional (Días y Horas)", expanded=True):
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.markdown("#### 🚜 Harvester")
            c1, c2, c3 = st.columns([1, 1, 1])
            h_days = c1.number_input("Días/Mes", value=28, key="h_days_month", on_change=save_config)
            h_hours_day = c2.number_input("Hrs/Día", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
            h_total_hours = h_days * h_hours_day
            c3.metric("Horas Mensuales", fmt(h_total_hours))
            
        with col_j2:
            st.markdown("#### 🚜 Forwarder")
            c4, c5, c6 = st.columns([1, 1, 1])
            f_days = c4.number_input("Días/Mes", value=25, key="f_days_month", on_change=save_config)
            f_hours_day = c5.number_input("Hrs/Día", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
            f_total_hours = f_days * f_hours_day
            c6.metric("Horas Mensuales", fmt(f_total_hours))

    st.divider()
    st.markdown("### 🚜 2. Costos Maquinaria")

    col_h_input, col_f_input = st.columns(2)

    def render_machine_section(prefix, hours_month, days_month, hours_day, fuel_p, col_obj):
        with col_obj:
            with st.container(border=True):
                st.subheader(f"{prefix}")
                
                # Defaults
                def_rent = 10900000 if prefix=="Harvester" else 8000000
                def_sal = 3800000 if prefix=="Harvester" else 1900000
                def_l_hr = 20.0 if prefix=="Harvester" else 15.0
                def_maint = 1500000 if prefix=="Harvester" else 900000
                def_consum = 410000 if prefix=="Harvester" else 200000
                
                # --- COMBUSTIBLE (NUEVO: L/H) ---
                st.markdown("##### ⛽ Combustible")
                col_f1, col_f2 = st.columns([1, 2])
                l_hr = col_f1.number_input(f"L/Hr {prefix}", value=def_l_hr, step=1.0, key=f"fuel_l_hr_{prefix}", on_change=save_config)
                
                # Cálculos Combustible
                cost_fuel_hr = l_hr * fuel_p
                cost_fuel_day = cost_fuel_hr * hours_day
                cost_fuel_month = cost_fuel_day * days_month
                
                col_f2.info(f"**${fmt(cost_fuel_hr)} /hr** | ${fmt(cost_fuel_day)} /día | ${fmt(cost_fuel_month)} /mes")

                # --- OTROS COSTOS (TOTALES) ---
                st.markdown("##### ⚙️ Otros Costos (Total Mensual)")
                rent = st.number_input(f"Arriendo Total ($) {prefix}", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
                sal = st.number_input(f"Sueldos Total ($) {prefix}", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)
                maint = st.number_input(f"Mantención Total ($) {prefix}", value=def_maint, step=100000, key=f"maint_total_{prefix}", on_change=save_config)
                consum = st.number_input(f"Consumibles Total ($) {prefix}", value=def_consum, step=50000, key=f"consumables_total_{prefix}", on_change=save_config)
                others = st.number_input(f"Otros ($) {prefix}", value=0, step=10000, key=f"others_total_{prefix}", on_change=save_config)

                # Totalizador
                total_month = rent + sal + cost_fuel_month + maint + consum + others
                tot_hr = total_month / hours_month if hours_month > 0 else 0
                
                st.markdown("---")
                st.markdown(f"**TOTAL {prefix}:** :green[${fmt(tot_hr)} /hr]")
                st.caption(f"Total Mes: ${fmt(total_month)}")
                
                return {"total_month": total_month, "hours_month": hours_month}

    h_data = render_machine_section("Harvester", h_total_hours, h_days, h_hours_day, fuel_price, col_h_input)
    f_data = render_machine_section("Forwarder", f_total_hours, f_days, f_hours_day, fuel_price, col_f_input)

    st.divider()
    
    # --- COSTOS COMPARTIDOS ---
    st.markdown("### 🏢 3. Costos Indirectos y Camionetas")
    with st.expander("Ver Detalle Faena", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🛻 Camionetas (Consumo L/h)")
            
            col_p1, col_p2 = st.columns(2)
            # Input L/h camionetas
            pickup_l_hr = col_p1.number_input("L/Hr Camioneta", value=1.5, step=0.1, key="pickup_fuel_l_hr", on_change=save_config)
            pickup_hrs_day = col_p2.number_input("Hrs Uso/Día", value=4.0, step=0.5, key="pickup_hours_day", on_change=save_config)
            pickup_days = st.number_input("Días Uso/Mes", value=30, key="pickup_days_month", on_change=save_config)

            # Cálculo Petróleo Camioneta
            p_fuel_cost_hr = pickup_l_hr * fuel_price
            p_fuel_cost_day = p_fuel_cost_hr * pickup_hrs_day
            p_fuel_cost_month = p_fuel_cost_day * pickup_days
            
            st.markdown(f"""
            <div class="fuel-card">
                <b>⛽ Gasto Diesel Camionetas:</b><br>
                • Hora: ${fmt(p_fuel_cost_hr)}<br>
                • Día: ${fmt(p_fuel_cost_day)}<br>
                • Mes: <b>${fmt(p_fuel_cost_month)}</b>
            </div>
            """, unsafe_allow_html=True)
            
            p_rent_uf = st.number_input("Arriendo Camionetas (UF/Mes)", value=38.0, step=0.5, key="pickup_rent_uf", on_change=save_config)
            p_rent_clp = p_rent_uf * current_uf
            
        with c2:
            st.markdown("##### 🛠️ Otros Gastos Faena")
            staff = st.number_input("Personal Apoyo ($)", value=2164000, step=100000, key="support_staff", on_change=save_config)
            fac = st.number_input("Instalaciones ($)", value=560000, step=50000, key="facilities", on_change=save_config)
            pen = st.number_input("Pensión/Alojamiento ($)", value=1890000, step=50000, key="pension", on_change=save_config)
            oth = st.number_input("Otros Faena ($)", value=100000, step=10000, key="others_shared", on_change=save_config)
            
            total_shared = p_rent_clp + p_fuel_cost_month + staff + fac + pen + oth
            st.success(f"**Total Costos Indirectos: ${fmt(total_shared)}**")
        
        # Asignación
        st.markdown("---")
        st.caption("Distribución de Costos Indirectos")
        opts = ["Porcentaje Manual", "Proporcional a Horas"]
        idx = opts.index(st.session_state.get("alloc_method", "Porcentaje Manual")) if st.session_state.get("alloc_method") in opts else 0
        alloc = st.radio("Método", opts, index=idx, key="alloc_method", horizontal=True, on_change=save_config)
        
        if alloc == "Porcentaje Manual":
            h_pct = st.slider("% Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
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

# Generar datos de sensibilidad
prod_m3 = np.arange(10, 51, 1)
rows = []
for m3 in prod_m3:
    mr = m3 / conversion_factor if conversion_factor else 0
    c_h_mr = h_cost_hr / mr if mr > 0 else 0
    c_f_mr = f_cost_hr / mr if mr > 0 else 0
    m_sys = sales_price_mr - (c_h_mr + c_f_mr)
    
    rows.append({
        "Prod M3": m3,
        "Prod MR": mr,
        "Costo H": c_h_mr,
        "Costo F": c_f_mr,
        "Costo Total": c_h_mr + c_f_mr,
        "Utilidad": m_sys,
        "Margen %": (m_sys/sales_price_mr) if sales_price_mr else 0
    })
df_sens = pd.DataFrame(rows)

with tab_dashboard:
    st.markdown("### 📊 KPIs de Proyecto")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: card("Costo Hora Harvester", f"${fmt(h_cost_hr)}", "Total c/ Indirectos")
    with k2: card("Costo Hora Forwarder", f"${fmt(f_cost_hr)}", "Total c/ Indirectos")
    with k3: card("Costo Hora Sistema", f"${fmt(sys_cost_hr)}", "H + F")
    with k4: card("Costo Mensual Faena", f"${fmt(h_data['total_month']+f_data['total_month']+total_shared)}", "Gasto Total")

    st.markdown("---")

    g1, g2 = st.columns([2, 1])
    with g1:
        st.subheader("Rentabilidad vs Productividad")
        fig = px.line(df_sens, x="Prod MR", y="Margen %", markers=True, title="Margen (%) según MR/hr", color_discrete_sequence=["#2e7d32"])
        fig.add_hline(y=0, line_dash="dot", line_color="red")
        fig.update_layout(xaxis_title="MR / Hora", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    with g2:
        st.subheader("Costos")
        # Pie chart con desglose de Combustible vs Resto
        # Calculamos totales globales
        total_fuel_h = h_data["total_month"] - (st.session_state.get("rent_total_Harvester",0) + st.session_state.get("salary_total_Harvester",0) + st.session_state.get("maint_total_Harvester",0) + st.session_state.get("consumables_total_Harvester",0)) # Aprox logic
        # Mejor usar los datos directos calculados si pudiéramos pasarlos, pero aquí haremos un pie simple H vs F vs Shared
        labels = ["Harvester", "Forwarder", "Indirectos"]
        vals = [h_data["total_month"], f_data["total_month"], total_shared]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=.4)])
        st.plotly_chart(fig_pie, use_container_width=True)

with tab_details:
    st.dataframe(df_sens, column_config={
        "Prod M3": st.column_config.NumberColumn("M3/hr"),
        "Prod MR": st.column_config.NumberColumn("MR/hr", format="%.1f"),
        "Costo Total": st.column_config.NumberColumn("Costo/MR", format="$%d"),
        "Utilidad": st.column_config.NumberColumn("Utilidad/MR", format="$%d"),
        "Margen %": st.column_config.ProgressColumn("Margen", format="%.1f%%", min_value=-0.5, max_value=0.5)
    }, hide_index=True, use_container_width=True)
    
    csv = df_sens.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV", csv, "reporte.csv", "text/csv")
