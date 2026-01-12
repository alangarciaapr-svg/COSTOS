import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN INICIAL Y ESTILO ---
st.set_page_config(
    page_title="ForestCost Analytics",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para "toque profesional" (Tarjetas de métricas, fuentes, espacios)
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #1a1a1a;
    }
    .metric-sub {
        font-size: 12px;
        color: #888;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_pro.json'

# --- 2. GESTIÓN DE PERSISTENCIA (AUTOGUARDADO) ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    # Guardamos las claves relevantes del session_state
    config_data = {k: v for k, v in st.session_state.items() if k in EXPECTED_KEYS}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f)

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    # Harvester
    "rent_total_Harvester", "fuel_liters_total_Harvester", "salary_total_Harvester", 
    "maint_total_Harvester", "consumables_total_Harvester", "others_total_Harvester",
    # Forwarder
    "rent_total_Forwarder", "fuel_liters_total_Forwarder", "salary_total_Forwarder", 
    "maint_total_Forwarder", "consumables_total_Forwarder", "others_total_Forwarder",
    # Shared
    "pickup_rent_uf", "pickup_fuel", "support_staff", 
    "facilities", "pension", "others_shared", "alloc_method", "h_share_pct_manual"
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

def card(title, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
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

# --- 4. SIDEBAR (PARÁMETROS GLOBALES) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2038/2038261.png", width=60) # Icono genérico
    st.title("Config. Global")
    
    st.markdown("### 💰 Economía")
    uf_api, fecha_api = get_uf_value()
    use_auto_uf = st.checkbox("UF Automática (API)", value=True, key="use_auto_uf", on_change=save_config)

    if use_auto_uf and uf_api:
        st.info(f"UF Hoy ({fecha_api}): ${fmt(uf_api)}")
        current_uf = uf_api
    else:
        current_uf = st.number_input("Valor UF Manual", value=39704.93, step=100.0, format="%.2f", key="uf_manual", on_change=save_config)

    fuel_price = st.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

    st.divider()
    st.markdown("### 🌲 Comercial & Conversión")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, format="%.2f", key="conversion_factor", help="Ej: 1 M3 Sólido / 0.65 = 1.54 MR", on_change=save_config)
    sales_price_mr = st.number_input("Precio Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)

    st.markdown("Distr. Ingresos (Harvester/Fwd)")
    h_rev_pct = st.slider("", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    st.caption(f"Harvester: {h_rev_pct}% | Forwarder: {100-h_rev_pct}%")
    
    h_price_mr = sales_price_mr * (h_rev_pct / 100)
    f_price_mr = sales_price_mr * ((100 - h_rev_pct) / 100)

# --- 5. CUERPO PRINCIPAL (TABS) ---

st.title("🌲 ForestCost Analytics")
st.markdown("**Sistema de Costeo y Rentabilidad de Maquinaria Forestal**")

tab_inputs, tab_dashboard, tab_details = st.tabs(["📝 Entrada de Datos", "📊 Dashboard Ejecutivo", "📉 Análisis Sensibilidad"])

# ==========================================
# TAB 1: ENTRADA DE DATOS
# ==========================================
with tab_inputs:
    st.info("ℹ️ Ingrese los **Totales Mensuales** y parámetros de jornada. El sistema calculará automáticamente los costos unitarios.")
    
    # --- JORNADA ---
    with st.container():
        st.subheader("1. Jornada Operacional")
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.markdown("**🚜 Harvester**")
            c1, c2 = st.columns(2)
            h_days = c1.number_input("Días/Mes (H)", value=28, key="h_days_month", on_change=save_config)
            h_hours_day = c2.number_input("Hrs/Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
            h_total_hours = h_days * h_hours_day
            st.caption(f"Total: **{fmt(h_total_hours)} hrs/mes**")
            
        with col_j2:
            st.markdown("**🚜 Forwarder**")
            c3, c4 = st.columns(2)
            f_days = c3.number_input("Días/Mes (F)", value=25, key="f_days_month", on_change=save_config)
            f_hours_day = c4.number_input("Hrs/Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
            f_total_hours = f_days * f_hours_day
            st.caption(f"Total: **{fmt(f_total_hours)} hrs/mes**")
    
    st.divider()

    # --- COSTOS MAQUINARIA ---
    st.subheader("2. Estructura de Costos Mensuales")
    
    col_maq1, col_maq2 = st.columns(2)
    
    # FUNCIÓN HELPER PARA INPUTS DE MAQUINA
    def render_machine_inputs(prefix, hours_month, fuel_p, col_obj):
        with col_obj:
            st.markdown(f"### {prefix}")
            with st.expander(f"Ver Detalle Costos {prefix}", expanded=True):
                # Defaults
                def_rent = 10900000 if prefix=="Harvester" else 8000000
                def_sal = 3800000 if prefix=="Harvester" else 1900000
                def_liters = 5600.0 if prefix=="Harvester" else 3000.0
                def_maint = 1500000 if prefix=="Harvester" else 900000
                def_consum = 410000 if prefix=="Harvester" else 200000
                
                rent = st.number_input(f"Arriendo Total ($)", value=def_rent, step=100000, key=f"rent_total_{prefix}", on_change=save_config)
                sal = st.number_input(f"Sueldos Operadores ($)", value=def_sal, step=100000, key=f"salary_total_{prefix}", on_change=save_config)
                
                # Petróleo destacado
                c_f1, c_f2 = st.columns([2, 1])
                liters = c_f1.number_input(f"Petróleo (Litros)", value=def_liters, step=100.0, key=f"fuel_liters_total_{prefix}", on_change=save_config)
                fuel_cost = liters * fuel_p
                c_f2.metric("Costo Comb.", f"${fmt(fuel_cost/1000)}k", help=f"Total: ${fmt(fuel_cost)}")
                
                maint = st.number_input(f"Mantención ($)", value=def_maint, step=100000, key=f"maint_total_{prefix}", on_change=save_config)
                consum = st.number_input(f"Consumibles ($)", value=def_consum, step=50000, key=f"consumables_total_{prefix}", on_change=save_config)
                others = st.number_input(f"Otros ($)", value=0, step=10000, key=f"others_total_{prefix}", on_change=save_config)
                
                total_m = rent + sal + fuel_cost + maint + consum + others
                cost_hr = total_m / hours_month if hours_month else 0
                
                st.markdown(f"**Total Mes:** :blue[${fmt(total_m)}] | **Hora:** :green[${fmt(cost_hr)}] ")
                
                return {"total_month": total_m, "hours_month": hours_month}

    h_data = render_machine_inputs("Harvester", h_total_hours, fuel_price, col_maq1)
    f_data = render_machine_inputs("Forwarder", f_total_hours, fuel_price, col_maq2)

    st.divider()

    # --- COSTOS COMPARTIDOS ---
    st.subheader("3. Costos Indirectos (Faena)")
    with st.expander("Desplegar Costos Compartidos (Camionetas, Instalaciones, etc.)", expanded=False):
        c_sh1, c_sh2, c_sh3 = st.columns(3)
        with c_sh1:
            st.markdown("**Transporte**")
            p_rent_uf = st.number_input("Arr. Camionetas (UF)", value=38.0, step=0.5, key="pickup_rent_uf", on_change=save_config)
            p_rent_clp = p_rent_uf * current_uf
            st.caption(f"= ${fmt(p_rent_clp)}")
            p_fuel = st.number_input("Comb. Camionetas ($)", value=535104, step=50000, key="pickup_fuel", on_change=save_config)
        
        with c_sh2:
            st.markdown("**Personal & Admin**")
            staff = st.number_input("Personal Apoyo ($)", value=2164000, step=100000, key="support_staff", on_change=save_config)
            fac = st.number_input("Instalación/Gastos ($)", value=560000, step=50000, key="facilities", on_change=save_config)
        
        with c_sh3:
            st.markdown("**Varios**")
            pen = st.number_input("Alojamiento ($)", value=1890000, step=50000, key="pension", on_change=save_config)
            oth = st.number_input("Otros Faena ($)", value=100000, step=10000, key="others_shared", on_change=save_config)
        
        total_shared = p_rent_clp + p_fuel + staff + fac + pen + oth
        st.success(f"**Total Costos Indirectos:** ${fmt(total_shared)}")
        
        # Asignación
        st.markdown("---")
        opts = ["Porcentaje Manual", "Proporcional a Horas"]
        idx = opts.index(st.session_state.get("alloc_method", "Porcentaje Manual")) if st.session_state.get("alloc_method") in opts else 0
        alloc = st.radio("Método Distribución", opts, index=idx, horizontal=True, key="alloc_method", on_change=save_config)
        
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

# ==========================================
# TAB 2: DASHBOARD EJECUTIVO
# ==========================================
with tab_dashboard:
    st.markdown("### 📊 Indicadores Clave de Desempeño (KPIs)")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        card("Costo Hora Harvester", f"${fmt(h_cost_hr)}", "Incluye costos indirectos")
    with col_kpi2:
        card("Costo Hora Forwarder", f"${fmt(f_cost_hr)}", "Incluye costos indirectos")
    with col_kpi3:
        card("Costo Hora Sistema", f"${fmt(sys_cost_hr)}", "Harvester + Forwarder")
    with col_kpi4:
        total_faena = h_data['total_month'] + f_data['total_month'] + total_shared
        card("Gasto Mensual Faena", f"${fmt(total_faena/1000000)} M", "Presupuesto Total")

    st.markdown("---")
    
    col_gr1, col_gr2 = st.columns([2, 1])
    
    with col_gr1:
        st.subheader("Rentabilidad vs Productividad")
        fig_prof = px.line(df_sens, x="Prod MR", y="Margen %", 
                          title="Margen de Utilidad (%) según Productividad",
                          markers=True, height=400)
        fig_prof.add_hline(y=0, line_dash="dash", line_color="red")
        fig_prof.update_layout(template="plotly_white", xaxis_title="Metros Ruma / Hora", yaxis_tickformat=".1%")
        st.plotly_chart(fig_prof, use_container_width=True)
        
    with col_gr2:
        st.subheader("Distribución Costo Hora")
        # Grafico de torta simple del sistema
        labels = ["Harvester", "Forwarder"]
        values = [h_cost_hr, f_cost_hr]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig_pie.update_layout(template="plotly_white", margin=dict(t=0, b=0, l=0, r=0), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Incidencia de cada máquina en el costo horario total del sistema.")

# ==========================================
# TAB 3: DETALLE Y SENSIBILIDAD
# ==========================================
with tab_details:
    st.subheader("📉 Tabla de Sensibilidad Detallada")
    st.markdown("Análisis de margen y costos unitarios variando la productividad horaria.")
    
    # Configuración de columnas para tabla profesional
    st.dataframe(
        df_sens,
        column_config={
            "Prod M3": st.column_config.NumberColumn("Prod (M3/hr)", format="%d"),
            "Prod MR": st.column_config.NumberColumn("Prod (MR/hr)", format="%.1f"),
            "Costo H": st.column_config.NumberColumn("Costo H ($/MR)", format="$%d"),
            "Costo F": st.column_config.NumberColumn("Costo F ($/MR)", format="$%d"),
            "Costo Total": st.column_config.NumberColumn("Costo Sys ($/MR)", format="$%d"),
            "Utilidad": st.column_config.NumberColumn("Utilidad ($/MR)", format="$%d"),
            "Margen %": st.column_config.ProgressColumn(
                "Margen %", 
                format="%.1f%%", 
                min_value=-0.5, 
                max_value=0.5
            ),
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Botón Descarga
    csv = df_sens.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Reporte Completo (CSV)",
        data=csv,
        file_name=f"reporte_forestal_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
