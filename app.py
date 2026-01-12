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
    page_title="ForestCost Analytics Pro",
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
    .income-box {
        background-color: #f0fdf4;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #bbf7d0;
        margin-bottom: 15px;
        color: #166534;
    }
    .stNumberInput label { font-weight: 600; color: #374151; }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v11_fixed.json'

# --- 2. GESTIÓN DE PERSISTENCIA (CORREGIDA) ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    # CORRECCIÓN: Convertir DataFrames a diccionarios antes de guardar
    config_data = {}
    for k, v in st.session_state.items():
        if k in EXPECTED_KEYS:
            if isinstance(v, pd.DataFrame):
                config_data[k] = v.to_dict('records') # Serializar tabla
            else:
                config_data[k] = v
                
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f)

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    # Shared
    "pickup_rent_uf", "pickup_liters_day", "pickup_days_month",
    "support_staff", "facilities", "pension", "others_shared", "alloc_method", "h_share_pct_manual",
    # Simulador
    "sim_m3_h", "sim_m3_f",
    # Dataframes (Tablas)
    "df_harvester_values", "df_forwarder_values"
]

# Inicialización segura
if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            # Si es una de las tablas, convertir de lista a DataFrame
            if key in ["df_harvester_values", "df_forwarder_values"]:
                st.session_state[key] = pd.DataFrame(val)
            else:
                st.session_state[key] = val
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

tab_inputs, tab_dashboard, tab_details = st.tabs(["📝 Planilla de Costos (Editable)", "📊 Dashboard Ejecutivo", "📉 Simulador por Máquina"])

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
    st.markdown("### 🚜 2. Estructura de Costos (Tablas Editables)")
    st.info("💡 Edita los valores en la columna **'Valor Mensual / Base'**. El sistema recalculará los totales automáticamente.")

    col_h_table, col_f_table = st.columns(2)

    def render_machine_table(prefix, hours_month, fuel_p, col_obj):
        with col_obj:
            st.subheader(f"🛠️ {prefix}")
            
            # Estructura de Datos Inicial (Si no existe en session_state)
            default_data = [
                {"Categoría": "Fijos", "Ítem": "Arriendo Maquinaria", "Valor": 10900000 if prefix=="Harvester" else 8000000, "Unidad": "$/Mes"},
                {"Categoría": "Fijos", "Ítem": "Sueldos Operadores", "Valor": 3800000 if prefix=="Harvester" else 1900000, "Unidad": "$/Mes"},
                {"Categoría": "Variable", "Ítem": "Consumo Combustible", "Valor": 20.0 if prefix=="Harvester" else 15.0, "Unidad": "Litros/Hora"},
                {"Categoría": "Mantención", "Ítem": "Preventiva", "Valor": 800000 if prefix=="Harvester" else 500000, "Unidad": "$/Mes"},
                {"Categoría": "Mantención", "Ítem": "Correctiva", "Valor": 500000 if prefix=="Harvester" else 300000, "Unidad": "$/Mes"},
                {"Categoría": "Mantención", "Ítem": "Neumáticos/Orugas", "Valor": 200000, "Unidad": "$/Mes"},
                {"Categoría": "Consumibles", "Ítem": "Elementos Corte", "Valor": 200000, "Unidad": "$/Mes"},
                {"Categoría": "Consumibles", "Ítem": "Aceite Hidráulico", "Valor": 150000, "Unidad": "$/Mes"},
                {"Categoría": "Consumibles", "Ítem": "Lubricantes/Filtros", "Valor": 60000, "Unidad": "$/Mes"},
                {"Categoría": "Otros", "Ítem": "Varios", "Valor": 0, "Unidad": "$/Mes"},
            ]
            
            # Cargar o Inicializar
            key_df = f"df_{prefix.lower()}_values"
            if key_df not in st.session_state:
                st.session_state[key_df] = pd.DataFrame(default_data)
            
            # EDITOR DE DATOS
            edited_df = st.data_editor(
                st.session_state[key_df],
                column_config={
                    "Valor": st.column_config.NumberColumn("Valor (Mensual o Base)", format="%d", required=True),
                    "Categoría": st.column_config.TextColumn(disabled=True),
                    "Ítem": st.column_config.TextColumn(disabled=True),
                    "Unidad": st.column_config.TextColumn(disabled=True),
                },
                hide_index=True,
                key=f"editor_{prefix}",
                use_container_width=True
            )
            
            # Actualizar estado para persistencia
            st.session_state[key_df] = edited_df
            save_config() # Guardar cambios

            # --- PROCESAMIENTO ---
            def get_val(item_name):
                row = edited_df[edited_df["Ítem"] == item_name]
                return row.iloc[0]["Valor"] if not row.empty else 0

            rent = get_val("Arriendo Maquinaria")
            sal = get_val("Sueldos Operadores")
            l_hr = get_val("Consumo Combustible")
            m_prev = get_val("Preventiva")
            m_corr = get_val("Correctiva")
            m_tires = get_val("Neumáticos/Orugas")
            c_cut = get_val("Elementos Corte")
            c_hyd = get_val("Aceite Hidráulico")
            c_lub = get_val("Lubricantes/Filtros")
            others = get_val("Varios")

            # Cálculos
            fuel_cost_month = l_hr * hours_month * fuel_p
            total_maint = m_prev + m_corr + m_tires
            total_consum = c_cut + c_hyd + c_lub
            
            total_month = rent + sal + fuel_cost_month + total_maint + total_consum + others
            total_hr = total_month / hours_month if hours_month else 0
            
            # Resumen Visual
            st.info(f"**Total {prefix}: ${fmt(total_hr)} /hora**")
            with st.expander(f"Ver Resumen {prefix}", expanded=False):
                st.write(f"Combustible ({l_hr} L/h): **${fmt(fuel_cost_month)}**")
                st.write(f"Mantención Total: **${fmt(total_maint)}**")
                st.write(f"Total Mes: **${fmt(total_month)}**")

            return total_month, total_hr

    h_total_m, h_total_hr = render_machine_table("Harvester", h_total_hours, fuel_price, col_h_table)
    f_total_m, f_total_hr = render_machine_table("Forwarder", f_total_hours, fuel_price, col_f_table)

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
def calc_final(total_m, hours_m, shared_tot, pct):
    if hours_m == 0: return 0
    direct = total_m
    allocated = shared_tot * pct
    return (direct + allocated) / hours_m

h_cost_hr = calc_final(h_total_m, h_total_hours, total_shared, h_pct)
f_cost_hr = calc_final(f_total_m, f_total_hours, total_shared, f_pct)
sys_cost_hr = h_cost_hr + f_cost_hr

# Datos Sensibilidad
prod_m3 = np.arange(10, 51, 1)
rows = []
for m3 in prod_m3:
    mr = m3 / conversion_factor if conversion_factor else 0
    c_h_mr = h_cost_hr / mr if mr > 0 else 0
    c_f_mr = f_cost_hr / mr if mr > 0 else 0
    
    # Márgenes
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
    with k4: card("Presupuesto Faena", f"${fmt((h_total_m+f_total_m+total_shared)/1e6)} M", "Total Mensual")

    st.markdown("---")

    g1, g2 = st.columns([2, 1])
    with g1:
        st.subheader("Curvas de Rentabilidad")
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
        labels = ["Harvester Directo", "Forwarder Directo", "Indirectos"]
        values = [h_total_m, f_total_m, total_shared]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 3: SIMULADOR Y SENSIBILIDAD
# ==========================================
with tab_details:
    st.markdown("### 🧮 Simulador: Ingreso de Producción por Máquina")
    st.caption("Ingresa la producción real estimada para cada equipo. El Sistema Total se calculará según el 'Cuello de Botella' (el equipo más lento).")
    
    # 1. INPUTS SEPARADOS
    col_sim_h, col_sim_f = st.columns(2)
    with col_sim_h:
        st.markdown("🚜 **HARVESTER**")
        m3_h = st.number_input("Volumen Harvester ($m^3$/hr)", value=25.0, step=0.5, key="sim_m3_h", on_change=save_config)
        mr_h = m3_h / conversion_factor if conversion_factor else 0
        st.markdown(f"**= {mr_h:.1f} MR/hr**")
    
    with col_sim_f:
        st.markdown("🚜 **FORWARDER**")
        m3_f = st.number_input("Volumen Forwarder ($m^3$/hr)", value=28.0, step=0.5, key="sim_m3_f", on_change=save_config)
        mr_f = m3_f / conversion_factor if conversion_factor else 0
        st.markdown(f"**= {mr_f:.1f} MR/hr**")
        
    st.divider()

    # 2. CÁLCULOS P&L
    sim_c_h = h_cost_hr / mr_h if mr_h else 0
    sim_m_h = h_income - sim_c_h
    sim_p_h = (sim_m_h / h_income * 100) if h_income else 0
    
    sim_c_f = f_cost_hr / mr_f if mr_f else 0
    sim_m_f = f_income - sim_c_f
    sim_p_f = (sim_m_f / f_income * 100) if f_income else 0
    
    mr_sys = min(mr_h, mr_f)
    sim_c_sys = sys_cost_hr / mr_sys if mr_sys else 0 
    sim_m_sys = sales_price_mr - sim_c_sys
    sim_p_sys = (sim_m_sys / sales_price_mr * 100) if sales_price_mr else 0
    
    # 3. TARJETAS DE RESULTADO
    def result_card(label, prod_val, val_inc, val_cost, val_util, val_pct, bottleneck=False):
        color = "#dcfce7" if val_util > 0 else "#fee2e2"
        text_color = "#166534" if val_util > 0 else "#991b1b"
        border = "2px solid #ef4444" if bottleneck else f"1px solid {text_color}"
        warn = "⚠️ CUELLO DE BOTELLA" if bottleneck else ""
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 8px; border: {border}; text-align: center;">
            <div style="font-weight: bold; color: #374151; margin-bottom: 5px;">{label}</div>
            <div style="font-size: 12px; font-weight:bold; color: #b91c1c;">{warn}</div>
            <div style="font-size: 13px; color: #4b5563; margin-top:5px;">
                Prod: {prod_val:.1f} MR/hr <br>
                Ingreso: ${fmt(val_inc)} <br>
                Costo: -${fmt(val_cost)}
            </div>
            <hr style="border-color: {text_color}; opacity: 0.3; margin: 8px 0;">
            <div style="font-size: 18px; font-weight: 800; color: {text_color};">
                Utilidad: ${fmt(val_util)} <br>
                ({val_pct:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    c_r1, c_r2, c_r3 = st.columns(3)
    
    with c_r1: 
        result_card("HARVESTER", mr_h, h_income, sim_c_h, sim_m_h, sim_p_h, bottleneck=(mr_h < mr_f))
    with c_r2: 
        result_card("FORWARDER", mr_f, f_income, sim_c_f, sim_m_f, sim_p_f, bottleneck=(mr_f < mr_h))
    with c_r3: 
        result_card("SISTEMA REAL", mr_sys, sales_price_mr, sim_c_sys, sim_m_sys, sim_p_sys)

    st.divider()
    st.subheader("📉 Tabla de Sensibilidad General")
    st.dataframe(df_sens, column_config={
        "Prod M3": st.column_config.NumberColumn("M3/hr", format="%d"),
        "Prod MR": st.column_config.NumberColumn("MR/hr", format="%.1f"),
        "Margen H %": st.column_config.ProgressColumn("Margen Harvester", format="%.1f%%", min_value=-0.5, max_value=0.5),
        "Margen F %": st.column_config.ProgressColumn("Margen Forwarder", format="%.1f%%", min_value=-0.5, max_value=0.5),
        "Margen Total %": st.column_config.ProgressColumn("Margen Global", format="%.1f%%", min_value=-0.5, max_value=0.5)
    }, hide_index=True, use_container_width=True)
    
    csv = df_sens.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Reporte (CSV)", csv, "reporte_forestal_pro.csv", "text/csv")
