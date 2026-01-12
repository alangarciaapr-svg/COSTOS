import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(
    page_title="Master Forestal Costing", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f4f6f9;}
    h1, h2, h3 {color: #1e293b; font-family: 'Helvetica Neue', sans-serif;}
    .stMetric {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .metric-container {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 10px;
    }
    .success-box { background-color: #dcfce7; border: 1px solid #86efac; color: #166534; }
    .warning-box { background-color: #fef9c3; border: 1px solid #fde047; color: #854d0e; }
    .danger-box { background-color: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_master_v4_dash.json'

# --- 2. FUNCIONES BACKEND ---

@st.cache_data(ttl=3600) 
def get_uf_api():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data['serie'][0]['valor'], data['serie'][0]['fecha'][:10]
    except:
        return None, None
    return None, None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.int64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float64)): return float(obj)
        return json.JSONEncoder.default(self, obj)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_config():
    keys = ["uf_manual", "fuel_price", "h_days", "h_hours", "f_days", "f_hours", 
            "df_harvester", "df_forwarder", "df_rrhh", "df_flota", "alloc_pct", "sales_price", "target_margin"]
    
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame): state_to_save[k] = val.to_dict('records')
            else: state_to_save[k] = val
                
    with open(CONFIG_FILE, 'w') as f:
        json.dump(state_to_save, f, cls=NumpyEncoder)

def fmt_money(x): 
    return f"$ {x:,.0f}".replace(",", ".")

# --- 3. INICIALIZACIÓN ROBUSTA ---
saved = load_config()

def init_key(key, default_value):
    if key not in st.session_state:
        loaded_val = saved.get(key)
        if loaded_val is not None:
            if isinstance(default_value, pd.DataFrame):
                st.session_state[key] = pd.DataFrame(loaded_val)
            else:
                st.session_state[key] = loaded_val
        else:
            st.session_state[key] = default_value

# Inicialización de variables
init_key('uf_manual', 39755.0)
init_key('fuel_price', 774.0)
init_key('sales_price', 11500.0)
init_key('alloc_pct', 0.5)
init_key('target_margin', 35.0)
init_key('h_days', 28)
init_key('h_hours', 10.0)
init_key('f_days', 28)
init_key('f_hours', 10.0)

# DataFrames
init_key('df_harvester', pd.DataFrame([
    {"Cat": "Fijos", "Ítem": "Arriendo Base", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
    {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Insumos", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frec": 1, "Valor": 450000},
    {"Cat": "Insumos", "Ítem": "Aceite Hidráulico", "Tipo": "$/Mes", "Frec": 1, "Valor": 180000},
    {"Cat": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Ev", "Frec": 600, "Valor": 350000},
    {"Cat": "Mayor", "Ítem": "Overhaul (Amort)", "Tipo": "$/Ev", "Frec": 20000, "Valor": 24000000},
]))

init_key('df_forwarder', pd.DataFrame([
    {"Cat": "Operación", "Ítem": "Arriendo", "Unidad": "$/Mes", "Valor": 8000000},
    {"Cat": "Operación", "Ítem": "Operador", "Unidad": "$/Mes", "Valor": 1900000},
    {"Cat": "Variable", "Ítem": "Petróleo", "Unidad": "Litros/Día", "Valor": 135.0},
    {"Cat": "Mantención", "Ítem": "Mantención Gral", "Unidad": "$/Mes", "Valor": 1500000},
    {"Cat": "Variable", "Ítem": "Neumáticos", "Unidad": "$/Mes", "Valor": 400000},
]))

init_key('df_rrhh', pd.DataFrame([
    {"Cargo": "Jefe de Faena", "Sueldo Líquido": 1800000, "Costo Empresa": 2300000},
    {"Cargo": "Mecánico", "Sueldo Líquido": 1200000, "Costo Empresa": 1600000},
    {"Cargo": "Prevencionista", "Sueldo Líquido": 900000, "Costo Empresa": 1200000},
    {"Cargo": "Estrobero/Cancha", "Sueldo Líquido": 600000, "Costo Empresa": 850000},
]))

init_key('df_flota', pd.DataFrame([
    {"Ítem": "Camioneta 1 (Arriendo)", "Monto": 800000},
    {"Ítem": "Camioneta 2 (Arriendo)", "Monto": 800000},
    {"Ítem": "Combustible Camionetas", "Monto": 600000},
    {"Ítem": "Pensión/Alojamiento", "Monto": 1500000},
    {"Ítem": "Gastos Adm. Central", "Monto": 500000},
]))

# --- 4. CÁLCULOS CENTRALIZADOS ---
def calculate_system_costs(h_df, f_df, rrhh_df, flota_df, days_h, hrs_h, days_f, hrs_f, uf, diesel):
    # Harvester
    total_h = 0
    total_h_hrs = days_h * hrs_h
    for _, row in h_df.iterrows():
        val = row['Valor'] or 0
        tipo = row['Tipo']
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf
        elif tipo == 'Litros/Día': cost = val * days_h * diesel
        elif tipo == '$/Ev': 
            frec = row.get('Frec', 1)
            if frec > 0 and total_h_hrs > 0: cost = (val / frec) * total_h_hrs
        total_h += cost

    # Forwarder
    total_f = 0
    total_f_hrs = days_f * hrs_f
    for _, row in f_df.iterrows():
        val = row['Valor'] or 0
        tipo = row['Unidad']
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'Litros/Día': cost = val * days_f * diesel
        total_f += cost

    # Indirectos
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect, total_h_hrs, total_f_hrs

# --- 5. INTERFAZ: SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2823/2823538.png", width=50)
    st.title("Configuración")
    
    with st.expander("💰 Mercado y Tarifas", expanded=True):
        use_api = st.checkbox("UF Online", value=True)
        uf_api_val, _ = get_uf_api()
        if use_api and uf_api_val:
            st.success(f"UF: ${uf_api_val:,.2f}")
            val_uf = uf_api_val
        else:
            val_uf = st.session_state['uf_manual']
        
        curr_uf = st.number_input("Valor UF", value=float(val_uf), disabled=use_api and uf_api_val is not None)
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()

        curr_fuel = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config, key="fuel_price")
        curr_sales = st.number_input("Tarifa Venta ($/MR)", value=float(st.session_state['sales_price']), on_change=save_config, key="sales_price")

    with st.expander("⚙️ Distribución Costos Indirectos"):
        alloc = st.slider("% Asignado a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()
        st.caption(f"Harvester: {alloc*100:.0f}% | Forwarder: {(1-alloc)*100:.0f}%")

    if st.button("💾 Guardar Cambios"):
        save_config()
        st.toast("Configuración guardada")

# --- 6. INTERFAZ: CUERPO ---
st.title("🌲 Sistema de Costos Forestales Profesional")

# Calculamos costos base antes de mostrar nada
tot_h_dir, tot_f_dir, tot_ind, hrs_h, hrs_f = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    int(st.session_state['h_days']), float(st.session_state['h_hours']), 
    int(st.session_state['f_days']), float(st.session_state['f_hours']), 
    curr_uf, curr_fuel
)
ind_h = tot_ind * st.session_state['alloc_pct']
ind_f = tot_ind * (1 - st.session_state['alloc_pct'])
final_h_mes = tot_h_dir + ind_h
final_f_mes = tot_f_dir + ind_f
cost_hr_sys = (final_h_mes / hrs_h) + (final_f_mes / hrs_f) if hrs_h > 0 else 0

tab_dash, tab_h, tab_f, tab_ind, tab_sim = st.tabs([
    "📊 Dashboard Gerencial", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos", "📈 Matriz Sensibilidad"
])

# --- DASHBOARD GERENCIAL DINÁMICO ---
with tab_dash:
    st.markdown("### 🎯 Simulador de Resultados Mensuales")
    
    # 1. CONTROL DINÁMICO
    col_kpi1, col_kpi2, col_kpi3 = st.columns([1, 1, 2])
    with col_kpi1:
        st.markdown("**1. Variable Crítica**")
        sim_prod_dash = st.number_input("Productividad Real (MR/Hr)", value=22.0, step=0.5, key="dash_prod")
        
    with col_kpi2:
        st.markdown("**2. Tarifa Actual**")
        st.metric("Precio Venta", fmt_money(st.session_state['sales_price']), "por Metro Ruma")

    # Cálculos Dinámicos
    ingreso_mensual_proy = sim_prod_dash * st.session_state['sales_price'] * hrs_h # Asumiendo hrs harvester mandan
    costo_mensual_total = final_h_mes + final_f_mes
    utilidad_mensual = ingreso_mensual_proy - costo_mensual_total
    margen_real = (utilidad_mensual / ingreso_mensual_proy * 100) if ingreso_mensual_proy > 0 else 0
    target = st.session_state['target_margin']
    
    with col_kpi3:
        st.markdown("**3. Resultado Proyectado**")
        delta_color = "normal" if margen_real >= target else "inverse"
        st.metric("Utilidad Mensual Estimada", fmt_money(utilidad_mensual), f"{margen_real:.1f}% Margen Real", delta_color=delta_color)

    st.divider()

    # 2. GRÁFICOS GERENCIALES
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.subheader("Cumplimiento de Meta")
        # Gráfico de Velocímetro (Gauge)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = margen_real,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Margen Operacional (%)"},
            delta = {'reference': target, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 60], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1e3a8a"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 10], 'color': '#fee2e2'},
                    {'range': [10, target], 'color': '#fef9c3'},
                    {'range': [target, 60], 'color': '#dcfce7'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target}}))
        fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if margen_real < 0:
            st.error(f"⚠️ ¡Atención! Estás perdiendo dinero con {sim_prod_dash} MR/hr.")
        elif margen_real < target:
            st.warning(f"⚠️ Margen positivo pero bajo la meta del {target}%.")
        else:
            st.success("✅ Operación saludable.")

    with c_right:
        st.subheader("Cascada de Rentabilidad (Waterfall)")
        # Gráfico Waterfall
        fig_water = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "total"],
            x = ["Ingresos Venta", "Costo Harvester", "Costo Forwarder", "Costo Indirecto", "UTILIDAD NETA"],
            textposition = "outside",
            text = [fmt_money(ingreso_mensual_proy), fmt_money(-tot_h_dir), fmt_money(-tot_f_dir), fmt_money(-tot_ind), fmt_money(utilidad_mensual)],
            y = [ingreso_mensual_proy, -tot_h_dir, -tot_f_dir, -tot_ind, utilidad_mensual],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title = "Estructura de Pérdidas y Ganancias del Mes", showlegend = False, height=400)
        st.plotly_chart(fig_water, use_container_width=True)

# --- OTRAS TABS (MANTENIENDO LO QUE YA FUNCIONABA) ---
with tab_h:
    c1, c2 = st.columns([1, 3])
    with c1:
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 28)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 10.0)), key="h_hours", on_change=save_config)
    with c2:
        st.session_state['df_harvester'] = st.data_editor(
            st.session_state['df_harvester'], use_container_width=True, num_rows="dynamic",
            column_config={"Valor": st.column_config.NumberColumn(format="$ %d"), "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"])}
        )
        save_config()

with tab_f:
    c1, c2 = st.columns([1, 3])
    with c1:
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 28)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 10.0)), key="f_hours", on_change=save_config)
    with c2:
        st.session_state['df_forwarder'] = st.data_editor(
            st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic",
            column_config={"Valor": st.column_config.NumberColumn(format="$ %d"), "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"])}
        )
        save_config()

with tab_ind:
    c_rrhh, c_flota = st.columns(2)
    with c_rrhh:
        st.markdown("### 👷 RRHH Indirecto")
        st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], use_container_width=True, num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c_flota:
        st.markdown("### 🛻 Flota y Gastos Generales")
        st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], use_container_width=True, num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()

with tab_sim:
    st.header("🎯 Calculadora de Tarifas Objetivo")
    st.markdown("Define cuánto quieres ganar y el sistema te dirá cuánto cobrar por Metro Ruma (MR).")

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        target_margin_slide = st.slider("Margen Deseado (%)", 0, 60, int(st.session_state.get('target_margin', 35)), 1)
        st.session_state['target_margin'] = target_margin_slide
        save_config()
    with col_input2:
        prod_estimada = st.number_input("Productividad Estimada para Cálculo (MR/Hr)", value=22.0, step=0.5)

    # Cálculo Tarifas
    cost_hr_h_total = (tot_h_dir + ind_h) / hrs_h if hrs_h > 0 else 0
    cost_hr_f_total = (tot_f_dir + ind_f) / hrs_f if hrs_f > 0 else 0
    cost_hr_sys_total = cost_hr_h_total + cost_hr_f_total

    costo_unit_h = cost_hr_h_total / prod_estimada if prod_estimada > 0 else 0
    costo_unit_f = cost_hr_f_total / prod_estimada if prod_estimada > 0 else 0
    costo_unit_sys = cost_hr_sys_total / prod_estimada if prod_estimada > 0 else 0

    margin_factor = 1 - (target_margin_slide / 100.0)
    if margin_factor <= 0: margin_factor = 0.01

    precio_req_h = costo_unit_h / margin_factor
    precio_req_f = costo_unit_f / margin_factor
    precio_req_sys = costo_unit_sys / margin_factor

    st.divider()
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.markdown('<div class="highlight-box">Harvester<br><span class="big-number">{}</span><br>per MR</div>'.format(fmt_money(precio_req_h)), unsafe_allow_html=True)
    with col_res2:
        st.markdown('<div class="highlight-box">Forwarder<br><span class="big-number">{}</span><br>per MR</div>'.format(fmt_money(precio_req_f)), unsafe_allow_html=True)
    with col_res3:
        st.markdown('<div class="highlight-box" style="background-color:#dbeafe; border-color:#93c5fd;">SISTEMA TOTAL<br><span class="big-number" style="color:#1e40af;">{}</span><br>per MR</div>'.format(fmt_money(precio_req_sys)), unsafe_allow_html=True)

    st.divider()
    
    # Heatmap
    st.subheader("📉 Matriz de Sensibilidad")
    base_price = st.session_state['sales_price']
    rango_precios = np.linspace(base_price * 0.8, base_price * 1.2, 10)
    rango_prod = np.linspace(10, 40, 10)
    
    z_data = []
    for p_prod in rango_prod:
        row = []
        for p_price in rango_precios:
            ing = p_prod * p_price
            mgn = ((ing - cost_hr_sys_total) / ing * 100) if ing > 0 else 0
            row.append(mgn)
        z_data.append(row)
        
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data, x=[f"${p:,.0f}" for p in rango_precios], y=[f"{pr:.1f}" for pr in rango_prod],
        colorscale='RdYlGn', zmin=0, zmax=50, texttemplate="%{z:.0f}%"
    ))
    fig_heat.update_layout(title="Margen (%) según Precio y Productividad", xaxis_title="Precio ($/MR)", yaxis_title="Prod (MR/Hr)", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
