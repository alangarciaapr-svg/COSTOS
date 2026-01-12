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
    .highlight-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #166534;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .big-number {
        font-size: 1.8em;
        font-weight: 800;
        color: #166534;
    }
    .profit-text {
        color: #15803d;
        font-weight: bold;
        font-size: 1.1em;
    }
    .cost-text {
        color: #64748b;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_master_v7_profit.json'

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
            "df_harvester", "df_forwarder", "df_rrhh", "df_flota", 
            "alloc_pct", "sales_price", "target_margin_h", "target_margin_f", "conv_factor"]
    
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

# --- 3. INICIALIZACIÓN ---
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

init_key('uf_manual', 39755.0)
init_key('fuel_price', 774.0)
init_key('sales_price', 11500.0)
init_key('alloc_pct', 0.5)
init_key('target_margin_h', 35.0)
init_key('target_margin_f', 35.0)
init_key('conv_factor', 2.44)
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

    total_f = 0
    total_f_hrs = days_f * hrs_f
    for _, row in f_df.iterrows():
        val = row['Valor'] or 0
        tipo = row['Unidad']
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'Litros/Día': cost = val * days_f * diesel
        total_f += cost

    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()
    return total_h, total_f, total_indirect, total_h_hrs, total_f_hrs

# --- 5. INTERFAZ SIDEBAR ---
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

    with st.expander("📏 Conversión Volumen", expanded=True):
        curr_factor = st.number_input("Factor (m³/MR)", value=float(st.session_state.get('conv_factor', 2.44)), step=0.01, format="%.2f", key="conv_factor_input")
        if curr_factor != st.session_state['conv_factor']:
            st.session_state['conv_factor'] = curr_factor
            save_config()

    with st.expander("⚙️ Distribución Costos"):
        alloc = st.slider("% Asignado a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()
        st.caption(f"Harvester: {alloc*100:.0f}% | Forwarder: {(1-alloc)*100:.0f}%")

    if st.button("💾 Guardar Cambios"):
        save_config()
        st.toast("Configuración guardada")

# --- 6. CÁLCULOS BASE ---
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
cost_mensual_sistema = final_h_mes + final_f_mes
cost_hr_sys = (final_h_mes / hrs_h) + (final_f_mes / hrs_f) if hrs_h > 0 else 0

# --- TABS ---
st.title("🌲 Sistema de Costos Forestales")
tab_dash, tab_h, tab_f, tab_ind, tab_sim = st.tabs(["📊 Cierre Mensual", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos", "📈 Matriz & Tarifas"])

# --- DASHBOARD ---
with tab_dash:
    st.header("📊 Simulador de Cierre Mensual")
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        st.markdown("#### 🚜 Harvester")
        prod_h_m3 = st.number_input("Producción H (m³)", value=5000.0, step=100.0)
        st.metric("Equivalente MR", f"{prod_h_m3 / st.session_state['conv_factor']:,.1f} MR")
    with c_in2:
        st.markdown("#### 🚜 Forwarder")
        prod_f_m3 = st.number_input("Producción F (m³)", value=5000.0, step=100.0)
        prod_f_mr = prod_f_m3 / st.session_state['conv_factor']
        st.metric("Equivalente MR", f"{prod_f_mr:,.1f} MR")
    with c_in3:
        st.markdown("#### 💰 Facturación")
        st.info(f"Factor: **{st.session_state['conv_factor']} m³/MR**")

    st.divider()
    ingresos_reales = prod_f_mr * st.session_state['sales_price']
    utilidad = ingresos_reales - cost_mensual_sistema
    margen_pct = (utilidad / ingresos_reales * 100) if ingresos_reales > 0 else 0
    target_avg = (st.session_state.get('target_margin_h', 35.0) + st.session_state.get('target_margin_f', 35.0)) / 2

    c1, c2, c3 = st.columns(3)
    c1.metric("Facturación Estimada", fmt_money(ingresos_reales))
    c2.metric("Costo Total Mes", fmt_money(cost_mensual_sistema))
    c3.metric("Utilidad Neta", fmt_money(utilidad), f"{margen_pct:.1f}% Margen", delta_color="normal" if margen_pct >= target_avg else "inverse")

# --- INPUTS COSTOS ---
with tab_h:
    c1, c2 = st.columns([1, 3])
    with c1:
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 28)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 10.0)), key="h_hours", on_change=save_config)
    with c2:
        st.session_state['df_harvester'] = st.data_editor(st.session_state['df_harvester'], use_container_width=True, num_rows="dynamic", column_config={"Valor": st.column_config.NumberColumn(format="$ %d"), "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"])})
        save_config()

with tab_f:
    c1, c2 = st.columns([1, 3])
    with c1:
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 28)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 10.0)), key="f_hours", on_change=save_config)
    with c2:
        st.session_state['df_forwarder'] = st.data_editor(st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic", column_config={"Valor": st.column_config.NumberColumn(format="$ %d"), "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"])})
        save_config()

with tab_ind:
    c1, c2 = st.columns(2)
    with c1: st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], use_container_width=True, num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c2: st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], use_container_width=True, num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()

# --- TAB SIMULACIÓN: UTILIDAD POR MÁQUINA ---
with tab_sim:
    st.header("🎯 Análisis de Utilidad y Tarifas por Máquina")
    
    # 1. Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        margin_h = st.slider("Margen Objetivo Harvester (%)", 0, 60, int(st.session_state.get('target_margin_h', 35)))
        st.session_state['target_margin_h'] = margin_h
    with c2:
        margin_f = st.slider("Margen Objetivo Forwarder (%)", 0, 60, int(st.session_state.get('target_margin_f', 35)))
        st.session_state['target_margin_f'] = margin_f
    with c3:
        prod_sim = st.number_input("Prod. Estimada (MR/Hr)", value=22.0, step=0.5)

    # 2. Cálculos Unitarios
    # Costo por Hora
    cost_h_hr = (tot_h_dir + ind_h) / hrs_h if hrs_h > 0 else 0
    cost_f_hr = (tot_f_dir + ind_f) / hrs_f if hrs_f > 0 else 0
    
    # Costo por MR
    cost_unit_h = cost_h_hr / prod_sim if prod_sim > 0 else 0
    cost_unit_f = cost_f_hr / prod_sim if prod_sim > 0 else 0
    
    # Precio Sugerido
    price_h = cost_unit_h / (1 - margin_h/100) if margin_h < 100 else 0
    price_f = cost_unit_f / (1 - margin_f/100) if margin_f < 100 else 0
    
    # Utilidad por MR
    profit_h = price_h - cost_unit_h
    profit_f = price_f - cost_unit_f
    
    # Totales Sistema
    price_sys = price_h + price_f
    cost_unit_sys = cost_unit_h + cost_unit_f
    profit_sys = profit_h + profit_f
    margin_sys_pct = (profit_sys / price_sys * 100) if price_sys > 0 else 0

    st.divider()
    
    # 3. Visualización Cards
    col_h, col_f, col_s = st.columns(3)
    
    with col_h:
        st.markdown(f"""
        <div class="highlight-box">
            <h3>🚜 Harvester</h3>
            <div class="big-number">{fmt_money(profit_h)}</div>
            <div class="profit-text">Utilidad por MR</div>
            <hr>
            <div class="cost-text">Tarifa Sugerida: <b>{fmt_money(price_h)}</b></div>
            <div class="cost-text">Costo Unitario: {fmt_money(cost_unit_h)}</div>
            <div class="cost-text">Margen: {margin_h}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_f:
        st.markdown(f"""
        <div class="highlight-box">
            <h3>🚜 Forwarder</h3>
            <div class="big-number">{fmt_money(profit_f)}</div>
            <div class="profit-text">Utilidad por MR</div>
            <hr>
            <div class="cost-text">Tarifa Sugerida: <b>{fmt_money(price_f)}</b></div>
            <div class="cost-text">Costo Unitario: {fmt_money(cost_unit_f)}</div>
            <div class="cost-text">Margen: {margin_f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s:
        st.markdown(f"""
        <div class="highlight-box" style="border-left-color: #1e40af;">
            <h3 style="color:#1e3a8a">🌲 SISTEMA TOTAL</h3>
            <div class="big-number" style="color:#1e3a8a">{fmt_money(profit_sys)}</div>
            <div class="profit-text" style="color:#1d4ed8">Utilidad Total por MR</div>
            <hr>
            <div class="cost-text">Tarifa Total: <b>{fmt_money(price_sys)}</b></div>
            <div class="cost-text">Costo Total: {fmt_money(cost_unit_sys)}</div>
            <div class="cost-text">Margen Promedio: {margin_sys_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 4. Tabla Detallada
    st.subheader("📋 Estado de Resultados Unitario Estimado")
    df_detail = pd.DataFrame({
        "Concepto": ["Harvester", "Forwarder", "TOTAL SISTEMA"],
        "Costo Unitario ($/MR)": [fmt_money(cost_unit_h), fmt_money(cost_unit_f), fmt_money(cost_unit_sys)],
        "Tarifa Sugerida ($/MR)": [fmt_money(price_h), fmt_money(price_f), fmt_money(price_sys)],
        "Utilidad ($/MR)": [fmt_money(profit_h), fmt_money(profit_f), fmt_money(profit_sys)],
        "Margen %": [f"{margin_h}%", f"{margin_f}%", f"{margin_sys_pct:.1f}%"]
    })
    st.dataframe(df_detail, use_container_width=True, hide_index=True)

    # 5. Gráfico Barras Apiladas
    st.subheader("📊 Composición de la Tarifa")
    df_chart = pd.DataFrame({
        "Máquina": ["Harvester", "Harvester", "Forwarder", "Forwarder"],
        "Tipo": ["Costo", "Utilidad", "Costo", "Utilidad"],
        "Valor": [cost_unit_h, profit_h, cost_unit_f, profit_f]
    })
    
    fig_bar = px.bar(df_chart, x="Máquina", y="Valor", color="Tipo", title="Estructura de Tarifa por Máquina",
                     color_discrete_map={"Costo": "#94a3b8", "Utilidad": "#22c55e"}, text_auto='.2s')
    st.plotly_chart(fig_bar, use_container_width=True)
