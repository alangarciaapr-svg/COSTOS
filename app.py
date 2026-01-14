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
    page_title="Forestal Costing Master", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f1f5f9;}
    h1, h2, h3 {color: #0f172a; font-family: 'Segoe UI', sans-serif;}
    
    .stMetric {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #16a34a;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .big-number { font-size: 1.8em; font-weight: 800; color: #15803d; }
    .label-text { font-size: 0.9em; color: #64748b; font-weight: 600; text-transform: uppercase; }
    
    .price-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    /* Ocultar índices */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v22_excel.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
    df = df.fillna(0)
    total = 0
    total_hours = days * hours
    for _, row in df.iterrows():
        val = float(row.get('Valor', 0))
        tipo = row.get('Tipo') if machine_type == 'H' else row.get('Unidad')
        if not tipo: tipo = '$/Mes'
        frec = float(row.get('Frec', 1))
        
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf
        elif tipo == 'Litros/Día': cost = val * days * diesel
        elif tipo == '$/Ev': 
            if frec > 0 and total_hours > 0: cost = (val / frec) * total_hours
        total += cost
    return total

def calculate_system_costs(h_df, f_df, rrhh_df, flota_df, h_days, h_hours, f_days, f_hours, uf, diesel):
    total_h = calculate_single_machine_monthly_cost(h_df, h_days, h_hours, uf, diesel, 'H')
    total_f = calculate_single_machine_monthly_cost(f_df, f_days, f_hours, uf, diesel, 'F')
    
    rrhh_df = rrhh_df.fillna(0)
    flota_df = flota_df.fillna(0)
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect

# --- FUNCIÓN GENERADORA DE EXCEL ---
def to_excel(state):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # 1. Resumen de Parámetros
    params = [
        {"Parámetro": "Valor UF", "Valor": state['uf_manual']},
        {"Parámetro": "Precio Diesel", "Valor": state['fuel_price']},
        {"Parámetro": "Factor Conversión (m3/MR)", "Valor": state['conv_factor']},
        {"Parámetro": "Días Harvester", "Valor": state['h_days']},
        {"Parámetro": "Horas Harvester", "Valor": state['h_hours']},
        {"Parámetro": "Días Forwarder", "Valor": state['f_days']},
        {"Parámetro": "Horas Forwarder", "Valor": state['f_hours']},
        {"Parámetro": "Tarifa Actual H", "Valor": state['price_h']},
        {"Parámetro": "Tarifa Actual F", "Valor": state['price_f']},
        {"Parámetro": "% Asig. Indirectos H", "Valor": state['alloc_pct']}
    ]
    pd.DataFrame(params).to_excel(writer, sheet_name='Parametros', index=False)
    
    # 2. Tablas de Costos
    state['df_harvester'].to_excel(writer, sheet_name='Costos_Harvester', index=False)
    state['df_forwarder'].to_excel(writer, sheet_name='Costos_Forwarder', index=False)
    state['df_rrhh'].to_excel(writer, sheet_name='RRHH', index=False)
    state['df_flota'].to_excel(writer, sheet_name='Flota_Indirectos', index=False)
    
    writer.close()
    processed_data = output.getvalue()
    return processed_data

@st.cache_data(ttl=3600) 
def get_uf_api():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data['serie'][0]['valor'], data['serie'][0]['fecha'][:10]
    except: return None, None
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
            "alloc_pct", "price_h", "price_f", "conv_factor"]
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame): state_to_save[k] = val.to_dict('records')
            else: state_to_save[k] = val
    with open(CONFIG_FILE, 'w') as f: json.dump(state_to_save, f, cls=NumpyEncoder)

# --- 3. INICIALIZACIÓN ---
saved = load_config()

def init_key(key, default_value):
    if key not in st.session_state:
        loaded_val = saved.get(key)
        if loaded_val is not None:
            if isinstance(default_value, pd.DataFrame): st.session_state[key] = pd.DataFrame(loaded_val)
            else: st.session_state[key] = loaded_val
        else: st.session_state[key] = default_value

init_key('uf_manual', 39755.0)
init_key('fuel_price', 774.0)
init_key('price_h', 6500.0)
init_key('price_f', 5000.0)
init_key('alloc_pct', 0.5)
init_key('conv_factor', 2.44)
init_key('h_days', 28)
init_key('h_hours', 10.0)
init_key('f_days', 28)
init_key('f_hours', 10.0)

# DataFrames
init_key('df_harvester', pd.DataFrame([
    {"Cat": "Fijos", "Ítem": "Arriendo Base", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
    {"Cat": "Variable", "Ítem": "Petróleo", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Ev", "Frec": 600, "Valor": 350000},
]))
init_key('df_forwarder', pd.DataFrame([
    {"Cat": "Operación", "Ítem": "Arriendo", "Unidad": "$/Mes", "Valor": 8000000},
    {"Cat": "Variable", "Ítem": "Petróleo", "Unidad": "Litros/Día", "Valor": 135.0},
]))
init_key('df_rrhh', pd.DataFrame([{"Cargo": "Jefe Faena", "Costo Empresa": 2300000}]))
init_key('df_flota', pd.DataFrame([{"Ítem": "Camionetas", "Monto": 1600000}]))

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    
    # BOTÓN EXPORTAR EXCEL
    excel_data = to_excel(st.session_state)
    st.download_button(
        label="📥 Descargar Reporte Excel",
        data=excel_data,
        file_name='Reporte_Costos_Forestal.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type="primary"
    )
    
    if st.button("♻️ Reset App"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    with st.expander("🕒 Jornada Laboral", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state['h_days'] = c1.number_input("Días H", value=int(st.session_state['h_days']), on_change=save_config)
        st.session_state['h_hours'] = c2.number_input("Horas H", value=float(st.session_state['h_hours']), on_change=save_config)
        
        c3, c4 = st.columns(2)
        st.session_state['f_days'] = c3.number_input("Días F", value=int(st.session_state['f_days']), on_change=save_config)
        st.session_state['f_hours'] = c4.number_input("Horas F", value=float(st.session_state['f_hours']), on_change=save_config)

    with st.expander("💰 Mercado", expanded=True):
        use_api = st.checkbox("UF Auto", value=True)
        uf_api, _ = get_uf_api()
        val_uf = uf_api if (use_api and uf_api) else st.session_state['uf_manual']
        
        curr_uf = st.number_input("Valor UF", value=float(val_uf), disabled=bool(use_api and uf_api))
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()
        st.session_state['fuel_price'] = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config)

    with st.expander("💵 Tarifas Actuales ($/MR)"):
        st.session_state['price_h'] = st.number_input("Tarifa Harvester", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Tarifa Forwarder", value=float(st.session_state['price_f']), on_change=save_config)

    with st.expander("📏 Conversión y Distribución"):
        st.session_state['conv_factor'] = st.number_input("Factor m³/MR", value=float(st.session_state['conv_factor']), step=0.01, on_change=save_config)
        alloc = st.slider("% Indirectos a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()

# --- 5. MOTOR DE CÁLCULO ---
h_dias = int(st.session_state['h_days'])
h_horas = float(st.session_state['h_hours'])
f_dias = int(st.session_state['f_days'])
f_horas = float(st.session_state['f_hours'])

tot_h_dir, tot_f_dir, tot_ind = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    h_dias, h_horas, f_dias, f_horas, 
    curr_uf, st.session_state['fuel_price']
)

# Costo Mensual Total por Máquina (Directo + Asignación Indirecta)
cost_h_total_mes = tot_h_dir + (tot_ind * st.session_state['alloc_pct'])
cost_f_total_mes = tot_f_dir + (tot_ind * (1 - st.session_state['alloc_pct']))

# --- 6. INTERFAZ PRINCIPAL ---
st.title("🌲 Forestal Costing Master")

tab_dash, tab_strat, tab_h, tab_f, tab_ind = st.tabs([
    "📊 Resultado Operacional", 
    "🎯 Estrategia de Precios (30-35%)", 
    "🚜 Harvester", 
    "🚜 Forwarder", 
    "👷 Indirectos"
])

# --- TAB 1: DASHBOARD ---
with tab_dash:
    st.markdown("### 1. Productividad Real en Faena")
    c1, c2 = st.columns(2)
    
    with c1:
        prod_h_m3_hr = st.number_input("Producción H (m³/hr)", value=25.0, step=0.5, key="p_h_real")
        mr_h_hr = prod_h_m3_hr / st.session_state['conv_factor']
        st.info(f"Harvester: {mr_h_hr:,.1f} MR/Hr")
        
    with c2:
        prod_f_m3_hr = st.number_input("Producción F (m³/hr)", value=28.0, step=0.5, key="p_f_real")
        mr_f_hr = prod_f_m3_hr / st.session_state['conv_factor']
        st.info(f"Forwarder: {mr_f_hr:,.1f} MR/Hr")

    st.divider()
    
    # Cálculos Reales
    inc_h_hr = mr_h_hr * st.session_state['price_h']
    inc_h_day = inc_h_hr * h_horas
    inc_h_mes = inc_h_day * h_dias
    cost_h_mes = cost_h_total_mes
    prof_h_mes = inc_h_mes - cost_h_mes
    
    inc_f_hr = mr_f_hr * st.session_state['price_f']
    inc_f_day = inc_f_hr * f_horas
    inc_f_mes = inc_f_day * f_dias
    cost_f_mes = cost_f_total_mes
    prof_f_mes = inc_f_mes - cost_f_mes

    col_h, col_f = st.columns(2)
    with col_h:
        st.markdown(f"#### 🚜 HARVESTER (Utilidad Mes: {fmt_money(prof_h_mes)})")
        df_h = pd.DataFrame({
            "Periodo": ["Hora", "Día", "Mes"],
            "Generado": [fmt_money(inc_h_hr), fmt_money(inc_h_day), fmt_money(inc_h_mes)],
            "Costo Total": [fmt_money(cost_h_mes/(h_dias*h_horas)), fmt_money(cost_h_mes/h_dias), fmt_money(cost_h_mes)],
            "Ganancia": [fmt_money(inc_h_hr - (cost_h_mes/(h_dias*h_horas))), fmt_money(inc_h_day - (cost_h_mes/h_dias)), fmt_money(prof_h_mes)]
        })
        st.dataframe(df_h, use_container_width=True, hide_index=True)

    with col_f:
        st.markdown(f"#### 🚜 FORWARDER (Utilidad Mes: {fmt_money(prof_f_mes)})")
        df_f = pd.DataFrame({
            "Periodo": ["Hora", "Día", "Mes"],
            "Generado": [fmt_money(inc_f_hr), fmt_money(inc_f_day), fmt_money(inc_f_mes)],
            "Costo Total": [fmt_money(cost_f_mes/(f_dias*f_horas)), fmt_money(cost_f_mes/f_dias), fmt_money(cost_f_mes)],
            "Ganancia": [fmt_money(inc_f_hr - (cost_f_mes/(f_dias*f_horas))), fmt_money(inc_f_day - (cost_f_mes/f_dias)), fmt_money(prof_f_mes)]
        })
        st.dataframe(df_f, use_container_width=True, hide_index=True)

    st.success(f"💰 **UTILIDAD FINAL SISTEMA: {fmt_money(prof_h_mes + prof_f_mes)}**")

# --- TAB 2: ESTRATEGIA DE PRECIOS (NUEVA PESTAÑA PEDIDA) ---
with tab_strat:
    st.header("🎯 Simulador de Tarifas Objetivo (30% - 35%)")
    st.markdown("Calcula cuánto cobrar por Metro Ruma para asegurar tu margen de ganancia.")
    
    # Input de Productividad Estimada para cotización
    prod_cotiza = st.number_input("Productividad Estimada para Cotización (m³/hr)", value=25.0, step=0.5, key="p_cotiza")
    mr_cotiza = prod_cotiza / st.session_state['conv_factor']
    
    st.info(f"Cálculos basados en una productividad de **{mr_cotiza:,.1f} MR/Hora**")
    
    # 1. Costo Unitario Real ($/MR)
    # Costo Hora Total / Producción Hora MR
    cost_h_hr_sys = cost_h_total_mes / (h_dias * h_horas)
    cost_f_hr_sys = cost_f_total_mes / (f_dias * f_horas)
    
    cost_unit_h = cost_h_hr_sys / mr_cotiza if mr_cotiza > 0 else 0
    cost_unit_f = cost_f_hr_sys / mr_cotiza if mr_cotiza > 0 else 0
    
    # 2. Cálculo de Precios para 30% y 35%
    # Harvester
    p_h_30 = calc_price(cost_unit_h, 30)
    p_h_35 = calc_price(cost_unit_h, 35)
    # Forwarder
    p_f_30 = calc_price(cost_unit_f, 30)
    p_f_35 = calc_price(cost_unit_f, 35)
    
    # 3. Tabla Comparativa
    st.subheader("📋 Matriz de Precios Sugeridos ($/MR)")
    
    df_prices = pd.DataFrame({
        "Concepto": ["Costo Unitario Real", "Tarifa para ganar 30%", "Tarifa para ganar 35%"],
        "Harvester": [fmt_money(cost_unit_h), fmt_money(p_h_30), fmt_money(p_h_35)],
        "Forwarder": [fmt_money(cost_unit_f), fmt_money(p_f_30), fmt_money(p_f_35)],
        "SISTEMA TOTAL": [fmt_money(cost_unit_h+cost_unit_f), fmt_money(p_h_30+p_f_30), fmt_money(p_h_35+p_f_35)]
    })
    st.dataframe(df_prices, use_container_width=True, hide_index=True)
    
    st.caption("Nota: El Costo Unitario incluye todos los costos directos e indirectos prorrateados.")

# --- TABS EDICIÓN ---
with tab_h:
    st.header("Costos Harvester")
    edited_h = st.data_editor(st.session_state['df_harvester'], num_rows="dynamic", use_container_width=True,
                              column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_harvester'] = edited_h
    save_config()
    live_h = calculate_single_machine_monthly_cost(edited_h, h_dias, h_horas, curr_uf, st.session_state['fuel_price'], 'H')
    st.info(f"Costo Directo Mensual: {fmt_money(live_h)}")

with tab_f:
    st.header("Costos Forwarder")
    edited_f = st.data_editor(st.session_state['df_forwarder'], num_rows="dynamic", use_container_width=True,
                              column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_forwarder'] = edited_f
    save_config()
    live_f = calculate_single_machine_monthly_cost(edited_f, f_dias, f_horas, curr_uf, st.session_state['fuel_price'], 'F')
    st.info(f"Costo Directo Mensual: {fmt_money(live_f)}")

with tab_ind:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RRHH")
        edited_rrhh = st.data_editor(st.session_state['df_rrhh'], num_rows="dynamic", use_container_width=True, column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_rrhh'] = edited_rrhh
    with c2:
        st.subheader("Flota")
        edited_flota = st.data_editor(st.session_state['df_flota'], num_rows="dynamic", use_container_width=True, column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_flota'] = edited_flota
    save_config()
    st.success(f"Total Indirectos: {fmt_money(st.session_state['df_rrhh']['Costo Empresa'].sum() + st.session_state['df_flota']['Monto'].sum())}")
