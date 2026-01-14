import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

# --- 1. CONFIGURACIÓN Y ESTILO PRO ---
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
    
    /* Estilos de Tarjetas */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
    }
    .card-title { font-size: 0.85em; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;}
    .card-value { font-size: 1.6em; font-weight: 800; color: #0f172a; margin: 5px 0;}
    .card-sub { font-size: 0.9em; color: #94a3b8; }
    
    /* Colores de Profit */
    .pos { color: #16a34a !important; }
    .neg { color: #dc2626 !important; }
    
    /* Headers de Sección */
    .section-header {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 6px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Tabla limpia */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v21_full_power.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
    """Calcula costo mensual real basado en los parámetros de turno de la máquina"""
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
    # Costos Directos Calculados con sus propios días y horas
    total_h = calculate_single_machine_monthly_cost(h_df, h_days, h_hours, uf, diesel, 'H')
    total_f = calculate_single_machine_monthly_cost(f_df, f_days, f_hours, uf, diesel, 'F')
    
    # Indirectos Fijos
    rrhh_df = rrhh_df.fillna(0)
    flota_df = flota_df.fillna(0)
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect

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

# --- 4. SIDEBAR (CONFIGURACIÓN) ---
with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    if st.button("♻️ Reset App", type="primary"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    # BLOQUE CRITICO: JORNADA LABORAL
    st.markdown("### 🕒 Turnos y Jornada")
    with st.container():
        st.markdown("**🚜 Harvester**")
        c1, c2 = st.columns(2)
        st.session_state['h_days'] = c1.number_input("Días H", value=int(st.session_state['h_days']), key="hd", on_change=save_config)
        st.session_state['h_hours'] = c2.number_input("Horas H", value=float(st.session_state['h_hours']), key="hh", on_change=save_config)
        
        st.markdown("**🚜 Forwarder**")
        c3, c4 = st.columns(2)
        st.session_state['f_days'] = c3.number_input("Días F", value=int(st.session_state['f_days']), key="fd", on_change=save_config)
        st.session_state['f_hours'] = c4.number_input("Horas F", value=float(st.session_state['f_hours']), key="fh", on_change=save_config)

    with st.expander("💰 Mercado", expanded=False):
        use_api = st.checkbox("UF Auto", value=True)
        uf_api, _ = get_uf_api()
        val_uf = uf_api if (use_api and uf_api) else st.session_state['uf_manual']
        curr_uf = st.number_input("Valor UF", value=float(val_uf), disabled=bool(use_api and uf_api))
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()
        st.session_state['fuel_price'] = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config)

    with st.expander("💵 Tarifas ($/MR)", expanded=False):
        st.session_state['price_h'] = st.number_input("Tarifa Harvester", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Tarifa Forwarder", value=float(st.session_state['price_f']), on_change=save_config)

    with st.expander("📏 Conversión y Distribución", expanded=False):
        st.session_state['conv_factor'] = st.number_input("Factor m³/MR", value=float(st.session_state['conv_factor']), step=0.01, on_change=save_config)
        alloc = st.slider("% Indirectos a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()

# --- 5. MOTOR DE CÁLCULO ---
# Recuperar variables para cálculo limpio
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
st.title("🌲 Dashboard Gerencial de Costos")

tab_dash, tab_h, tab_f, tab_ind = st.tabs(["📊 Resultado Operacional", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos"])

# --- TAB DASHBOARD ---
with tab_dash:
    # 1. INPUTS
    st.markdown('<div class="section-header">1. PRODUCTIVIDAD EN TERRENO (m³ Sólidos / Hora)</div>', unsafe_allow_html=True)
    c_in1, c_in2 = st.columns(2)
    
    with c_in1:
        prod_h_m3 = st.number_input("Productividad Harvester (m³/hr)", value=25.0, step=0.5, key="p_h_in")
        mr_h_hr = prod_h_m3 / st.session_state['conv_factor']
        st.caption(f"Equivale a: {mr_h_hr:,.1f} MR/Hr")
        
    with c_in2:
        prod_f_m3 = st.number_input("Productividad Forwarder (m³/hr)", value=28.0, step=0.5, key="p_f_in")
        mr_f_hr = prod_f_m3 / st.session_state['conv_factor']
        st.caption(f"Equivale a: {mr_f_hr:,.1f} MR/Hr")

    # 2. CÁLCULOS DETALLADOS (HORA / DÍA / MES)
    
    # --- Harvester ---
    rev_h_hr = mr_h_hr * st.session_state['price_h']
    rev_h_day = rev_h_hr * h_horas
    rev_h_mes = rev_h_day * h_dias
    
    cost_h_mes = cost_h_total_mes
    cost_h_day = cost_h_mes / h_dias if h_dias > 0 else 0
    cost_h_hr = cost_h_day / h_horas if h_horas > 0 else 0
    
    prof_h_hr = rev_h_hr - cost_h_hr
    prof_h_day = rev_h_day - cost_h_day
    prof_h_mes = rev_h_mes - cost_h_mes
    margin_h = (prof_h_mes / rev_h_mes * 100) if rev_h_mes > 0 else 0

    # --- Forwarder ---
    rev_f_hr = mr_f_hr * st.session_state['price_f']
    rev_f_day = rev_f_hr * f_horas
    rev_f_mes = rev_f_day * f_dias
    
    cost_f_mes = cost_f_total_mes
    cost_f_day = cost_f_mes / f_dias if f_dias > 0 else 0
    cost_f_hr = cost_f_day / f_horas if f_horas > 0 else 0
    
    prof_f_hr = rev_f_hr - cost_f_hr
    prof_f_day = rev_f_day - cost_f_day
    prof_f_mes = rev_f_mes - cost_f_mes
    margin_f = (prof_f_mes / rev_f_mes * 100) if rev_f_mes > 0 else 0

    # 3. VISUALIZACIÓN
    st.markdown('<div class="section-header">2. ANÁLISIS DE RENTABILIDAD DETALLADO</div>', unsafe_allow_html=True)
    
    # Tablas de Resultados
    col_h, col_f = st.columns(2)
    
    with col_h:
        st.markdown(f"#### 🚜 HARVESTER ({margin_h:.1f}% Margen)")
        df_h_res = pd.DataFrame({
            "Periodo": ["Hora", "Día", "Mes"],
            "Generado ($)": [fmt_money(rev_h_hr), fmt_money(rev_h_day), fmt_money(rev_h_mes)],
            "Costo Total ($)": [fmt_money(cost_h_hr), fmt_money(cost_h_day), fmt_money(cost_h_mes)],
            "Ganancia ($)": [fmt_money(prof_h_hr), fmt_money(prof_h_day), fmt_money(prof_h_mes)]
        })
        st.dataframe(df_h_res, use_container_width=True, hide_index=True)
        
        # Mini gráfico
        fig_h = go.Figure(go.Indicator(
            mode = "gauge+number", value = margin_h, 
            title = {'text': "Margen %"},
            gauge = {'axis': {'range': [None, 60]}, 'bar': {'color': "#eab308"}}
        ))
        fig_h.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_f:
        st.markdown(f"#### 🚜 FORWARDER ({margin_f:.1f}% Margen)")
        df_f_res = pd.DataFrame({
            "Periodo": ["Hora", "Día", "Mes"],
            "Generado ($)": [fmt_money(rev_f_hr), fmt_money(rev_f_day), fmt_money(rev_f_mes)],
            "Costo Total ($)": [fmt_money(cost_f_hr), fmt_money(cost_f_day), fmt_money(cost_f_mes)],
            "Ganancia ($)": [fmt_money(prof_f_hr), fmt_money(prof_f_day), fmt_money(prof_f_mes)]
        })
        st.dataframe(df_f_res, use_container_width=True, hide_index=True)
        
        # Mini gráfico
        fig_f = go.Figure(go.Indicator(
            mode = "gauge+number", value = margin_f, 
            title = {'text': "Margen %"},
            gauge = {'axis': {'range': [None, 60]}, 'bar': {'color': "#22c55e"}}
        ))
        fig_f.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_f, use_container_width=True)

    # 4. CONSOLIDADO
    st.divider()
    total_prof = prof_h_mes + prof_f_mes
    total_rev = rev_h_mes + rev_f_mes
    total_cost = cost_h_mes + cost_f_mes
    total_margin = (total_prof / total_rev * 100) if total_rev > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Venta Total Sistema", fmt_money(total_rev))
    c2.metric("Costo Total Sistema", fmt_money(total_cost))
    c3.metric("Ganancia Neta", fmt_money(total_prof), delta_color="normal")
    c4.metric("Margen Global", f"{total_margin:.1f}%")

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
        st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c2:
        st.subheader("Flota")
        st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()
    st.success(f"Total Indirectos: {fmt_money(st.session_state['df_rrhh']['Costo Empresa'].sum() + st.session_state['df_flota']['Monto'].sum())}")
