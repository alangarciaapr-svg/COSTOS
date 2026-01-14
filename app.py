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
    page_title="Forestal Costing Pro", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    h1, h2, h3 {color: #0f172a; font-family: 'Segoe UI', sans-serif;}
    .stMetric {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .highlight-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #16a34a;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .kpi-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .kpi-title { font-size: 0.9em; color: #64748b; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; border-bottom: 1px solid #f1f5f9; padding-bottom: 5px;}
    .kpi-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 0.95em; }
    .val-inc { font-weight: 600; color: #0f172a; }
    .val-cost { font-weight: 600; color: #ef4444; }
    .val-prof { font-weight: 700; color: #16a34a; font-size: 1.1em; }
    .val-loss { font-weight: 700; color: #dc2626; font-size: 1.1em; }
    
    .mr-badge { background-color: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v20_shifts_fixed.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
    """Calcula costo mensual usando los días y horas ESPECÍFICOS de la máquina"""
    df = df.fillna(0)
    total = 0
    total_hours = days * hours
    
    for _, row in df.iterrows():
        val = float(row.get('Valor', 0))
        tipo = row.get('Tipo') if machine_type == 'H' else row.get('Unidad')
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
    # Calcular usando parámetros independientes
    total_h = calculate_single_machine_monthly_cost(h_df, h_days, h_hours, uf, diesel, 'H')
    total_f = calculate_single_machine_monthly_cost(f_df, f_days, f_hours, uf, diesel, 'F')
    
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

# --- 4. SIDEBAR CONFIGURACIÓN ---
with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    if st.button("♻️ Resetear App", type="secondary"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    # --- JORNADA LABORAL (RESTAURADO) ---
    with st.expander("🕒 Jornada Laboral", expanded=True):
        st.caption("Define los turnos de cada equipo:")
        
        c_h, c_f = st.columns(2)
        with c_h:
            st.markdown("**🚜 Harvester**")
            st.session_state['h_days'] = st.number_input("Días/Mes H", value=int(st.session_state['h_days']), key="hd_in", on_change=save_config)
            st.session_state['h_hours'] = st.number_input("Horas/Día H", value=float(st.session_state['h_hours']), key="hh_in", on_change=save_config)
        
        with c_f:
            st.markdown("**🚜 Forwarder**")
            st.session_state['f_days'] = st.number_input("Días/Mes F", value=int(st.session_state['f_days']), key="fd_in", on_change=save_config)
            st.session_state['f_hours'] = st.number_input("Horas/Día F", value=float(st.session_state['f_hours']), key="fh_in", on_change=save_config)

    with st.expander("💰 Mercado", expanded=True):
        use_api = st.checkbox("UF Auto", value=True)
        uf_api, _ = get_uf_api()
        val_uf = uf_api if (use_api and uf_api) else st.session_state['uf_manual']
        
        curr_uf = st.number_input("Valor UF", value=float(val_uf), disabled=bool(use_api and uf_api))
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()
        
        st.session_state['fuel_price'] = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config)

    with st.expander("💵 Tarifas Venta"):
        st.session_state['price_h'] = st.number_input("Tarifa Harvester ($/MR)", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Tarifa Forwarder ($/MR)", value=float(st.session_state['price_f']), on_change=save_config)

    with st.expander("📏 Conversión y Distribución"):
        st.session_state['conv_factor'] = st.number_input("Factor m³/MR", value=float(st.session_state['conv_factor']), step=0.01, on_change=save_config)
        alloc = st.slider("% Indirectos a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()

# --- 5. CÁLCULOS GLOBALES ---
# Recuperar valores de la sesión
h_dias = int(st.session_state['h_days'])
h_horas = float(st.session_state['h_hours'])
f_dias = int(st.session_state['f_days'])
f_horas = float(st.session_state['f_hours'])

# Calcular costos usando las jornadas específicas
tot_h_dir, tot_f_dir, tot_ind = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    h_dias, h_horas, f_dias, f_horas, 
    curr_uf, st.session_state['fuel_price']
)

# Costo Mensual Total por Máquina (Directo + Indirecto Asignado)
cost_h_total_mes = tot_h_dir + (tot_ind * st.session_state['alloc_pct'])
cost_f_total_mes = tot_f_dir + (tot_ind * (1 - st.session_state['alloc_pct']))

# --- 6. INTERFAZ ---
st.title("🌲 Dashboard Gerencial de Costos")

tab_dash, tab_h, tab_f, tab_ind = st.tabs(["📊 Resultado Operacional", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos"])

# --- TAB DASHBOARD ---
with tab_dash:
    # 1. Inputs Producción
    st.markdown("### 1. Productividad Real (m³/Hora)")
    c1, c2 = st.columns(2)
    with c1:
        prod_h_m3_hr = st.number_input("Producción Harvester (m³/hr)", value=25.0, step=0.5)
        mr_h_hr = prod_h_m3_hr / st.session_state['conv_factor']
        st.markdown(f'<span class="mr-badge">Equivale a: {mr_h_hr:,.1f} MR/Hr</span>', unsafe_allow_html=True)
        st.caption(f"Jornada H: {h_dias} días x {h_horas} hrs")
    with c2:
        prod_f_m3_hr = st.number_input("Producción Forwarder (m³/hr)", value=28.0, step=0.5)
        mr_f_hr = prod_f_m3_hr / st.session_state['conv_factor']
        st.markdown(f'<span class="mr-badge">Equivale a: {mr_f_hr:,.1f} MR/Hr</span>', unsafe_allow_html=True)
        st.caption(f"Jornada F: {f_dias} días x {f_horas} hrs")

    st.divider()
    
    # 2. LÓGICA DE CÁLCULO DETALLADA (Respetando Jornadas)
    
    # --- HARVESTER ---
    # Ingresos
    inc_h_hr = mr_h_hr * st.session_state['price_h']
    inc_h_day = inc_h_hr * h_horas # Usa horas H
    inc_h_mo = inc_h_day * h_dias  # Usa días H
    # Costos (Ya calculados mensual con días/horas H, desglosamos)
    cost_h_mo = cost_h_total_mes
    cost_h_day = cost_h_mo / h_dias if h_dias > 0 else 0
    cost_h_hr = cost_h_day / h_horas if h_horas > 0 else 0
    # Utilidad
    prof_h_hr = inc_h_hr - cost_h_hr
    prof_h_day = inc_h_day - cost_h_day
    prof_h_mo = inc_h_mo - cost_h_mo
    margin_h = (prof_h_mo / inc_h_mo * 100) if inc_h_mo > 0 else 0

    # --- FORWARDER ---
    # Ingresos
    inc_f_hr = mr_f_hr * st.session_state['price_f']
    inc_f_day = inc_f_hr * f_horas # Usa horas F
    inc_f_mo = inc_f_day * f_dias  # Usa días F
    # Costos
    cost_f_mo = cost_f_total_mes
    cost_f_day = cost_f_mo / f_dias if f_dias > 0 else 0
    cost_f_hr = cost_f_day / f_horas if f_horas > 0 else 0
    # Utilidad
    prof_f_hr = inc_f_hr - cost_f_hr
    prof_f_day = inc_f_day - cost_f_day
    prof_f_mo = inc_f_mo - cost_f_mo
    margin_f = (prof_f_mo / inc_f_mo * 100) if inc_f_mo > 0 else 0

    # --- VISUALIZACIÓN ---
    col_left, col_right = st.columns(2)

    # CARD HARVESTER
    with col_left:
        st.markdown('<div class="machine-header" style="background-color:#eab308; color:#422006">🚜 HARVESTER</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-table-container">', unsafe_allow_html=True)
        
        # Tabla H
        df_h = pd.DataFrame({
            "Periodo": ["Hora", "Día", "Mes"],
            "Ingreso": [fmt_money(inc_h_hr), fmt_money(inc_h_day), fmt_money(inc_h_mo)],
            "Costo": [fmt_money(cost_h_hr), fmt_money(cost_h_day), fmt_money(cost_h_mo)],
            "Ganancia": [fmt_money(prof_h_hr), fmt_money(prof_h_day), fmt_money(prof_h_mo)]
        })
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        st.metric("Margen H", f"{margin_h:.1f}%", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)

    # CARD FORWARDER
    with col_right:
        st.markdown('<div class="machine-header" style="background-color:#22c55e;">🚜 FORWARDER</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-table-container">', unsafe_allow_html=True)
        
        # Tabla F
        df_f = pd.DataFrame({
            "Periodo": ["Hora", "Día", "Mes"],
            "Ingreso": [fmt_money(inc_f_hr), fmt_money(inc_f_day), fmt_money(inc_f_mo)],
            "Costo": [fmt_money(cost_f_hr), fmt_money(cost_f_day), fmt_money(cost_f_mo)],
            "Ganancia": [fmt_money(prof_f_hr), fmt_money(prof_f_day), fmt_money(prof_f_mo)]
        })
        st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.metric("Margen F", f"{margin_f:.1f}%", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    
    # TOTAL CONSOLIDADO
    total_prof = prof_h_mo + prof_f_mo
    total_inc = inc_h_mo + inc_f_mo
    total_margin = (total_prof / total_inc * 100) if total_inc > 0 else 0
    
    st.info(f"💰 **RESULTADO MENSUAL SISTEMA: {fmt_money(total_prof)} de Utilidad ({total_margin:.1f}% Margen)**")

# --- TABS EDICIÓN ---
with tab_h:
    st.header("Costos Harvester")
    edited_h = st.data_editor(st.session_state['df_harvester'], num_rows="dynamic", use_container_width=True,
                              column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_harvester'] = edited_h
    save_config()
    live_h = calculate_single_machine_monthly_cost(edited_h, h_dias, h_horas, curr_uf, st.session_state['fuel_price'], 'H')
    st.success(f"Costo Mensual Directo H: **{fmt_money(live_h)}** (Calculado con {h_dias} días x {h_horas} hrs)")

with tab_f:
    st.header("Costos Forwarder")
    edited_f = st.data_editor(st.session_state['df_forwarder'], num_rows="dynamic", use_container_width=True,
                              column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_forwarder'] = edited_f
    save_config()
    live_f = calculate_single_machine_monthly_cost(edited_f, f_dias, f_horas, curr_uf, st.session_state['fuel_price'], 'F')
    st.success(f"Costo Mensual Directo F: **{fmt_money(live_f)}** (Calculado con {f_dias} días x {f_horas} hrs)")

with tab_ind:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RRHH")
        edited_rrhh = st.data_editor(st.session_state['df_rrhh'], num_rows="dynamic", use_container_width=True, column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_rrhh'] = edited_rrhh
        st.info(f"Total RRHH: {fmt_money(edited_rrhh['Costo Empresa'].sum())}")
    with c2:
        st.subheader("Flota y Varios")
        edited_flota = st.data_editor(st.session_state['df_flota'], num_rows="dynamic", use_container_width=True, column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_flota'] = edited_flota
        st.info(f"Total Flota: {fmt_money(edited_flota['Monto'].sum())}")
    save_config()
    
    total_ind_live = edited_rrhh['Costo Empresa'].sum() + edited_flota['Monto'].sum()
    st.success(f"💰 Total Costos Indirectos Mensual: **{fmt_money(total_ind_live)}**")
