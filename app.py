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
    .machine-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 5px solid #3b82f6; /* Azul por defecto */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card-h { border-top-color: #eab308; } /* Amarillo Harvester */
    .card-f { border-top-color: #22c55e; } /* Verde Forwarder */
    
    .card-title { font-size: 1.2em; font-weight: 800; color: #334155; margin-bottom: 15px; text-transform: uppercase; }
    
    .res-row { display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 4px; }
    .res-label { color: #64748b; font-weight: 500; }
    .res-val { color: #0f172a; font-weight: 700; }
    .res-net { font-size: 1.3em; font-weight: 800; color: #16a34a; margin-top: 10px; text-align: right; }
    .res-loss { font-size: 1.3em; font-weight: 800; color: #dc2626; margin-top: 10px; text-align: right; }

    .mr-badge { background-color: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v18_machine_split.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
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

def calculate_system_costs(h_df, f_df, rrhh_df, flota_df, days_h, hrs_h, days_f, hrs_f, uf, diesel):
    total_h = calculate_single_machine_monthly_cost(h_df, days_h, hrs_h, uf, diesel, 'H')
    total_f = calculate_single_machine_monthly_cost(f_df, days_f, hrs_f, uf, diesel, 'F')
    
    rrhh_df = rrhh_df.fillna(0)
    flota_df = flota_df.fillna(0)
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect, days_h*hrs_h, days_f*hrs_f

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
            "alloc_pct", "price_h", "price_f", "conv_factor"] # Changed sales_price to price_h/f
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
init_key('price_h', 6500.0) # Tarifa Harvester
init_key('price_f', 5000.0) # Tarifa Forwarder
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
    st.markdown("## ⚙️ Configuración")
    if st.button("♻️ Reset App"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    with st.expander("💰
