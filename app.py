import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import io
import requests
from datetime import datetime

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(
    page_title="Forestal Costing Auto", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50;}
    .stMetric {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1); 
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v6_final.json'

# --- 2. FUNCIONES AUXILIARES (API Y PERSISTENCIA) ---

@st.cache_data(ttl=3600) 
def get_uf_api():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            valor = data['serie'][0]['valor']
            fecha = data['serie'][0]['fecha'][:10]
            return valor, fecha
    except Exception as e:
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
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {} # Si el archivo está corrupto, retorna vacío
    return {}

def save_config():
    keys = ["uf_manual", "fuel_price", "h_days", "h_hours", "f_days", "f_hours", 
            "df_harvester", "df_forwarder", "df_indirectos", "pickup_days", "alloc_pct", "sales_price", "monthly_prod"]
    
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame):
                state_to_save[k] = val.to_dict('records')
            else:
                state_to_save[k] = val
                
    with open(CONFIG_FILE, 'w') as f:
        json.dump(state_to_save, f, cls=NumpyEncoder)

# --- FUNCIÓN DE SEGURIDAD PARA CARGAR NÚMEROS ---
def safe_float(val, default_val):
    try:
        if val is None: return default_val
        return float(val)
    except (ValueError, TypeError):
        return default_val

# Inicializar State
if 'init' not in st.session_state:
    saved = load_config()
    
    # Aquí aplicamos la función de seguridad safe_float
    st.session_state['uf_manual'] = safe_float(saved.get('uf_manual'), 38000.0)
    st.session_state['fuel_price'] = safe_float(saved.get('fuel_price'), 1000.0)
    st.session_state['alloc_pct'] = safe_float(saved.get('alloc_pct'), 0.6)
    st.session_state['sales_price'] = safe_float(saved.get('sales_price'), 12000.0)
    st.session_state['monthly_prod'] = safe_float(saved.get('monthly_prod'), 4500.0)
    
    # Cargar DataFrames
    if 'df_harvester' in saved: st.session_state['df_harvester'] = pd.DataFrame(saved['df_harvester'])
    else:
        st.session_state['df_harvester'] = pd.DataFrame([
            {"Cat": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
            {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
            {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 100.0},
            {"Cat": "Insumos", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frec": 1, "Valor": 450000},
            {"Cat": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Ev", "Frec": 600, "Valor": 181840},
        ])

    if 'df_forwarder' in saved: st.session_state['df_forwarder'] = pd.DataFrame(saved['df_forwarder'])
    else:
        st.session_state['df_forwarder'] = pd.DataFrame([
            {"Cat": "Operación", "Ítem": "Arriendo", "Unidad": "$/Mes", "Valor": 8000000},
            {"Cat": "Operación", "Ítem": "Operador", "Unidad": "$/Mes", "Valor": 1900000},
            {"Cat": "Variable", "Ítem": "Petróleo", "Unidad": "Litros/Día", "Valor": 135.0},
        ])
    
    st.session_state['init'] = True

# Helper Formato
def fmt_money(x): return f"$ {x:,.0f}".replace(",", ".")

# --- 3. BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.title("Configuración")
    
    # A. UF AUTOMÁTICA
    st.subheader("1. Indicadores Económicos")
    use_api = st.checkbox("🔄 Obtener UF Automática", value=True)
    
    uf_val = st.session_state.get('uf_manual', 38000.0)
    
    if use_api:
        api_val, api_date = get_uf_api()
        if api_val:
            uf_val = api_val
            st.success(f"UF Hoy ({api_date}): ${api_val:,.2f}")
    
    # Input UF
    curr_uf = st.number_input("Valor UF ($)", value=float(uf_val), disabled=use_api and 'api_val' in locals() and api_val is not None, key="uf_input")
    
    if curr_uf != st.session_state.get('uf_manual'):
        st.session_state['uf_manual'] = curr_uf
        save_config()

    # Input Petróleo
    curr_fuel = st.number_input("Diesel ($/Lt)", value=float(st.session_state.get('fuel_price', 1000.0)), key="fuel_price", on_change=save_config)
    
    st.divider()
    
    # B. VENTAS Y PRODUCCIÓN
    st.subheader("2. Venta y Producción")
    
    # Recuperación segura de valores
    val_sales = st.session_state.get('sales_price', 12000.0)
    val_prod = st.session_state.get('monthly_prod', 4500.0)

    curr_sales_price = st.number_input("Precio Venta ($/m³)", value=float(val_sales), key="sales_price", on_change=save_config)
    curr_prod = st.number_input("Producción Mensual (m³)", value=float(val_prod), key="monthly_prod", on_change=save_config)

    st.divider()

    # C. DISTRIBUCIÓN UNIFICADA
    st.subheader("3. Asignación Unificada")
    st.info("Define el % de Costos Indirectos para Harvester.")
    
    raw_alloc = st.session_state.get('alloc_pct', 0.6)
    # Corrección de rango
    if raw_alloc is None: raw_alloc = 0.6
    if raw_alloc > 1.0: raw_alloc = raw_alloc / 100.0
    raw_alloc = max(0.0, min(1.0, raw_alloc))
    
    default_alloc_int = int(raw_alloc * 100)
    
    alloc_slider = st.slider("% Asignado a Harvester", 0, 100, default_alloc_int, key="alloc_slider")
    
    new_pct = alloc_slider / 100.0
    if new_pct != st.session_state.get('alloc_pct'):
        st.session_state['alloc_pct'] = new_pct
        save_config()
    
    alloc_h = st.session_state['alloc_pct']
    
    st.markdown(f"* **Harvester:** {alloc_h*100:.0f}% \n* **Forwarder:** {(1-alloc_h)*100:.0f}%")

    st.divider()
    
    if st.button("📥 Descargar Excel"):
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state['df_harvester'].to_excel(writer, sheet_name='Harvester', index=False)
                st.session_state['df_forwarder'].to_excel(writer, sheet_name='Forwarder', index=False)
                if 'df_indirectos' in st.session_state:
                    st.session_state['df_indirectos'].to_excel(writer, sheet_name='Indirectos', index=False)
            st.download_button(label="Bajar .xlsx", data=output.getvalue(), file_name="Costos_Forestal.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except:
            st.error("Error generando Excel.")

# --- 4. LÓGICA DE CÁLCULO ---
def calc_harvester(df, days, hours, fuel_p, uf_p):
    total_m = 0
    breakdown = []
    total_hours = days * hours
    for _, row in df.iterrows():
        cost = 0
        tipo, valor = row['Tipo'], row['Valor']
        if valor is None: valor = 0.0
        
        if tipo == '$/Mes': cost = valor
        elif tipo == 'UF/Mes': cost = valor * uf_p
        elif tipo == 'Litros/Día': cost = valor * days * fuel_p
        elif tipo == '$/Ev': 
            frec = row.get('Frec', 1)
            if frec > 0 and total_hours > 0: cost = (valor / frec) * total_hours
        total_m += cost
        breakdown.append({"Categoría": row['Cat'], "Costo": cost})
    return total_m, pd.DataFrame(breakdown)

def calc_forwarder(df, days, hours, fuel_p):
    total_m = 0
    breakdown = []
    for _, row in df.iterrows():
        cost = 0
        unidad, valor = row['Unidad'], row['Valor']
        if valor is None: valor = 0.0
        
        if unidad == '$/Mes': cost = valor
        elif unidad == 'Litros/Día': cost = valor * days * fuel_p
        total_m += cost
        breakdown.append({"Categoría": row['Cat'], "Costo": cost})
    return total_m, pd.DataFrame(breakdown)

# --- 5. INTERFAZ PRINCIPAL ---
st.title("🌲 Sistema de Costos Forestal")

tab_res, tab_harv, tab_forw, tab_ind = st.tabs(["📊 Resultados y Margen", "🚜 Harvester", "🚜 Forwarder", "📋 Indirectos"])

# --- TAB 2: HARVESTER ---
with tab_harv:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Parámetros H")
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 24)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 9.0)), key="h_hours", on_change=save_config)
        st.caption(f"Horas Totales: {h_days * h_hours}")
    with c2:
        edited_h = st.data_editor(
            st.session_state['df_harvester'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", default=0),
                "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"], required=True),
                "Cat": st.column_config.SelectboxColumn(options=["Fijos", "Variable", "Mantención", "Insumos", "Mayor"], required=True)
            }
        )
        st.session_state['df_harvester'] = edited_h
        save_config()

# --- TAB 3: FORWARDER ---
with tab_forw:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Parámetros F")
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 24)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 9.0)), key="f_hours", on_change=save_config)
    with c2:
        edited_f = st.data_editor(
            st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="%f", default=0),
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
            }
        )
        st.session_state['df_forwarder'] = edited_f
        save_config()

# CÁLCULOS
cost_h, df_h_bk = calc_harvester(edited_h, h_days, h_hours, curr_fuel, curr_uf)
cost_f, df_f_bk = calc_forwarder(edited_f, f_days, f_hours, curr_fuel)

# --- TAB 4: INDIRECTOS ---
with tab_ind:
    if 'df_indirectos' not in st.session_state:
        st.session_state['df_indirectos'] = pd.DataFrame([{"Ítem": "Camionetas", "Monto": 1500000}, {"Ítem": "Prevencionista", "Monto": 800000}])
    elif isinstance(st.session_state['df_indirectos'], list):
         st.session_state['df_indirectos'] = pd.DataFrame(st.session_state['df_indirectos'])
        
    c1, c2 = st.columns([2, 1])
    with c1:
        edited_ind = st.data_editor(st.session_state['df_indirectos'], use_container_width=True, num_rows="dynamic",
                                   column_config={"Monto": st.column_config.NumberColumn(format="$ %d", default=0)})
        st.session_state['df_indirectos'] = edited_ind
        save_config()
    
    total_ind = edited_ind['Monto'].fillna(0).sum()
    with c2:
        st.metric("Total Indirectos", fmt_money(total_ind))
        
        safe_alloc = float(alloc_h)
        if safe_alloc > 1.0: safe_alloc = safe_alloc / 100.0
        safe_alloc = max(0.0, min(1.0, safe_alloc))
        
        st.progress(safe_alloc)
        st.caption(f"Se asigna {safe_alloc*100:.0f}% a Harvester según slider.")

# --- TAB 1: RESULTADOS ---
with tab_res:
    # 1. Cálculos de Negocio
    # Blindaje contra nulos en el cálculo final
    safe_prod = curr_prod if curr_prod is not None else 0.0
    safe_price = curr_sales_price if curr_sales_price is not None else 0.0
    
    total_sales = safe_prod * safe_price
    
    total_cost_h = cost_h + (total_ind * alloc_h)
    total_cost_f = cost_f + (total_ind * (1-alloc_h))
    total_cost_global = total_cost_h + total_cost_f
    
    margin_global = total_sales - total_cost_global
    margin_pct = (margin_global / total_sales) * 100 if total_sales > 0 else 0
    
    st.header("📊 Rentabilidad del Proyecto")
    
    # KPIs Superiores
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ventas Totales", fmt_money(total_sales), f"{safe_prod:,.0f} m³")
    k2.metric("Costo Total", fmt_money(total_cost_global), "- Operativo + Ind.")
    k3.metric("Margen ($)", fmt_money(margin_global), delta_color="normal")
    k4.metric("Margen (%)", f"{margin_pct:.1f}%", delta=f"{margin_pct:.1f}%")
    
    st.divider()

    c_h, c_f = st.columns(2)
    
    with c_h:
        st.subheader("🚜 Harvester")
        st.write(f"Costo Directo: **{fmt_money(cost_h)}**")
        st.write(f"Indirecto ({alloc_h*100:.0f}%): **{fmt_money(total_ind * alloc_h)}**")
        st.info(f"Costo Total: **{fmt_money(total_cost_h)}**")
        if df_h_bk is not None and not df_h_bk.empty:
            fig_h = px.pie(df_h_bk, values='Costo', names='Categoría', hole=0.4, title="Distribución Costos H")
            st.plotly_chart(fig_h, use_container_width=True)

    with c_f:
        st.subheader("🚜 Forwarder")
        st.write(f"Costo Directo: **{fmt_money(cost_f)}**")
        st.write(f"Indirecto ({(1-alloc_h)*100:.0f}%): **{fmt_money(total_ind * (1-alloc_h))}**")
        st.info(f"Costo Total: **{fmt_money(total_cost_f)}**")
        df_comp = pd.DataFrame({
            "Tipo": ["Directo", "Indirecto"],
            "Monto": [cost_f, total_ind * (1-alloc_h)]
        })
        fig_f = px.bar(df_comp, x="Tipo", y="Monto", title="Estructura Costos F", color="Tipo")
        st.plotly_chart(fig_f, use_container_width=True)
