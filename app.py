import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import io
import requests

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(
    page_title="Gestión Costos Forestales", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
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
</style>
""", unsafe_allow_html=True)

# Actualizamos versión para evitar conflictos de caché
CONFIG_FILE = 'forest_config_v8_mr.json'

# --- 2. FUNCIONES DE SOPORTE ---

@st.cache_data(ttl=3600) 
def get_uf_api():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=5)
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
            "df_harvester", "df_forwarder", "df_indirectos", "alloc_pct", "sales_price"]
    
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame): state_to_save[k] = val.to_dict('records')
            else: state_to_save[k] = val
                
    with open(CONFIG_FILE, 'w') as f:
        json.dump(state_to_save, f, cls=NumpyEncoder)

def safe_float(val, default_val):
    try: return float(val) if val is not None else default_val
    except: return default_val

# --- 3. INICIALIZACIÓN DE ESTADO ---
if 'init' not in st.session_state:
    saved = load_config()
    
    st.session_state['uf_manual'] = safe_float(saved.get('uf_manual'), 38000.0)
    st.session_state['fuel_price'] = safe_float(saved.get('fuel_price'), 1000.0)
    st.session_state['alloc_pct'] = safe_float(saved.get('alloc_pct'), 0.6)
    st.session_state['sales_price'] = safe_float(saved.get('sales_price'), 8500.0) # Valor típico MR referencia
    
    # Listas Iniciales
    if 'df_harvester' in saved: st.session_state['df_harvester'] = pd.DataFrame(saved['df_harvester'])
    else:
        st.session_state['df_harvester'] = pd.DataFrame([
            {"Cat": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
            {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
            {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 100.0},
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

# --- 4. BARRA LATERAL ---
with st.sidebar:
    st.header("1. Variables Mercado")
    
    # UF
    use_api = st.checkbox("UF Automática", value=True)
    uf_val = st.session_state.get('uf_manual', 38000.0)
    if use_api:
        api_val, api_date = get_uf_api()
        if api_val: 
            uf_val = api_val
            st.success(f"UF Hoy: ${api_val:,.2f}")
    
    curr_uf = st.number_input("Valor UF ($)", value=float(uf_val), disabled=use_api and 'api_val' in locals() and api_val is not None)
    if curr_uf != st.session_state.get('uf_manual'):
        st.session_state['uf_manual'] = curr_uf
        save_config()

    # Diesel
    curr_fuel = st.number_input("Diesel ($/Lt)", value=float(st.session_state.get('fuel_price', 1000.0)), on_change=save_config, key="fuel_price")
    
    st.divider()
    st.header("2. Precio Venta")
    
    # CAMBIO AQUÍ: Etiqueta MR
    curr_sales_price = st.number_input("Precio Venta ($/MR)", value=float(st.session_state.get('sales_price', 8500.0)), on_change=save_config, key="sales_price")

    st.divider()
    st.header("3. Distribución Costos Fijos")
    st.info("Porcentaje de Costos Fijos asignado a Harvester:")
    
    # Slider Seguro
    raw_alloc = st.session_state.get('alloc_pct', 0.6)
    if raw_alloc > 1.0: raw_alloc = raw_alloc / 100.0
    default_alloc_int = int(raw_alloc * 100)
    
    alloc_slider = st.slider("% Harvester", 0, 100, default_alloc_int)
    
    new_pct = alloc_slider / 100.0
    if new_pct != st.session_state.get('alloc_pct'):
        st.session_state['alloc_pct'] = new_pct
        save_config()
    
    alloc_h = st.session_state['alloc_pct']

    st.write(f"🚜 **Harvester:** {alloc_h*100:.0f}%")
    st.write(f"🚜 **Forwarder:** {(1-alloc_h)*100:.0f}%")
    
    st.divider()
    if st.button("📥 Bajar Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state['df_harvester'].to_excel(writer, sheet_name='Harvester', index=False)
            st.session_state['df_forwarder'].to_excel(writer, sheet_name='Forwarder', index=False)
        st.download_button("Descargar", data=output.getvalue(), file_name="Costos_Forestal_MR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- 5. CÁLCULOS ---
def calc_harvester(df, days, hours, fuel_p, uf_p):
    total = 0
    total_hrs = days * hours
    for _, row in df.iterrows():
        val = row['Valor'] or 0.0
        tipo = row['Tipo']
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf_p
        elif tipo == 'Litros/Día': cost = val * days * fuel_p
        elif tipo == '$/Ev': 
            frec = row.get('Frec', 1)
            if frec > 0 and total_hrs > 0: cost = (val / frec) * total_hrs
        total += cost
    return total

def calc_forwarder(df, days, hours, fuel_p):
    total = 0
    for _, row in df.iterrows():
        val = row['Valor'] or 0.0
        cost = 0
        if row['Unidad'] == '$/Mes': cost = val
        elif row['Unidad'] == 'Litros/Día': cost = val * days * fuel_p
        total += cost
    return total

# --- 6. INTERFAZ PRINCIPAL ---
st.title("🌲 Sistema de Costos y Punto de Equilibrio (MR)")

tab_res, tab_harv, tab_forw, tab_fijos = st.tabs(["📊 Punto de Equilibrio", "🚜 Harvester", "🚜 Forwarder", "🏢 Costos Fijos Compartidos"])

# TAB HARVESTER
with tab_harv:
    c1, c2 = st.columns([1, 3])
    with c1:
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 24)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 9.0)), key="h_hours", on_change=save_config)
    with c2:
        st.session_state['df_harvester'] = st.data_editor(
            st.session_state['df_harvester'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", default=0),
                "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"], required=True),
                "Cat": st.column_config.SelectboxColumn(options=["Fijos", "Variable", "Mantención", "Insumos"], required=True)
            }
        )
        save_config()

# TAB FORWARDER
with tab_forw:
    c1, c2 = st.columns([1, 3])
    with c1:
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 24)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 9.0)), key="f_hours", on_change=save_config)
    with c2:
        st.session_state['df_forwarder'] = st.data_editor(
            st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", default=0),
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
            }
        )
        save_config()

# CÁLCULOS DIRECTOS
cost_h = calc_harvester(st.session_state['df_harvester'], h_days, h_hours, curr_fuel, curr_uf)
cost_f = calc_forwarder(st.session_state['df_forwarder'], f_days, f_hours, curr_fuel)

# TAB COSTOS FIJOS (INDIRECTOS)
with tab_fijos:
    st.info(f"Estos costos se repartirán: **{alloc_h*100:.0f}% al Harvester** y **{(1-alloc_h)*100:.0f}% al Forwarder**.")
    
    if 'df_indirectos' not in st.session_state:
        st.session_state['df_indirectos'] = pd.DataFrame([{"Ítem": "Camionetas", "Monto": 1500000}, {"Ítem": "Prevencionista", "Monto": 800000}])
    elif isinstance(st.session_state['df_indirectos'], list):
         st.session_state['df_indirectos'] = pd.DataFrame(st.session_state['df_indirectos'])
         
    st.session_state['df_indirectos'] = st.data_editor(st.session_state['df_indirectos'], use_container_width=True, num_rows="dynamic",
                                   column_config={"Monto": st.column_config.NumberColumn(format="$ %d", default=0)})
    save_config()
    
    total_fijos = st.session_state['df_indirectos']['Monto'].fillna(0).sum()
    st.metric("Total Costos Fijos Compartidos", fmt_money(total_fijos))

# --- TAB RESULTADOS ---
with tab_res:
    # Asignación de Costos Fijos según Porcentaje
    fijo_h = total_fijos * alloc_h
    fijo_f = total_fijos * (1 - alloc_h)
    
    total_h = cost_h + fijo_h
    total_f = cost_f + fijo_f
    grand_total = total_h + total_f
    
    # Cálculos de Negocio (Punto de Equilibrio)
    price = curr_sales_price if curr_sales_price > 0 else 1.0
    breakeven_units = grand_total / price
    
    st.subheader("🏁 Punto de Equilibrio Mensual")
    c1, c2, c3 = st.columns(3)
    c1.metric("Costo Mensual Total", fmt_money(grand_total), "Operativo + Fijos")
    
    # CAMBIO AQUÍ: Etiqueta MR
    c2.metric("Precio Venta", fmt_money(price), "por MR")
    c3.metric("Producción Necesaria", f"{breakeven_units:,.0f} MR", "Para cubrir costos", delta_color="off")
    
    st.progress(0.5) 
    st.caption(f"Si produces más de **{breakeven_units:,.0f} MR**, el sistema es rentable.")

    st.divider()
    
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("🚜 Costos Harvester")
        st.write(f"Operativo Directo: {fmt_money(cost_h)}")
        st.write(f"Fijo Asignado ({alloc_h*100:.0f}%): {fmt_money(fijo_h)}")
        st.success(f"**Total H: {fmt_money(total_h)}**")
        
        # Donut Chart
        fig = px.pie(names=["Operativo", "Fijo Asignado"], values=[cost_h, fijo_h], hole=0.5, height=250)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("🚜 Costos Forwarder")
        st.write(f"Operativo Directo: {fmt_money(cost_f)}")
        st.write(f"Fijo Asignado ({(1-alloc_h)*100:.0f}%): {fmt_money(fijo_f)}")
        st.success(f"**Total F: {fmt_money(total_f)}**")
        
        fig2 = px.pie(names=["Operativo", "Fijo Asignado"], values=[cost_f, fijo_f], hole=0.5, height=250)
        st.plotly_chart(fig2, use_container_width=True)
