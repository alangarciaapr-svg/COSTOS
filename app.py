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
    page_title="Costos Forestales (Enfoque Hora)", 
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
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Actualizamos versión para limpiar caché
CONFIG_FILE = 'forest_config_v9_hourly.json'

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

# --- 3. INICIALIZACIÓN ---
if 'init' not in st.session_state:
    saved = load_config()
    
    st.session_state['uf_manual'] = safe_float(saved.get('uf_manual'), 38000.0)
    st.session_state['fuel_price'] = safe_float(saved.get('fuel_price'), 1000.0)
    st.session_state['alloc_pct'] = safe_float(saved.get('alloc_pct'), 0.6)
    st.session_state['sales_price'] = safe_float(saved.get('sales_price'), 8500.0)
    
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
    st.header("2. Ingreso Venta")
    curr_sales_price = st.number_input("Precio Venta ($/MR)", value=float(st.session_state.get('sales_price', 8500.0)), on_change=save_config, key="sales_price")

    st.divider()
    st.header("3. Distribución Indirectos")
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
        st.download_button("Descargar", data=output.getvalue(), file_name="Costos_Hora.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- 5. CÁLCULOS (MOTOR PRINCIPAL) ---
def calc_monthly_total(df, days, hours, fuel_p, uf_p, machine_type='H'):
    """Calcula el costo total mensual sumando todos los ítems"""
    total_month = 0
    total_hrs = days * hours
    
    for _, row in df.iterrows():
        val = row['Valor'] or 0.0
        
        # Unificar nombres de columnas (Tipo vs Unidad)
        if machine_type == 'H': tipo = row['Tipo']
        else: tipo = row['Unidad']
            
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf_p
        elif tipo == 'Litros/Día': cost = val * days * fuel_p
        elif tipo == '$/Ev': 
            frec = row.get('Frec', 1)
            if frec > 0 and total_hrs > 0: cost = (val / frec) * total_hrs
        
        total_month += cost
        
    return total_month

# --- 6. INTERFAZ ---
st.title("🌲 Costos Forestales: Análisis por Hora")

tab_res, tab_harv, tab_forw, tab_fijos = st.tabs(["📊 Resultado Hora/MR", "🚜 Harvester", "🚜 Forwarder", "🏢 Indirectos"])

# TAB HARVESTER
with tab_harv:
    c1, c2 = st.columns([1, 3])
    with c1:
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 24)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 9.0)), key="h_hours", on_change=save_config)
        h_total_hrs = h_days * h_hours
        st.info(f"Horas Totales: **{h_total_hrs:,.0f}** hrs/mes")
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
        f_total_hrs = f_days * f_hours
        st.info(f"Horas Totales: **{f_total_hrs:,.0f}** hrs/mes")
    with c2:
        st.session_state['df_forwarder'] = st.data_editor(
            st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", default=0),
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
            }
        )
        save_config()

# TAB INDIRECTOS
with tab_fijos:
    if 'df_indirectos' not in st.session_state:
        st.session_state['df_indirectos'] = pd.DataFrame([{"Ítem": "Camionetas", "Monto": 1500000}, {"Ítem": "Prevencionista", "Monto": 800000}])
    elif isinstance(st.session_state['df_indirectos'], list):
         st.session_state['df_indirectos'] = pd.DataFrame(st.session_state['df_indirectos'])
         
    c1, c2 = st.columns([2, 1])
    with c1:
        st.session_state['df_indirectos'] = st.data_editor(st.session_state['df_indirectos'], use_container_width=True, num_rows="dynamic",
                                   column_config={"Monto": st.column_config.NumberColumn(format="$ %d", default=0)})
        save_config()
    with c2:
        total_indirectos_mes = st.session_state['df_indirectos']['Monto'].fillna(0).sum()
        st.metric("Total Indirectos Mensual", fmt_money(total_indirectos_mes))
        st.write(f"Asignación Harvester: **{alloc_h*100:.0f}%**")

# --- PROCESAMIENTO FINAL (CÁLCULO POR HORA) ---

# 1. Calcular Totales Mensuales Directos
costo_mensual_h_directo = calc_monthly_total(st.session_state['df_harvester'], h_days, h_hours, curr_fuel, curr_uf, 'H')
costo_mensual_f_directo = calc_monthly_total(st.session_state['df_forwarder'], f_days, f_hours, curr_fuel, curr_uf, 'F')

# 2. Prorratear Indirectos Mensuales
indirecto_mensual_h = total_indirectos_mes * alloc_h
indirecto_mensual_f = total_indirectos_mes * (1 - alloc_h)

# 3. Totales Mensuales Globales
total_mensual_h = costo_mensual_h_directo + indirecto_mensual_h
total_mensual_f = costo_mensual_f_directo + indirecto_mensual_f
total_sistema_mensual = total_mensual_h + total_mensual_f

# 4. CONVERSIÓN A COSTO HORA (EL CORAZÓN DEL SISTEMA)
# Evitamos división por cero
safe_h_hrs = h_total_hrs if h_total_hrs > 0 else 1
safe_f_hrs = f_total_hrs if f_total_hrs > 0 else 1

costo_hora_h = total_mensual_h / safe_h_hrs
costo_hora_f = total_mensual_f / safe_f_hrs

# Costo Sistema Hora = Suma de costos hora individuales (Asumiendo operación simultánea)
costo_sistema_hora = costo_hora_h + costo_hora_f

# 5. PUNTO DE EQUILIBRIO POR HORA
# ¿Cuántos MR debo hacer por hora para pagar el costo de esa hora?
precio_mr = curr_sales_price if curr_sales_price > 0 else 1
prod_necesaria_hora = costo_sistema_hora / precio_mr

# --- TAB RESULTADOS ---
with tab_res:
    st.markdown("### 🏁 Análisis de Rentabilidad por Hora")
    
    # KPIs Principales
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Costo Sistema Total", fmt_money(costo_sistema_hora), "Costo por Hora Operativa")
    k2.metric("Precio Venta", fmt_money(precio_mr), "por MR")
    k3.metric("Prod. Necesaria", f"{prod_necesaria_hora:.1f} MR/Hr", "Equilibrio (Breakeven)")
    
    # Cálculo inverso: Si hago X producción, cuánto gano
    st.divider()
    st.markdown("#### 🔮 Simulador de Margen")
    
    prod_real = st.slider("Producción Real Estimada (MR/Hr)", min_value=0.0, max_value=30.0, value=float(prod_necesaria_hora)*1.2, step=0.1)
    
    ingreso_hora = prod_real * precio_mr
    utilidad_hora = ingreso_hora - costo_sistema_hora
    margen_pct = (utilidad_hora / ingreso_hora) * 100 if ingreso_hora > 0 else 0
    
    col_sim1, col_sim2 = st.columns(2)
    col_sim1.metric("Ingreso por Hora", fmt_money(ingreso_hora))
    col_sim2.metric("Utilidad por Hora", fmt_money(utilidad_hora), f"{margen_pct:.1f}% Margen", delta_color="normal")

    st.divider()
    
    # Tabla Detalle Visual (Mes, Día, Hora)
    st.subheader("📋 Desglose de Costos (Visualización)")
    
    # Preparamos datos para la tabla
    data_view = {
        "Ítem": ["Harvester (Directo + Ind.)", "Forwarder (Directo + Ind.)", "TOTAL SISTEMA"],
        "Mensual ($)": [fmt_money(total_mensual_h), fmt_money(total_mensual_f), fmt_money(total_sistema_mensual)],
        "Diario Aprox ($)": [fmt_money(total_mensual_h/h_days) if h_days else 0, fmt_money(total_mensual_f/f_days) if f_days else 0, fmt_money(total_sistema_mensual/h_days)], # Diario sistema ref H
        "Costo Hora Real ($)": [fmt_money(costo_hora_h), fmt_money(costo_hora_f), fmt_money(costo_sistema_hora)]
    }
    st.dataframe(pd.DataFrame(data_view), use_container_width=True)
    
    # Gráfico
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.caption("Composición Costo Hora Harvester")
        fig_h = px.pie(names=["Directo Hora", "Indirecto Hora"], values=[costo_mensual_h_directo/safe_h_hrs, indirecto_mensual_h/safe_h_hrs], hole=0.4)
        st.plotly_chart(fig_h, use_container_width=True)
    with c_chart2:
        st.caption("Composición Costo Hora Forwarder")
        fig_f = px.pie(names=["Directo Hora", "Indirecto Hora"], values=[costo_mensual_f_directo/safe_f_hrs, indirecto_mensual_f/safe_f_hrs], hole=0.4)
        st.plotly_chart(fig_f, use_container_width=True)
