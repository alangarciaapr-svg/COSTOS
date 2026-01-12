import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import io

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(
    page_title="Forestal Costing Pro", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

# CSS Personalizado para dar look "Enterprise"
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
    .css-1d391kg {padding-top: 1rem;} 
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v3_pro.json'

# --- 2. GESTIÓN DE ESTADO (PERSISTENCIA) ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.int64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float64)): return float(obj)
        return json.JSONEncoder.default(self, obj)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config():
    keys = ["uf_manual", "fuel_price", "h_days", "h_hours", "f_days", "f_hours", 
            "df_harvester", "df_forwarder", "df_indirectos", "pickup_days", "alloc_pct"]
    
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

# Inicializar State
if 'init' not in st.session_state:
    saved = load_config()
    st.session_state['uf_manual'] = saved.get('uf_manual', 37000.0)
    st.session_state['fuel_price'] = saved.get('fuel_price', 1000.0)
    
    # DataFrames Iniciales (Si no existen en el save)
    if 'df_harvester' in saved: st.session_state['df_harvester'] = pd.DataFrame(saved['df_harvester'])
    else:
        # Datos Harvester (Avanzado)
        st.session_state['df_harvester'] = pd.DataFrame([
            {"Cat": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
            {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
            {"Cat": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
            {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 100.0},
            {"Cat": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frec": 1, "Valor": 100.0},
            {"Cat": "Insumos", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frec": 1, "Valor": 450000},
            {"Cat": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Ev", "Frec": 600, "Valor": 181840},
            {"Cat": "Mantención", "Ítem": "Mant. 1200h", "Tipo": "$/Ev", "Frec": 1200, "Valor": 181840},
            {"Cat": "Mayor", "Ítem": "Overhaul Motor", "Tipo": "$/Ev", "Frec": 20000, "Valor": 20000000},
        ])

    if 'df_forwarder' in saved: st.session_state['df_forwarder'] = pd.DataFrame(saved['df_forwarder'])
    else:
        st.session_state['df_forwarder'] = pd.DataFrame([
            {"Cat": "Operación", "Ítem": "Arriendo", "Unidad": "$/Mes", "Valor": 8000000},
            {"Cat": "Operación", "Ítem": "Operador", "Unidad": "$/Mes", "Valor": 1900000},
            {"Cat": "Variable", "Ítem": "Petróleo", "Unidad": "Litros/Día", "Valor": 135.0},
            {"Cat": "Mantención", "Ítem": "Mantención Gral", "Unidad": "$/Mes", "Valor": 1500000},
        ])

    st.session_state['init'] = True

# Helper Formato
def fmt_money(x): return f"$ {x:,.0f}".replace(",", ".")

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2959/2959624.png", width=60)
    st.title("Configuración")
    
    st.subheader("1. Variables Mercado")
    curr_uf = st.number_input("Valor UF ($)", value=st.session_state.get('uf_manual', 37000.0), key="uf_manual", on_change=save_config)
    curr_fuel = st.number_input("Diesel ($/Lt)", value=st.session_state.get('fuel_price', 1000.0), key="fuel_price", on_change=save_config)
    
    st.divider()
    st.subheader("2. Distribución Indirectos")
    alloc_h = st.slider("% Asignado a Harvester", 0, 100, 60, key="alloc_pct", on_change=save_config) / 100.0
    
    st.divider()
    st.caption("Forestal System v3.0 Pro")
    
    # Botón Exportar Excel
    if st.button("📥 Descargar Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state['df_harvester'].to_excel(writer, sheet_name='Harvester')
            st.session_state['df_forwarder'].to_excel(writer, sheet_name='Forwarder')
            # (Aquí podrías agregar más hojas)
        st.download_button(label="Click para Bajar .xlsx", data=output.getvalue(), file_name="Costos_Forestal.xlsx", mime="application/vnd.ms-excel")

# --- 4. LÓGICA DE CÁLCULO ---
def calc_harvester(df, days, hours, fuel_p, uf_p):
    total_m = 0
    breakdown = []
    
    total_hours = days * hours
    
    for _, row in df.iterrows():
        cost = 0
        if row['Tipo'] == '$/Mes': cost = row['Valor']
        elif row['Tipo'] == 'UF/Mes': cost = row['Valor'] * uf_p
        elif row['Tipo'] == 'Litros/Día': cost = row['Valor'] * days * fuel_p
        elif row['Tipo'] == '$/Ev': 
            if row['Frec'] > 0 and total_hours > 0:
                cost = (row['Valor'] / row['Frec']) * total_hours
        
        total_m += cost
        breakdown.append({"Categoría": row['Cat'], "Costo": cost})
        
    return total_m, pd.DataFrame(breakdown)

def calc_forwarder(df, days, hours, fuel_p):
    total_m = 0
    breakdown = []
    
    for _, row in df.iterrows():
        cost = 0
        if row['Unidad'] == '$/Mes': cost = row['Valor']
        elif row['Unidad'] == 'Litros/Día': cost = row['Valor'] * days * fuel_p
        
        total_m += cost
        breakdown.append({"Categoría": row['Cat'], "Costo": cost})
        
    return total_m, pd.DataFrame(breakdown)

# --- 5. INTERFAZ PRINCIPAL ---
st.title("🚜 Panel de Control de Costos")
st.markdown("Gestión detallada de costos operativos para sistema CTL (Harvester + Forwarder).")

# PESTAÑAS PRINCIPALES
tab_res, tab_harv, tab_forw, tab_ind = st.tabs(["📊 Dashboard Resumen", "🚜 Harvester (Detalle)", "🚜 Forwarder", "📋 Indirectos"])

# --- TAB 2: HARVESTER ---
with tab_harv:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Parámetros")
        h_days = st.number_input("Días/Mes", value=24, key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día", value=9.0, key="h_hours", on_change=save_config)
        st.info(f"Total: **{h_days * h_hours:,.0f}** horas mes")
    
    with c2:
        st.subheader("Estructura de Costos")
        edited_h = st.data_editor(
            st.session_state['df_harvester'], 
            use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d"),
                "Tipo": st.column_config.SelectColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"]),
                "Cat": st.column_config.SelectColumn(options=["Fijos", "Variable", "Mantención", "Insumos", "Mayor"])
            }
        )
        st.session_state['df_harvester'] = edited_h
        save_config()

# --- TAB 3: FORWARDER ---
with tab_forw:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Parámetros")
        f_days = st.number_input("Días/Mes Fwd", value=24, key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día Fwd", value=9.0, key="f_hours", on_change=save_config)
    
    with c2:
        st.subheader("Estructura Simple")
        edited_f = st.data_editor(
            st.session_state['df_forwarder'], 
            use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="%f"),
                "Unidad": st.column_config.SelectColumn(options=["$/Mes", "Litros/Día"])
            }
        )
        st.session_state['df_forwarder'] = edited_f
        save_config()

# --- CÁLCULOS CENTRALIZADOS ---
cost_h, df_h_bk = calc_harvester(edited_h, h_days, h_hours, curr_fuel, curr_uf)
cost_f, df_f_bk = calc_forwarder(edited_f, f_days, f_hours, curr_fuel)

# --- TAB 4: INDIRECTOS ---
with tab_ind:
    st.info("Estos costos se prorratean entre Harvester y Forwarder según el % definido en la barra lateral.")
    
    if 'df_indirectos' not in st.session_state:
        st.session_state['df_indirectos'] = pd.DataFrame([
            {"Ítem": "Camionetas (Arr+Comb)", "Monto": 1500000},
            {"Ítem": "Prevencionista", "Monto": 800000},
            {"Ítem": "Administración", "Monto": 500000}
        ])
        
    c1, c2 = st.columns([2, 1])
    with c1:
        edited_ind = st.data_editor(st.session_state['df_indirectos'], use_container_width=True, num_rows="dynamic",
                                   column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_indirectos'] = edited_ind
        save_config()
    
    total_ind = edited_ind['Monto'].sum()
    with c2:
        st.metric("Total Indirectos", fmt_money(total_ind))
        st.write(f"Harvester ({alloc_h*100:.0f}%): **{fmt_money(total_ind * alloc_h)}**")
        st.write(f"Forwarder ({(1-alloc_h)*100:.0f}%): **{fmt_money(total_ind * (1-alloc_h))}**")

# --- TAB 1: DASHBOARD (RESULTADOS) ---
with tab_res:
    # 1. KPIs Principales
    total_h_full = cost_h + (total_ind * alloc_h)
    total_f_full = cost_f + (total_ind * (1-alloc_h))
    
    h_per_hour = total_h_full / (h_days*h_hours) if (h_days*h_hours) > 0 else 0
    f_per_hour = total_f_full / (f_days*f_hours) if (f_days*f_hours) > 0 else 0
    
    st.markdown("### 💰 Resumen Ejecutivo")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo Total Sistema", fmt_money(total_h_full + total_f_full), delta="Mensual")
    col2.metric("Costo Hora Harvester", fmt_money(h_per_hour), help="Incluye Prorrateo Indirectos")
    col3.metric("Costo Hora Forwarder", fmt_money(f_per_hour), help="Incluye Prorrateo Indirectos")
    col4.metric("Producción Objet.", "4.500 m3/mes", delta_color="off") # Placeholder
    
    st.divider()
    
    # 2. Gráficos
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.subheader("Distribución Harvester")
        if not df_h_bk.empty:
            fig_h = px.pie(df_h_bk, values='Costo', names='Categoría', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_h, use_container_width=True)
            
    with c_chart2:
        st.subheader("Comparativa Mensual")
        df_comp = pd.DataFrame({
            "Máquina": ["Harvester", "Forwarder"],
            "Costo Directo": [cost_h, cost_f],
            "Indirecto Asignado": [total_ind * alloc_h, total_ind * (1-alloc_h)]
        })
        fig_bar = px.bar(df_comp, x="Máquina", y=["Costo Directo", "Indirecto Asignado"], title="Estructura de Costos", 
                         color_discrete_sequence=[ "#2ecc71", "#95a5a6"])
        st.plotly_chart(fig_bar, use_container_width=True)
