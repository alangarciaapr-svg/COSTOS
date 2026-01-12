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

# CSS Personalizado
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
    /* Ajuste para que los tabs se vean mejor */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1); 
    }
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
    st.session_state['alloc_pct'] = saved.get('alloc_pct', 0.6) # Default 60%
    
    # DataFrames Iniciales
    if 'df_harvester' in saved: st.session_state['df_harvester'] = pd.DataFrame(saved['df_harvester'])
    else:
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
    st.title("Configuración")
    
    st.subheader("1. Variables Mercado")
    curr_uf = st.number_input("Valor UF ($)", value=st.session_state.get('uf_manual', 37000.0), key="uf_manual", on_change=save_config)
    curr_fuel = st.number_input("Diesel ($/Lt)", value=st.session_state.get('fuel_price', 1000.0), key="fuel_price", on_change=save_config)
    
    st.divider()
    st.subheader("2. Distribución Indirectos")
    # Multiplicamos por 100 para visualizar, dividimos para guardar
    default_alloc = int(st.session_state.get('alloc_pct', 0.6) * 100)
    alloc_slider = st.slider("% Asignado a Harvester", 0, 100, default_alloc, key="alloc_slider")
    
    # Actualizar estado si cambia el slider
    if alloc_slider / 100.0 != st.session_state.get('alloc_pct'):
        st.session_state['alloc_pct'] = alloc_slider / 100.0
        save_config()
    
    alloc_h = st.session_state['alloc_pct']

    st.divider()
    
    # Botón Exportar Excel
    if st.button("📥 Descargar Excel"):
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state['df_harvester'].to_excel(writer, sheet_name='Harvester', index=False)
                st.session_state['df_forwarder'].to_excel(writer, sheet_name='Forwarder', index=False)
                if 'df_indirectos' in st.session_state:
                    st.session_state['df_indirectos'].to_excel(writer, sheet_name='Indirectos', index=False)
            
            st.download_button(
                label="Click para Guardar .xlsx", 
                data=output.getvalue(), 
                file_name="Costos_Forestal.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error al generar Excel: {e}. Verifica haber instalado 'xlsxwriter'.")

# --- 4. LÓGICA DE CÁLCULO ---
def calc_harvester(df, days, hours, fuel_p, uf_p):
    total_m = 0
    breakdown = []
    
    total_hours = days * hours
    
    for _, row in df.iterrows():
        cost = 0
        tipo = row['Tipo']
        valor = row['Valor']
        
        if tipo == '$/Mes': cost = valor
        elif tipo == 'UF/Mes': cost = valor * uf_p
        elif tipo == 'Litros/Día': cost = valor * days * fuel_p
        elif tipo == '$/Ev': 
            frec = row.get('Frec', 1)
            if frec > 0 and total_hours > 0:
                cost = (valor / frec) * total_hours
        
        total_m += cost
        breakdown.append({"Categoría": row['Cat'], "Costo": cost})
        
    return total_m, pd.DataFrame(breakdown)

def calc_forwarder(df, days, hours, fuel_p):
    total_m = 0
    breakdown = []
    
    for _, row in df.iterrows():
        cost = 0
        unidad = row['Unidad']
        valor = row['Valor']
        
        if unidad == '$/Mes': cost = valor
        elif unidad == 'Litros/Día': cost = valor * days * fuel_p
        
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
        h_days = st.number_input("Días/Mes", value=st.session_state.get('h_days', 24), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día", value=st.session_state.get('h_hours', 9.0), key="h_hours", on_change=save_config)
        st.info(f"Total: **{h_days * h_hours:,.0f}** horas mes")
    
    with c2:
        st.subheader("Estructura de Costos")
        # --- AQUÍ ESTABA EL ERROR CORREGIDO ---
        edited_h = st.data_editor(
            st.session_state['df_harvester'], 
            use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d"),
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
        st.subheader("Parámetros")
        f_days = st.number_input("Días/Mes Fwd", value=st.session_state.get('f_days', 24), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día Fwd", value=st.session_state.get('f_hours', 9.0), key="f_hours", on_change=save_config)
    
    with c2:
        st.subheader("Estructura Simple")
        # --- AQUÍ TAMBIÉN CORREGIMOS EL SELECTBOX ---
        edited_f = st.data_editor(
            st.session_state['df_forwarder'], 
            use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="%f"),
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
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
    elif isinstance(st.session_state['df_indirectos'], list): # Recuperar de JSON si se guardó como lista
         st.session_state['df_indirectos'] = pd.DataFrame(st.session_state['df_indirectos'])
        
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
