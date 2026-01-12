import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Calculadora Forestal CTL", layout="wide")

st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v22_harvester_adv.json'

# --- 2. PERSISTENCIA Y CARGA DE DATOS ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config():
    config = {key: st.session_state[key] for key in st.session_state if key in EXPECTED_KEYS}
    # Convertir DataFrames a dict para JSON
    for key in ["df_harvester_v22", "df_forwarder_v17", "df_indirect_list_v19"]:
        if key in config and isinstance(config[key], pd.DataFrame):
            config[key] = config[key].to_dict('records')
            
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, cls=NumpyEncoder)

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    "df_harvester_v22", "df_forwarder_v17", "df_indirect_list_v19",
    "alloc_method", "h_share_pct_manual", 
    "sim_m3_h_val", "sim_m3_f_val",
    "pickup_days_use"
]

# Inicializar Session State
if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            if key in ["df_harvester_v22", "df_forwarder_v17", "df_indirect_list_v19"]:
                st.session_state[key] = pd.DataFrame(val)
            else:
                st.session_state[key] = val
    st.session_state['config_loaded'] = True

# --- 3. BARRA LATERAL (INPUTS GLOBALES) ---
with st.sidebar:
    st.header("1. Parámetros Económicos")
    current_uf = st.number_input("Valor UF ($)", value=37000.0, step=100.0, key="uf_manual", on_change=save_config)
    fuel_price = st.number_input("Precio Diesel ($/Lt)", value=1000.0, step=10.0, key="fuel_price", on_change=save_config)
    
    st.divider()
    st.header("2. Jornada Laboral")
    h_days = st.number_input("Días/Mes Harvester", value=24, key="h_days_month", on_change=save_config)
    h_hours = st.number_input("Horas/Día Harvester", value=9.0, key="h_hours_day", on_change=save_config)
    f_days = st.number_input("Días/Mes Forwarder", value=24, key="f_days_month", on_change=save_config)
    f_hours = st.number_input("Horas/Día Forwarder", value=9.0, key="f_hours_day", on_change=save_config)

    h_total_hours = h_days * h_hours
    f_total_hours = f_days * f_hours
    
    st.info(f"H-Total: {h_total_hours} hrs | F-Total: {f_total_hours} hrs")

# Helper de formato
def fmt(x): return "{:,.0f}".format(x).replace(",", ".")

# --- 4. FUNCIONES DE CÁLCULO ---

def render_harvester_advanced(col_obj, machine_days, machine_hours_total, fuel_p, uf_val):
    with col_obj:
        st.subheader("🚜 Harvester")
        key_df = "df_harvester_v22"

        # Cargar datos por defecto si no existen
        if key_df not in st.session_state:
            data = [
                {"Categoría": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 10900000},
                {"Categoría": "Fijos", "Ítem": "Operador turno 1", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 1923721},
                {"Categoría": "Fijos", "Ítem": "Operador turno 2", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 1923721},
                
                {"Categoría": "Variable", "Ítem": "Petróleo turno 1", "Tipo": "Litros/Día", "Frecuencia": 1, "Valor Input": 100.0},
                {"Categoría": "Variable", "Ítem": "Petróleo turno 2", "Tipo": "Litros/Día", "Frecuencia": 1, "Valor Input": 100.0},
                
                {"Categoría": "Mantención", "Ítem": "Mantención 600 horas", "Tipo": "$/Evento", "Frecuencia": 600, "Valor Input": 181840},
                {"Categoría": "Mantención", "Ítem": "Mantención 1200 horas", "Tipo": "$/Evento", "Frecuencia": 1200, "Valor Input": 181840},
                {"Categoría": "Mantención", "Ítem": "Mantención 1800 horas", "Tipo": "$/Evento", "Frecuencia": 1800, "Valor Input": 1990848},
                {"Categoría": "Mantención", "Ítem": "Mant. Hidráulica 6000h", "Tipo": "$/Evento", "Frecuencia": 6000, "Valor Input": 19500000},
                {"Categoría": "Mantención", "Ítem": "Mant. Grúa 10000h", "Tipo": "$/Evento", "Frecuencia": 10000, "Valor Input": 15000000},
                {"Categoría": "Mantención", "Ítem": "Sist. Electrónico Cabezal", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 50000},
                {"Categoría": "Mantención", "Ítem": "Sist. Electrónico Base", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 50000},
                {"Categoría": "Mantención", "Ítem": "Mant. Mecánica 10000h", "Tipo": "$/Evento", "Frecuencia": 10000, "Valor Input": 8328816},
                
                {"Categoría": "Consumibles", "Ítem": "Cadenas turno 1 (5)", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 30580},
                {"Categoría": "Consumibles", "Ítem": "Cadenas turno 2 (5)", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 30580},
                {"Categoría": "Consumibles", "Ítem": "Espadas turno 1", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 131250},
                {"Categoría": "Consumibles", "Ítem": "Espadas turno 2", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 131250},
                {"Categoría": "Consumibles", "Ítem": "Grasa (10 tubos)", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 80270},
                {"Categoría": "Consumibles", "Ítem": "Aceite Hidráulico", "Tipo": "$/Mes", "Frecuencia": 1, "Valor Input": 167428},
                
                {"Categoría": "Otros", "Ítem": "Seguro RENTA (UF)", "Tipo": "UF/Mes", "Frecuencia": 1, "Valor Input": 19},
                
                {"Categoría": "Amortización", "Ítem": "Neumáticos 20000 hrs", "Tipo": "$/Evento", "Frecuencia": 20000, "Valor Input": 20000000},
                {"Categoría": "Amortización", "Ítem": "Valtras 20000 hrs", "Tipo": "$/Evento", "Frecuencia": 20000, "Valor Input": 20000000},
                {"Categoría": "Amortización", "Ítem": "Overhaul Tren Motriz", "Tipo": "$/Evento", "Frecuencia": 20000, "Valor Input": 24000000},
                {"Categoría": "Amortización", "Ítem": "Overhaul Motor", "Tipo": "$/Evento", "Frecuencia": 20000, "Valor Input": 20000000},
            ]
            st.session_state[key_df] = pd.DataFrame(data)

        st.info("Ingresa **Litros/Día** para Petróleo. Ingresa **Valor del Evento** para mantenciones, el sistema prorratea.")

        edited_df = st.data_editor(
            st.session_state[key_df],
            key="editor_harvester_adv",
            column_config={
                "Categoría": st.column_config.TextColumn(disabled=True),
                "Ítem": st.column_config.TextColumn(disabled=True),
                "Unidad": st.column_config.TextColumn(disabled=True),
                "Tipo": st.column_config.TextColumn(disabled=True),
                "Frecuencia": st.column_config.NumberColumn("Frec (Hrs)", disabled=True),
                "Valor Input": st.column_config.NumberColumn("Valor Input", format="%d", required=True),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        st.session_state[key_df] = edited_df
        save_config()

        # CÁLCULO
        total_month = 0
        
        for idx, row in edited_df.iterrows():
            val = row["Valor Input"]
            tipo = row["Tipo"]
            freq = row["Frecuencia"]
            
            item_cost_month = 0
            
            if tipo == "$/Mes":
                item_cost_month = val
            elif tipo == "UF/Mes":
                item_cost_month = val * uf_val
            elif tipo == "Litros/Día":
                # Litros * Días Trabajados Harvester * Precio
                item_cost_month = val * machine_days * fuel_p
            elif tipo == "$/Evento":
                # Costo Evento / Frecuencia * Horas Mensuales
                if freq > 0 and machine_hours_total > 0:
                    cost_per_hour_maint = val / freq
                    item_cost_month = cost_per_hour_maint * machine_hours_total
            
            total_month += item_cost_month
            
        total_hr = total_month / machine_hours_total if machine_hours_total else 0
        
        st.success(f"**Costo Hora: ${fmt(total_hr)}**")
        with st.expander("Ver Resumen Mensual"):
            st.write(f"Total Gasto Mensual: **${fmt(total_month)}**")
            
        return total_month, total_hr

def render_forwarder(col_obj, machine_days, machine_hours_total, fuel_p):
    with col_obj:
        st.subheader("🚜 Forwarder")
        key_df = "df_forwarder_v17"
        
        if key_df not in st.session_state:
            data = [
                {"Ítem": "Arriendo Forwarder", "Valor Input": 8000000, "Unidad": "$/Mes"},
                {"Ítem": "Sueldo Operador", "Valor Input": 1900000, "Unidad": "$/Mes"},
                {"Ítem": "Petróleo (Litros/Día)", "Valor Input": 135.0, "Unidad": "Litros/Día"},
                {"Ítem": "Mantenciones / Insumos", "Valor Input": 1500000, "Unidad": "$/Mes"},
            ]
            st.session_state[key_df] = pd.DataFrame(data)
            
        edited_df = st.data_editor(
            st.session_state[key_df],
            key="editor_forwarder_simple",
            hide_index=True,
            use_container_width=True
        )
        st.session_state[key_df] = edited_df
        save_config()

        total_month_sum = 0
        for index, row in edited_df.iterrows():
            val = row["Valor Input"]
            unit = row["Unidad"]
            row_cost = 0
            
            if unit == "Litros/Día":
                row_cost = val * machine_days * fuel_p
            else:
                row_cost = val
            
            total_month_sum += row_cost
            
        total_hr = total_month_sum / machine_hours_total if machine_hours_total else 0
        st.success(f"**Costo Hora: ${fmt(total_hr)}**")
        return total_month_sum, total_hr

# --- 5. RENDERIZADO PRINCIPAL ---
st.title("🌲 Calculadora de Costos Maquinaria")

col_t1, col_t2 = st.columns(2)
# Aquí llamamos a la nueva función avanzada para Harvester
h_total_m, h_total_hr = render_harvester_advanced(col_t1, h_days, h_total_hours, fuel_price, current_uf)
# Y la simple para Forwarder
f_total_m, f_total_hr = render_forwarder(col_t2, f_days, f_total_hours, fuel_price)

# --- 6. INDIRECTOS ---
st.markdown("---")
st.header("📋 Costos Indirectos")

key_ind = "df_indirect_list_v19"

if key_ind not in st.session_state:
    data_ind = [
        {"Ítem": "Instalación de faena", "Tipo": "$/Mes", "Valor Input": 0},
        {"Ítem": "Arriendo camioneta 1", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Arriendo camioneta 2", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Combustible camioneta 1", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Combustible camioneta 2", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Prevencionista de riesgos", "Tipo": "$/Mes", "Valor Input": 800000},
        {"Ítem": "Pensión personal", "Tipo": "$/Mes", "Valor Input": 1890000},
        {"Ítem": "EPP Y Ropa de trabajo", "Tipo": "$/Mes", "Valor Input": 200000},
        {"Ítem": "Gastos Adm y Gerencia", "Tipo": "$/Mes", "Valor Input": 500000},
    ]
    st.session_state[key_ind] = pd.DataFrame(data_ind)

with st.expander("📝 Editar Costos Indirectos", expanded=True):
    col_i1, col_i2 = st.columns([2, 1])
    
    with col_i2:
        pickup_days = st.number_input("Días Mes Camionetas", value=30, key="pickup_days_use", on_change=save_config)
        st.caption(f"UF Actual: ${fmt(current_uf)} | Diesel: ${fmt(fuel_price)}")

    with col_i1:
        edited_ind = st.data_editor(
            st.session_state[key_ind],
            column_config={
                "Tipo": st.column_config.TextColumn(disabled=True),
                "Valor Input": st.column_config.NumberColumn("Valor", format="%f")
            },
            hide_index=True,
            use_container_width=True
        )
        st.session_state[key_ind] = edited_ind
        save_config()

# Cálculo Indirectos
total_shared = 0
for idx, row in edited_ind.iterrows():
    val = row["Valor Input"]
    tipo = row["Tipo"]
    
    row_cost = 0
    if tipo == "UF/Mes": row_cost = val * current_uf
    elif tipo == "Litros/Día": row_cost = val * pickup_days * fuel_price
    else: row_cost = val
    total_shared += row_cost

st.success(f"**TOTAL INDIRECTOS: ${fmt(total_shared)} /Mes**")

# --- 7. RESUMEN FINAL ---
st.markdown("---")
st.subheader("📊 Resultados Sistema (Harvester + Forwarder)")

# Distribución Indirectos
if st.session_state.get("alloc_method") == "Manual":
    h_pct = st.sidebar.slider("% Indirectos Harvester", 0, 100, 60, key="h_share_pct_manual", on_change=save_config) / 100.0
else:
    tt = h_total_hours + f_total_hours
    h_pct = h_total_hours / tt if tt > 0 else 0.5

ind_h = total_shared * h_pct
ind_f = total_shared * (1 - h_pct)

# Totales
final_h_mes = h_total_m + ind_h
final_f_mes = f_total_m + ind_f

final_h_hr = final_h_mes / h_total_hours if h_total_hours else 0
final_f_hr = final_f_mes / f_total_hours if f_total_hours else 0

c1, c2, c3 = st.columns(3)
c1.metric("Costo Total Sistema/Mes", f"${fmt(final_h_mes + final_f_mes)}")
c2.metric("Costo Hora Harvester (Total)", f"${fmt(final_h_hr)}")
c3.metric("Costo Hora Forwarder (Total)", f"${fmt(final_f_hr)}")
