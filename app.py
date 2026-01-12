import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
import requests

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(
    page_title="ForestCost Pro Final",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Profesionales
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #166534; /* Verde Bosque */
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #1f2937;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .total-row {
        font-weight: bold;
        background-color: #f0fdf4;
    }
    .stDataEditor {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    .sim-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v27_fixed.json'

# --- 2. GESTIÓN DE GUARDADO (PERSISTENCIA) ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config():
    config_data = {}
    for k, v in st.session_state.items():
        if k in EXPECTED_KEYS:
            if isinstance(v, pd.DataFrame):
                config_data[k] = v.to_dict('records')
            elif isinstance(v, (int, float, str, bool, list, dict)):
                config_data[k] = v
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, cls=NumpyEncoder)

EXPECTED_KEYS = [
    "use_auto_uf", "uf_manual", "fuel_price", 
    "conversion_factor", "sales_price_mr", "h_rev_pct",
    "h_days_month", "h_hours_day", "f_days_month", "f_hours_day",
    "df_harvester_v25", "df_forwarder_v25", "df_indirect_v24",
    "sim_m3_h_val", "sim_m3_f_val", "pickup_days_use"
]

if 'config_loaded' not in st.session_state:
    saved_config = load_config()
    for key in EXPECTED_KEYS:
        if key in saved_config:
            val = saved_config[key]
            if key in ["df_harvester_v25", "df_forwarder_v25", "df_indirect_v24"]:
                st.session_state[key] = pd.DataFrame(val)
            else:
                st.session_state[key] = val
    st.session_state['config_loaded'] = True

# --- 3. UTILS ---
def fmt(x):
    return f"{x:,.0f}".replace(",", ".")

def result_card(col, title, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="font-size:11px; color:#888;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_uf_value():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['serie'][0]['valor'], data['serie'][0]['fecha'][:10]
    except:
        pass
    return None, None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Parámetros Base")
    
    st.markdown("### 1. Economía")
    uf_api, fecha_api = get_uf_value()
    use_auto_uf = st.checkbox("UF Automática", value=True, key="use_auto_uf", on_change=save_config)
    if use_auto_uf and uf_api:
        current_uf = uf_api
        st.success(f"UF Hoy: ${fmt(current_uf)}")
    else:
        current_uf = st.number_input("Valor UF", value=39704.93, step=100.0, key="uf_manual", on_change=save_config)

    fuel_price = st.number_input("Precio Petróleo ($/L)", value=774, step=10, key="fuel_price", on_change=save_config)

    st.markdown("### 2. Comercial")
    conversion_factor = st.number_input("Factor (M3 / F = MR)", value=0.65, step=0.01, key="conversion_factor", on_change=save_config)
    sales_price_mr = st.number_input("Venta ($/MR)", value=4500, step=100, key="sales_price_mr", on_change=save_config)
    
    st.markdown("### 3. Distribución")
    st.caption("Define el % de Ingresos y Costos Fijos por máquina.")
    h_rev_pct = st.slider("% Harvester", 0, 100, 70, key="h_rev_pct", on_change=save_config)
    f_rev_pct = 100 - h_rev_pct
    
    h_income = sales_price_mr * (h_rev_pct / 100)
    f_income = sales_price_mr * (f_rev_pct / 100)
    
    st.info(f"Ingreso H: ${fmt(h_income)} | F: ${fmt(f_income)}")

# --- 5. CUERPO PRINCIPAL ---

st.title("🌲 ForestCost: Gestión de Costos y Rentabilidad")

# ==========================================
# A. CONFIGURACIÓN DE JORNADA
# ==========================================
with st.expander("📅 Configuración de Jornada Laboral", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester**")
        h_days = st.number_input("Días/Mes (H)", value=28, step=1, key="h_days_month", on_change=save_config)
        h_hours_day = st.number_input("Hrs/Día (H)", value=10.0, step=0.5, key="h_hours_day", on_change=save_config)
        h_total_hours = h_days * h_hours_day
        st.caption(f"Total: {fmt(h_total_hours)} Hrs Mes")
    with c2:
        st.markdown("**Forwarder**")
        f_days = st.number_input("Días/Mes (F)", value=25, step=1, key="f_days_month", on_change=save_config)
        f_hours_day = st.number_input("Hrs/Día (F)", value=9.0, step=0.5, key="f_hours_day", on_change=save_config)
        f_total_hours = f_days * f_hours_day
        st.caption(f"Total: {fmt(f_total_hours)} Hrs Mes")

st.divider()

# ==========================================
# B. TABLAS DE COSTOS (HARVESTER Y FORWARDER)
# ==========================================
# Función CORREGIDA para renderizar tabla idéntica
def render_machine_table(prefix, col_obj, machine_days, machine_hours_total, fuel_p, uf_val):
    with col_obj:
        st.subheader(f"🚜 {prefix}")
        key_df = f"df_{prefix.lower()}_v25"
        
        # Estructura Idéntica para ambas
        if key_df not in st.session_state:
            data = [
                {"Categoría": "Fijos", "Ítem": "Arriendo", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 10000000},
                {"Categoría": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 1900000},
                {"Categoría": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 1900000},
                {"Categoría": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frecuencia (Hrs)": 1, "Valor Input": 150.0},
                {"Categoría": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frecuencia (Hrs)": 1, "Valor Input": 150.0},
                {"Categoría": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 600, "Valor Input": 180000},
                {"Categoría": "Mantención", "Ítem": "Mant. 1200h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 1200, "Valor Input": 180000},
                {"Categoría": "Mantención", "Ítem": "Mant. 1800h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 1800, "Valor Input": 1900000},
                {"Categoría": "Mantención", "Ítem": "Hidráulica 6000h", "Tipo": "$/Evento", "Frecuencia (Hrs)": 6000, "Valor Input": 19500000},
                {"Categoría": "Mantención", "Ítem": "Grúa/Garra 10kh", "Tipo": "$/Evento", "Frecuencia (Hrs)": 10000, "Valor Input": 15000000},
                {"Categoría": "Mantención", "Ítem": "Electrónica Mensual", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 100000},
                {"Categoría": "Consumibles", "Ítem": "Insumos (Cadenas/Neum)", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 300000},
                {"Categoría": "Consumibles", "Ítem": "Grasa/Aceites", "Tipo": "$/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 250000},
                {"Categoría": "Otros", "Ítem": "Seguro (UF)", "Tipo": "UF/Mes", "Frecuencia (Hrs)": 1, "Valor Input": 19},
                {"Categoría": "Reserva", "Ítem": "Overhaul/Motor", "Tipo": "$/Evento", "Frecuencia (Hrs)": 20000, "Valor Input": 40000000},
            ]
            st.session_state[key_df] = pd.DataFrame(data)
            
        edited_df = st.data_editor(
            st.session_state[key_df],
            key=f"editor_{prefix}",
            column_config={
                "Categoría": None, # Oculta
                "Ítem": st.column_config.TextColumn(disabled=True),
                "Tipo": st.column_config.TextColumn(disabled=True),
                "Frecuencia (Hrs)": st.column_config.NumberColumn(format="%d"),
                "Valor Input": st.column_config.NumberColumn("Valor Input", format="%d", required=True),
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
        st.session_state[key_df] = edited_df
        save_config()
        
        # --- Cálculo ---
        total_month = 0
        for index, row in edited_df.iterrows():
            val = row["Valor Input"]
            tipo = row["Tipo"]
            freq = row.get("Frecuencia (Hrs)", 1)
            
            row_cost = 0
            if tipo == "$/Mes": row_cost = val
            elif tipo == "UF/Mes": row_cost = val * uf_val
            elif tipo == "Litros/Día": row_cost = val * machine_days * fuel_p
            elif tipo == "$/Evento":
                if freq > 0 and machine_hours_total > 0:
                    row_cost = (val / freq) * machine_hours_total
            total_month += row_cost
            
        total_hr = total_month / machine_hours_total if machine_hours_total else 0
        total_day = total_month / 30
        
        # Tarjetas
        k1, k2 = st.columns(2)
        result_card(k1, "Total Mensual", f"${fmt(total_month)}")
        result_card(k2, "Costo Hora", f"${fmt(total_hr)}")
        
        return total_month, total_hr

col_tab1, col_tab2 = st.columns(2)
h_total_m, h_total_hr = render_machine_table("Harvester", col_tab1, h_days, h_total_hours, fuel_price, current_uf)
f_total_m, f_total_hr = render_machine_table("Forwarder", col_tab2, f_days, f_total_hours, fuel_price, current_uf)

# ==========================================
# C. COSTOS INDIRECTOS (FIXED COSTS)
# ==========================================
st.markdown("---")
st.subheader("🏢 Costos Indirectos (Fijos)")

key_ind = "df_indirect_v24"
if key_ind not in st.session_state:
    data_ind = [
        {"Ítem": "Instalación de faena", "Tipo": "$/Mes", "Valor Input": 0},
        {"Ítem": "Arriendo camioneta 1", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Arriendo camioneta 2", "Tipo": "UF/Mes", "Valor Input": 38.0},
        {"Ítem": "Combustible camioneta 1", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Combustible camioneta 2", "Tipo": "Litros/Día", "Valor Input": 12.0},
        {"Ítem": "Prevencionista de riesgos", "Tipo": "$/Mes", "Valor Input": 800000},
        {"Ítem": "Otros / agregar mensual", "Tipo": "$/Mes", "Valor Input": 100000},
        {"Ítem": "Pensión personal", "Tipo": "$/Mes", "Valor Input": 1890000},
        {"Ítem": "EPP Y Ropa de trabajo", "Tipo": "$/Mes", "Valor Input": 200000},
        {"Ítem": "Gastos Adm y Gerencia", "Tipo": "$/Mes", "Valor Input": 500000},
    ]
    st.session_state[key_ind] = pd.DataFrame(data_ind)

col_ind_edit, col_ind_res = st.columns([2, 1])

with col_ind_edit:
    edited_ind = st.data_editor(
        st.session_state[key_ind],
        key="editor_indirect_list",
        column_config={
            "Ítem": st.column_config.TextColumn(disabled=True),
            "Tipo": st.column_config.TextColumn(disabled=True),
            "Valor Input": st.column_config.NumberColumn("Valor", format="%f")
        },
        hide_index=True,
        use_container_width=True,
        height=350
    )
    st.session_state[key_ind] = edited_ind
    save_config()

# Parámetro auxiliar para combustible camionetas
with col_ind_edit:
    pickup_days = st.number_input("Días de Uso Camionetas (Para cálculo litros)", value=30, key="pickup_days_use", on_change=save_config)

# Cálculo Total Indirectos
total_shared = 0
for idx, row in edited_ind.iterrows():
    val = row["Valor Input"]
    tipo = row["Tipo"]
    rc = 0
    if tipo == "UF/Mes": rc = val * current_uf
    elif tipo == "Litros/Día": rc = val * pickup_days * fuel_price
    else: rc = val
    total_shared += rc

# MOSTRAR RESULTADO TOTAL FIJO
with col_ind_res:
    st.markdown("##### Resumen Costos Fijos")
    result_card(st, "TOTAL MENSUAL", f"${fmt(total_shared)}")
    result_card(st, "TOTAL DIARIO (30d)", f"${fmt(total_shared/30)}")

# Asignación automática por %
h_share_pct = h_rev_pct / 100.0
f_share_pct = f_rev_pct / 100.0
shared_h = total_shared * h_share_pct
shared_f = total_shared * f_share_pct

st.info(f"Asignación Automática ({h_rev_pct}% / {f_rev_pct}%): Harvester absorbe **${fmt(shared_h)}** y Forwarder absorbe **${fmt(shared_f)}**")

# ==========================================
# D. RESUMEN DE COSTOS (PRESUPUESTO)
# ==========================================
st.divider()
st.subheader("📊 Resumen de Costos por Máquina (Presupuesto)")

final_h_hr = (h_total_m + shared_h) / h_total_hours if h_total_hours else 0
final_f_hr = (f_total_m + shared_f) / f_total_hours if f_total_hours else 0
sys_hr = final_h_hr + final_f_hr

summary_df = pd.DataFrame([
    {
        "Ítem": "Costo Directo Máquina",
        "Harvester ($/Mes)": h_total_m,
        "Forwarder ($/Mes)": f_total_m,
        "SISTEMA ($/Mes)": h_total_m + f_total_m
    },
    {
        "Ítem": "Costo Fijo Asignado",
        "Harvester ($/Mes)": shared_h,
        "Forwarder ($/Mes)": shared_f,
        "SISTEMA ($/Mes)": total_shared
    },
    {
        "Ítem": "TOTAL PRESUPUESTO",
        "Harvester ($/Mes)": h_total_m + shared_h,
        "Forwarder ($/Mes)": f_total_m + shared_f,
        "SISTEMA ($/Mes)": h_total_m + f_total_m + total_shared
    },
    {
        "Ítem": "COSTO HORA FINAL",
        "Harvester ($/Mes)": final_h_hr,
        "Forwarder ($/Mes)": final_f_hr,
        "SISTEMA ($/Mes)": sys_hr
    }
])

st.dataframe(
    summary_df,
    column_config={
        "Harvester ($/Mes)": st.column_config.NumberColumn(format="$%d"),
        "Forwarder ($/Mes)": st.column_config.NumberColumn(format="$%d"),
        "SISTEMA ($/Mes)": st.column_config.NumberColumn(format="$%d"),
    },
    hide_index=True,
    use_container_width=True
)

# ==========================================
# E. SIMULADOR DE RENTABILIDAD
# ==========================================
st.divider()
st.subheader("🚀 Simulador de Rentabilidad Real")
st.markdown("Ingresa la producción esperada para calcular márgenes reales.")

col_sim_in, col_sim_res = st.columns([1, 2])

with col_sim_in:
    st.markdown("#### Producción")
    m3_h = st.number_input("Harvester ($m^3$/hr)", value=25.0, step=0.5, key="sim_m3_h_val", on_change=save_config)
    m3_f = st.number_input("Forwarder ($m^3$/hr)", value=28.0, step=0.5, key="sim_m3_f_val", on_change=save_config)
    
    # Conversión
    mr_h = m3_h / conversion_factor if conversion_factor else 0
    mr_f = m3_f / conversion_factor if conversion_factor else 0
    
    st.markdown("---")
    st.write(f"🚜 H: **{mr_h:.1f} MR/hr**")
    st.write(f"🚜 F: **{mr_f:.1f} MR/hr**")

with col_sim_res:
    # Cálculos Rentabilidad
    # Costo Unitario = Costo Hora Total / Producción MR
    c_unit_h = final_h_hr / mr_h if mr_h else 0
    c_unit_f = final_f_hr / mr_f if mr_f else 0
    
    # Utilidad = Precio Venta Asignado - Costo Unitario
    util_h = h_income - c_unit_h
    util_f = f_income - c_unit_f
    
    # Margen %
    pct_h = (util_h / h_income * 100) if h_income else 0
    pct_f = (util_f / f_income * 100) if f_income else 0
    
    st.markdown("#### Estado de Resultados por Máquina")
    
    sim_results = pd.DataFrame([
        {
            "Máquina": "Harvester",
            "Ingreso ($/MR)": h_income,
            "Costo Total ($/MR)": c_unit_h,
            "Utilidad ($/MR)": util_h,
            "Margen %": pct_h
        },
        {
            "Máquina": "Forwarder",
            "Ingreso ($/MR)": f_income,
            "Costo Total ($/MR)": c_unit_f,
            "Utilidad ($/MR)": util_f,
            "Margen %": pct_f
        }
    ])
    
    st.dataframe(
        sim_results,
        column_config={
            "Ingreso ($/MR)": st.column_config.NumberColumn(format="$%d"),
            "Costo Total ($/MR)": st.column_config.NumberColumn(format="$%d"),
            "Utilidad ($/MR)": st.column_config.NumberColumn(format="$%d"),
            "Margen %": st.column_config.ProgressColumn(format="%.1f%%", min_value=-0.5, max_value=0.5, 
                                                        help="Verde positivo, Rojo negativo")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Kpi Global Sistema
    mr_sys = min(mr_h, mr_f)
    c_sys = sys_hr / mr_sys if mr_sys else 0
    u_sys = sales_price_mr - c_sys
    m_sys = (u_sys / sales_price_mr * 100) if sales_price_mr else 0
    
    st.success(f"**MARGEN SISTEMA TOTAL:** {m_sys:.1f}% (${fmt(u_sys)}/MR) calculado sobre producción limitada de {mr_sys:.1f} MR/hr")
