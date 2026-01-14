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
    
    .mr-info {
        font-size: 0.9em;
        color: #0ea5e9;
        font-weight: 600;
        background-color: #e0f2fe;
        padding: 4px 8px;
        border-radius: 4px;
        margin-top: 5px;
        display: inline-block;
    }

    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v17_final_reqs.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    """Formatea números como moneda CLP"""
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
    """Calcula el costo mensual de una tabla específica para mostrar totales en vivo"""
    df = df.fillna(0)
    total = 0
    total_hours = days * hours
    
    for _, row in df.iterrows():
        val = float(row.get('Valor', 0))
        # Determinar columna de tipo/unidad
        tipo = row.get('Tipo') if machine_type == 'H' else row.get('Unidad')
        if not tipo: tipo = '$/Mes' # Fallback
        
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
    # Wrapper global usando la función individual
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
            "df_harvester", "df_forwarder", "df_rrhh", "df_flota", 
            "alloc_pct", "sales_price", "target_margin_h", "target_margin_f", "conv_factor"]
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame): state_to_save[k] = val.to_dict('records')
            else: state_to_save[k] = val
    with open(CONFIG_FILE, 'w') as f:
        json.dump(state_to_save, f, cls=NumpyEncoder)

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
init_key('sales_price', 11500.0)
init_key('alloc_pct', 0.5)
init_key('target_margin_h', 35.0)
init_key('target_margin_f', 35.0)
init_key('conv_factor', 2.44)
init_key('h_days', 28)
init_key('h_hours', 10.0)
init_key('f_days', 28)
init_key('f_hours', 10.0)

# DataFrames
init_key('df_harvester', pd.DataFrame([
    {"Cat": "Fijos", "Ítem": "Arriendo Base", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
    {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Insumos", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frec": 1, "Valor": 450000},
    {"Cat": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Ev", "Frec": 600, "Valor": 350000},
]))

init_key('df_forwarder', pd.DataFrame([
    {"Cat": "Operación", "Ítem": "Arriendo", "Unidad": "$/Mes", "Valor": 8000000},
    {"Cat": "Operación", "Ítem": "Operador", "Unidad": "$/Mes", "Valor": 1900000},
    {"Cat": "Variable", "Ítem": "Petróleo", "Unidad": "Litros/Día", "Valor": 135.0},
]))

init_key('df_rrhh', pd.DataFrame([
    {"Cargo": "Jefe de Faena", "Sueldo Líquido": 1800000, "Costo Empresa": 2300000},
    {"Cargo": "Mecánico", "Sueldo Líquido": 1200000, "Costo Empresa": 1600000},
]))

init_key('df_flota', pd.DataFrame([
    {"Ítem": "Camionetas (Arriendo)", "Monto": 1600000},
    {"Ítem": "Combustible Apoyo", "Monto": 600000},
]))

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    if st.button("♻️ Resetear App", type="secondary"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    with st.expander("💰 Mercado y Tarifas", expanded=True):
        use_api = st.checkbox("UF Automática", value=True)
        uf_api_val, _ = get_uf_api()
        val_uf_display = uf_api_val if (use_api and uf_api_val) else st.session_state['uf_manual']
        
        curr_uf = st.number_input("Valor UF ($)", value=float(val_uf_display), disabled=(use_api and uf_api_val is not None))
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()

        curr_fuel = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config, key="fuel_price")
        curr_sales = st.number_input("Tarifa Venta ($/MR)", value=float(st.session_state['sales_price']), on_change=save_config, key="sales_price")

    with st.expander("📏 Conversión Volumen"):
        curr_factor = st.number_input("Factor (m³/MR)", value=float(st.session_state.get('conv_factor', 2.44)), step=0.01, key="conv_factor_input")
        if curr_factor != st.session_state['conv_factor']:
            st.session_state['conv_factor'] = curr_factor
            save_config()

    with st.expander("⚖️ Distribución Indirectos"):
        alloc = st.slider("% Carga a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()

# --- 5. CÁLCULOS GLOBALES ---
tot_h_dir, tot_f_dir, tot_ind, hrs_h, hrs_f = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    int(st.session_state['h_days']), float(st.session_state['h_hours']), 
    int(st.session_state['f_days']), float(st.session_state['f_hours']), 
    curr_uf, curr_fuel
)

ind_h = tot_ind * st.session_state['alloc_pct']
ind_f = tot_ind * (1 - st.session_state['alloc_pct'])
final_h_mes = tot_h_dir + ind_h
final_f_mes = tot_f_dir + ind_f
cost_mensual_sistema = final_h_mes + final_f_mes
hrs_sistema_mes = max(hrs_h, hrs_f) if max(hrs_h, hrs_f) > 0 else 1

# --- 6. INTERFAZ PRINCIPAL ---
st.title("🌲 Sistema de Costos Forestales Profesional")

tab_dash, tab_h, tab_f, tab_ind, tab_sim = st.tabs([
    "📊 Dashboard Gerencial", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos", "📈 Simulador de Tarifas"
])

# --- TAB 1: DASHBOARD ---
with tab_dash:
    st.markdown("### 📊 Resultado Operacional (Basado en Productividad Horaria)")
    
    # 1. Inputs Producción
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        st.markdown("**Productividad Harvester**")
        prod_h_hr = st.number_input("Producción (m³ Sólidos/Hora)", value=25.0, step=0.5, key="p_h")
        # Detalle de Conversión
        prod_h_mr = prod_h_hr / st.session_state['conv_factor']
        st.markdown(f'<div class="mr-info">Equivale a: <b>{prod_h_mr:,.2f} MR/hr</b></div>', unsafe_allow_html=True)
        
    with c_in2:
        st.markdown("**Productividad Forwarder**")
        prod_f_hr = st.number_input("Producción (m³ Sólidos/Hora)", value=28.0, step=0.5, key="p_f")
        # Detalle de Conversión
        prod_f_mr = prod_f_hr / st.session_state['conv_factor']
        st.markdown(f'<div class="mr-info">Equivale a: <b>{prod_f_mr:,.2f} MR/hr</b></div>', unsafe_allow_html=True)
        
    with c_in3:
        st.markdown("**Venta Proyectada**")
        ingresos_hr = prod_f_mr * st.session_state['sales_price']
        st.metric("Ingreso por Hora Operativa", fmt_money(ingresos_hr), f"Tarifa: {fmt_money(st.session_state['sales_price'])}/MR")

    st.divider()

    # 2. Cálculos Temporales
    costo_hr = cost_mensual_sistema / hrs_sistema_mes
    costo_sem = cost_mensual_sistema / 4
    costo_mes = cost_mensual_sistema
    
    ingresos_mes = ingresos_hr * hrs_sistema_mes
    ingreso_sem = ingresos_mes / 4
    
    utilidad_hr = ingresos_hr - costo_hr
    utilidad_sem = ingreso_sem - costo_sem
    utilidad_mes = ingresos_mes - costo_mes
    
    margen_pct_hr = (utilidad_hr / ingresos_hr * 100) if ingresos_hr > 0 else 0

    # 3. Tarjetas
    def render_kpi_card(title, inc, cost, prof):
        profit_class = "val-prof" if prof >= 0 else "val-loss"
        return f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-row"><span>Ingresos</span><span class="val-inc">{fmt_money(inc)}</span></div>
            <div class="kpi-row"><span>Costos</span><span class="val-cost">{fmt_money(cost)}</span></div>
            <hr style="margin:8px 0; border-color:#f1f5f9">
            <div class="kpi-row"><span style="font-weight:700">GANANCIA</span><span class="{profit_class}">{fmt_money(prof)}</span></div>
        </div>
        """

    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(render_kpi_card("⏱️ Por Hora", ingresos_hr, costo_hr, utilidad_hr), unsafe_allow_html=True)
    with col2: st.markdown(render_kpi_card("📅 Por Semana", ingreso_sem, costo_sem, utilidad_sem), unsafe_allow_html=True)
    with col3: st.markdown(render_kpi_card("🗓️ Por Mes", ingresos_mes, costo_mes, utilidad_mes), unsafe_allow_html=True)

    # 4. Gráfico con Porcentaje
    st.write("")
    st.subheader("Comparativa Financiera")
    
    df_chart = pd.DataFrame({
        "Periodo": ["Hora", "Hora", "Hora", "Semana", "Semana", "Semana", "Mes", "Mes", "Mes"],
        "Concepto": ["Ingreso", "Costo", "Ganancia"] * 3,
        "Monto": [ingresos_hr, costo_hr, utilidad_hr, ingreso_sem, costo_sem, utilidad_sem, ingresos_mes, costo_mes, utilidad_mes]
    })
    
    # Lógica de etiquetas: Si es Ganancia mostrar %, si no mostrar $
    df_chart['Etiqueta'] = df_chart.apply(
        lambda x: f"{margen_pct_hr:.1f}%" if x['Concepto'] == 'Ganancia' else fmt_money(x['Monto']), axis=1
    )
    
    colors = {"Ingreso": "#3b82f6", "Costo": "#ef4444", "Ganancia": "#22c55e"}
    
    fig = px.bar(df_chart, x="Concepto", y="Monto", color="Concepto", facet_col="Periodo",
                 color_discrete_map=colors, text="Etiqueta", 
                 title="Desempeño Financiero")
    
    fig.update_yaxes(matches=None, showticklabels=True) 
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: HARVESTER ---
with tab_h:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Jornada")
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 28)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 10.0)), key="h_hours", on_change=save_config)
        st.info(f"Total: {h_days*h_hours} Horas")
    with c2:
        st.subheader("Estructura de Costos (Editable)")
        # Tabla Editable
        edited_h = st.data_editor(
            st.session_state['df_harvester'], use_container_width=True, num_rows="dynamic", 
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", required=True), 
                "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"], required=True)
            }
        )
        st.session_state['df_harvester'] = edited_h
        save_config()
        
        # Calcular total de esta tabla en vivo
        total_h_live = calculate_single_machine_monthly_cost(edited_h, h_days, h_hours, curr_uf, curr_fuel, 'H')
        st.success(f"💰 Costo Mensual Total Harvester: **{fmt_money(total_h_live)}**")

# --- TAB 3: FORWARDER ---
with tab_f:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Jornada")
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 28)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 10.0)), key="f_hours", on_change=save_config)
        st.info(f"Total: {f_days*f_hours} Horas")
    with c2:
        st.subheader("Estructura de Costos (Editable)")
        edited_f = st.data_editor(
            st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic", 
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", required=True), 
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
            }
        )
        st.session_state['df_forwarder'] = edited_f
        save_config()
        
        total_f_live = calculate_single_machine_monthly_cost(edited_f, f_days, f_hours, curr_uf, curr_fuel, 'F')
        st.success(f"💰 Costo Mensual Total Forwarder: **{fmt_money(total_f_live)}**")

# --- TAB 4: INDIRECTOS ---
with tab_ind:
    c_rrhh, c_flota = st.columns(2)
    with c_rrhh:
        st.markdown("### 👷 RRHH Indirecto")
        edited_rrhh = st.data_editor(st.session_state['df_rrhh'], use_container_width=True, num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_rrhh'] = edited_rrhh
        st.info(f"Total RRHH: **{fmt_money(edited_rrhh['Costo Empresa'].sum())}**")
        
    with c_flota:
        st.markdown("### 🛻 Flota y Gastos Generales")
        edited_flota = st.data_editor(st.session_state['df_flota'], use_container_width=True, num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
        st.session_state['df_flota'] = edited_flota
        st.info(f"Total Flota: **{fmt_money(edited_flota['Monto'].sum())}**")
    save_config()

# --- TAB 5: SIMULADOR TARIFAS ---
with tab_sim:
    st.header("🎯 Calculadora de Tarifas y Márgenes")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        margin_h = st.slider("Margen Harvester (%)", 0, 60, int(st.session_state.get('target_margin_h', 35)))
        st.session_state['target_margin_h'] = margin_h
    with col_input2:
        margin_f = st.slider("Margen Forwarder (%)", 0, 60, int(st.session_state.get('target_margin_f', 35)))
        st.session_state['target_margin_f'] = margin_f
    with col_input3:
        prod_sim = st.number_input("Prod. Estimada (MR/Hr)", value=22.0, step=0.5)
        save_config()

    # Cálculos
    cost_h_hr_real = (tot_h_dir + ind_h) / hrs_h if hrs_h > 0 else 0
    cost_f_hr_real = (tot_f_dir + ind_f) / hrs_f if hrs_f > 0 else 0
    safe_prod = prod_sim if prod_sim > 0 else 1
    
    cost_unit_h = cost_h_hr_real / safe_prod
    cost_unit_f = cost_f_hr_real / safe_prod
    cost_unit_sys = cost_unit_h + cost_unit_f

    # Análisis Rango
    st.divider()
    st.subheader("📊 Análisis de Rango Objetivo (30% - 35%)")
    
    p_h_30 = calc_price(cost_unit_h, 30)
    p_f_30 = calc_price(cost_unit_f, 30)
    p_sys_30 = p_h_30 + p_f_30

    p_h_35 = calc_price(cost_unit_h, 35)
    p_f_35 = calc_price(cost_unit_f, 35)
    p_sys_35 = p_h_35 + p_f_35

    col_30, col_35 = st.columns(2)
    with col_30:
        st.markdown(f"""
        <div class="highlight-box" style="border-left-color:#fcd34d; background-color:#fffbeb;">
            <div style="color:#b45309; font-weight:bold">Escenario Base (30%)</div>
            <div style="font-size:1.5em; font-weight:800; color:#b45309">{fmt_money(p_sys_30)} / MR</div>
            <div style="font-size:0.9em; text-align:left; margin-top:10px">H: {fmt_money(p_h_30)} | F: {fmt_money(p_f_30)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_35:
        st.markdown(f"""
        <div class="highlight-box" style="border-left-color:#86efac; background-color:#f0fdf4;">
            <div style="color:#15803d; font-weight:bold">Escenario Ideal (35%)</div>
            <div style="font-size:1.5em; font-weight:800; color:#15803d">{fmt_money(p_sys_35)} / MR</div>
            <div style="font-size:0.9em; text-align:left; margin-top:10px">H: {fmt_money(p_h_35)} | F: {fmt_money(p_f_35)}</div>
        </div>
        """, unsafe_allow_html=True)
