import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

# --- 1. CONFIGURACIÓN Y ESTILO CORPORATIVO ---
st.set_page_config(
    page_title="Forestal Costing Pro", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

# CSS para ocultar índices de tablas y mejorar métricas
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
    .big-number { font-size: 2em; font-weight: 800; color: #15803d; }
    .label-text { font-size: 0.9em; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .sub-text { font-size: 0.85em; color: #94a3b8; }
    
    /* Ocultar índice de las tablas */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v12_final.json'

# --- 2. FUNCIONES GLOBALES (LÓGICA DE NEGOCIO) ---

def fmt_money(x): 
    """Formatea números como moneda CLP"""
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    """Precio = Costo / (1 - Margen%)"""
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

def calculate_system_costs(h_df, f_df, rrhh_df, flota_df, days_h, hrs_h, days_f, hrs_f, uf, diesel):
    # Pre-procesamiento para evitar errores de celdas vacías
    h_df = h_df.fillna(0)
    f_df = f_df.fillna(0)
    rrhh_df = rrhh_df.fillna(0)
    flota_df = flota_df.fillna(0)

    # 1. Harvester
    total_h = 0
    total_h_hrs = days_h * hrs_h
    for _, row in h_df.iterrows():
        val = float(row.get('Valor', 0))
        tipo = row.get('Tipo', '$/Mes')
        frec = float(row.get('Frec', 1))
        
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf
        elif tipo == 'Litros/Día': cost = val * days_h * diesel
        elif tipo == '$/Ev': 
            if frec > 0 and total_h_hrs > 0: 
                cost = (val / frec) * total_h_hrs
        total_h += cost

    # 2. Forwarder
    total_f = 0
    total_f_hrs = days_f * hrs_f
    for _, row in f_df.iterrows():
        val = float(row.get('Valor', 0))
        tipo = row.get('Unidad', '$/Mes') # Ojo: en Forwarder usamos 'Unidad'
        
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'Litros/Día': cost = val * days_f * diesel
        total_f += cost

    # 3. Indirectos
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect, total_h_hrs, total_f_hrs

@st.cache_data(ttl=3600) 
def get_uf_api():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=2) # Timeout corto para no bloquear
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

# --- 3. GESTIÓN DE ESTADO (INICIALIZACIÓN) ---
saved = load_config()

def init_key(key, default_value):
    if key not in st.session_state:
        loaded_val = saved.get(key)
        if loaded_val is not None:
            if isinstance(default_value, pd.DataFrame):
                st.session_state[key] = pd.DataFrame(loaded_val)
            else:
                st.session_state[key] = loaded_val
        else:
            st.session_state[key] = default_value

# Valores por defecto (Basados en tu Excel)
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

# DataFrames Base
init_key('df_harvester', pd.DataFrame([
    {"Cat": "Fijos", "Ítem": "Arriendo Base", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
    {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Insumos", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frec": 1, "Valor": 450000},
    {"Cat": "Mantención", "Ítem": "Mant. 600h", "Tipo": "$/Ev", "Frec": 600, "Valor": 350000},
    {"Cat": "Mayor", "Ítem": "Overhaul (Amort)", "Tipo": "$/Ev", "Frec": 20000, "Valor": 24000000},
]))

init_key('df_forwarder', pd.DataFrame([
    {"Cat": "Operación", "Ítem": "Arriendo", "Unidad": "$/Mes", "Valor": 8000000},
    {"Cat": "Operación", "Ítem": "Operador", "Unidad": "$/Mes", "Valor": 1900000},
    {"Cat": "Variable", "Ítem": "Petróleo", "Unidad": "Litros/Día", "Valor": 135.0},
    {"Cat": "Mantención", "Ítem": "Mantención Gral", "Unidad": "$/Mes", "Valor": 1500000},
    {"Cat": "Variable", "Ítem": "Neumáticos", "Unidad": "$/Mes", "Valor": 400000},
]))

init_key('df_rrhh', pd.DataFrame([
    {"Cargo": "Jefe de Faena", "Sueldo Líquido": 1800000, "Costo Empresa": 2300000},
    {"Cargo": "Mecánico", "Sueldo Líquido": 1200000, "Costo Empresa": 1600000},
    {"Cargo": "Prevencionista", "Sueldo Líquido": 900000, "Costo Empresa": 1200000},
]))

init_key('df_flota', pd.DataFrame([
    {"Ítem": "Camionetas (Arriendo)", "Monto": 1600000},
    {"Ítem": "Combustible Apoyo", "Monto": 600000},
    {"Ítem": "Pensión/Alojamiento", "Monto": 1500000},
    {"Ítem": "Gastos Adm. Central", "Monto": 500000},
]))

# --- 4. SIDEBAR CONFIGURACIÓN ---
with st.sidebar:
    # Header con logo o ícono
    st.markdown("## ⚙️ Panel de Control")
    
    # 1. Botón de Pánico (Reset)
    if st.button("♻️ Resetear Valores Fábrica", type="secondary", help="Borra la configuración y restaura los valores por defecto"):
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    # 2. Mercado
    with st.expander("💰 Mercado y Tarifas", expanded=True):
        use_api = st.checkbox("UF Automática (API)", value=True)
        uf_api_val, _ = get_uf_api()
        
        val_uf_display = uf_api_val if (use_api and uf_api_val) else st.session_state['uf_manual']
        
        if use_api and uf_api_val:
            st.success(f"UF Conectada: ${uf_api_val:,.2f}")
        
        curr_uf = st.number_input("Valor UF ($)", value=float(val_uf_display), disabled=(use_api and uf_api_val is not None))
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()

        curr_fuel = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config, key="fuel_price")
        curr_sales = st.number_input("Tarifa Venta ($/MR)", value=float(st.session_state['sales_price']), on_change=save_config, key="sales_price")

    # 3. Conversión
    with st.expander("📏 Conversión Volumen"):
        curr_factor = st.number_input("Factor (m³/MR)", value=float(st.session_state.get('conv_factor', 2.44)), step=0.01, format="%.2f", key="conv_factor_input")
        if curr_factor != st.session_state['conv_factor']:
            st.session_state['conv_factor'] = curr_factor
            save_config()

    # 4. Distribución
    with st.expander("⚖️ Distribución Indirectos"):
        alloc = st.slider("% Carga a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()
        st.caption(f"Harvester: {alloc*100:.0f}% | Forwarder: {(1-alloc)*100:.0f}%")

# --- 5. CÁLCULOS DEL SISTEMA (EJECUCIÓN) ---
tot_h_dir, tot_f_dir, tot_ind, hrs_h, hrs_f = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    int(st.session_state['h_days']), float(st.session_state['h_hours']), 
    int(st.session_state['f_days']), float(st.session_state['f_hours']), 
    curr_uf, curr_fuel
)

# Prorrateo
ind_h = tot_ind * st.session_state['alloc_pct']
ind_f = tot_ind * (1 - st.session_state['alloc_pct'])

# Totales Mensuales
final_h_mes = tot_h_dir + ind_h
final_f_mes = tot_f_dir + ind_f
cost_mensual_sistema = final_h_mes + final_f_mes

# Costos por Hora (Evitando div/0)
cost_hr_h = final_h_mes / hrs_h if hrs_h > 0 else 0
cost_hr_f = final_f_mes / hrs_f if hrs_f > 0 else 0
# Costo Hora Sistema (Asumiendo operación paralela, sumamos costo hora de c/u)
cost_hr_sys = cost_hr_h + cost_hr_f

# --- 6. INTERFAZ PRINCIPAL ---
st.title("🌲 Sistema de Costos Forestales Profesional")

tab_dash, tab_h, tab_f, tab_ind, tab_sim = st.tabs([
    "📊 Dashboard Gerencial", 
    "🚜 Harvester", 
    "🚜 Forwarder", 
    "👷 Indirectos (RRHH/Flota)", 
    "📈 Simulador de Tarifas"
])

# --- TAB 1: DASHBOARD ---
with tab_dash:
    st.subheader("Simulador de Cierre Mensual")
    
    # Inputs Producción
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        prod_h_m3 = st.number_input("Producción H (m³ Sólidos)", value=5000.0, step=100.0)
        st.metric("Equivalente MR", f"{prod_h_m3 / st.session_state['conv_factor']:,.1f} MR")
    with c_in2:
        prod_f_m3 = st.number_input("Producción F (m³ Sólidos)", value=5000.0, step=100.0)
        prod_f_mr = prod_f_m3 / st.session_state['conv_factor']
        st.metric("Equivalente MR", f"{prod_f_mr:,.1f} MR", help="Producción Real en Cancha")
    with c_in3:
        st.info(f"Factor Conversión: **{st.session_state['conv_factor']} m³/MR**")
        st.caption("Cálculos financieros basados en producción Forwarder (Cancha)")

    st.divider()

    # Resultados
    ingresos_reales = prod_f_mr * st.session_state['sales_price']
    utilidad = ingresos_reales - cost_mensual_sistema
    margen_pct = (utilidad / ingresos_reales * 100) if ingresos_reales > 0 else 0
    
    # Target promedio para colores
    target_avg = (st.session_state.get('target_margin_h', 35.0) + st.session_state.get('target_margin_f', 35.0)) / 2

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Facturación Estimada", fmt_money(ingresos_reales), "Ingreso Bruto")
    k2.metric("Costo Total Mes", fmt_money(cost_mensual_sistema), "Operativo + Fijo")
    k3.metric("Utilidad Neta", fmt_money(utilidad), f"{margen_pct:.1f}% Margen", 
              delta_color="normal" if margen_pct >= target_avg else "inverse")

    # Gráfico Waterfall (Cascada de Rentabilidad)
    st.subheader("Cascada de Rentabilidad")
    fig_water = go.Figure(go.Waterfall(
        name = "Finanzas", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Ventas Totales", "Costo Harvester", "Costo Forwarder", "Costo Indirecto", "UTILIDAD FINAL"],
        textposition = "outside",
        text = [fmt_money(ingresos_reales), fmt_money(-final_h_mes), fmt_money(-final_f_mes), fmt_money(-tot_ind), fmt_money(utilidad)],
        y = [ingresos_reales, -tot_h_dir, -tot_f_dir, -tot_ind, utilidad],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ef4444"}}, # Rojo para costos
        increasing = {"marker":{"color":"#22c55e"}}, # Verde para ingresos/utilidad
        totals = {"marker":{"color":"#15803d"}}
    ))
    fig_water.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_water, use_container_width=True)

# --- TAB 2: HARVESTER ---
with tab_h:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Jornada")
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 28)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 10.0)), key="h_hours", on_change=save_config)
        st.info(f"Total: {h_days*h_hours} Horas")
    with c2:
        st.subheader("Estructura de Costos")
        st.session_state['df_harvester'] = st.data_editor(
            st.session_state['df_harvester'], 
            use_container_width=True, 
            num_rows="dynamic", 
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", required=True), 
                "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"], required=True),
                "Frec": st.column_config.NumberColumn(label="Frecuencia (Hrs)", help="Solo para $/Ev")
            }
        )
        save_config()

# --- TAB 3: FORWARDER ---
with tab_f:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Jornada")
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 28)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 10.0)), key="f_hours", on_change=save_config)
        st.info(f"Total: {f_days*f_hours} Horas")
    with c2:
        st.subheader("Estructura de Costos")
        st.session_state['df_forwarder'] = st.data_editor(
            st.session_state['df_forwarder'], 
            use_container_width=True, 
            num_rows="dynamic", 
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d", required=True), 
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
            }
        )
        save_config()

# --- TAB 4: INDIRECTOS ---
with tab_ind:
    c_rrhh, c_flota = st.columns(2)
    with c_rrhh:
        st.markdown("### 👷 RRHH Indirecto")
        st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], use_container_width=True, num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c_flota:
        st.markdown("### 🛻 Flota y Gastos Generales")
        st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], use_container_width=True, num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()

# --- TAB 5: SIMULADOR TARIFAS ---
with tab_sim:
    st.header("🎯 Calculadora de Tarifas y Márgenes")
    
    # Inputs Simulación
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

    # Cálculos de Costo Unitario (Base Hora / Prod)
    cost_h_hr_real = (tot_h_dir + ind_h) / hrs_h if hrs_h > 0 else 0
    cost_f_hr_real = (tot_f_dir + ind_f) / hrs_f if hrs_f > 0 else 0
    
    # Evitar división por cero si prod_sim es 0
    safe_prod = prod_sim if prod_sim > 0 else 1
    
    cost_unit_h = cost_h_hr_real / safe_prod
    cost_unit_f = cost_f_hr_real / safe_prod
    cost_unit_sys = cost_unit_h + cost_unit_f

    # --- ANÁLISIS DE ESCENARIOS ---
    st.divider()
    st.subheader("📊 Comparativa de Escenarios de Tarifas")

    # Escenario 30%
    p_h_30 = calc_price(cost_unit_h, 30)
    p_f_30 = calc_price(cost_unit_f, 30)
    p_sys_30 = p_h_30 + p_f_30

    # Escenario 35%
    p_h_35 = calc_price(cost_unit_h, 35)
    p_f_35 = calc_price(cost_unit_f, 35)
    p_sys_35 = p_h_35 + p_f_35

    col_30, col_35 = st.columns(2)

    with col_30:
        st.markdown(f"""
        <div class="range-card" style="border-color: #fcd34d; background-color: #fffbeb;">
            <div class="range-title" style="color: #b45309;">Escenario Base (30% Margen)</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #b45309;">{fmt_money(p_sys_30)} / MR</div>
            <hr>
            <div style="text-align: left; padding-left: 20px;">
                <span class="label-text">Harvester:</span> {fmt_money(p_h_30)}<br>
                <span class="label-text">Forwarder:</span> {fmt_money(p_f_30)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_35:
        st.markdown(f"""
        <div class="range-card" style="border-color: #86efac; background-color: #f0fdf4;">
            <div class="range-title" style="color: #15803d;">Escenario Ideal (35% Margen)</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #15803d;">{fmt_money(p_sys_35)} / MR</div>
            <hr>
            <div style="text-align: left; padding-left: 20px;">
                <span class="label-text">Harvester:</span> {fmt_money(p_h_35)}<br>
                <span class="label-text">Forwarder:</span> {fmt_money(p_f_35)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- DETALLE PERSONALIZADO ---
    st.divider()
    st.subheader(f"🎛️ Detalle Simulación Manual ({margin_h}% H / {margin_f}% F)")
    
    price_h_sim = calc_price(cost_unit_h, margin_h)
    price_f_sim = calc_price(cost_unit_f, margin_f)
    profit_h = price_h_sim - cost_unit_h
    profit_f = price_f_sim - cost_unit_f
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.markdown(f'<div class="highlight-box"><span class="label-text">Harvester</span><br><span class="big-number">{fmt_money(profit_h)}</span><br><span class="sub-text">Utilidad / MR</span></div>', unsafe_allow_html=True)
    with col_d2:
        st.markdown(f'<div class="highlight-box"><span class="label-text">Forwarder</span><br><span class="big-number">{fmt_money(profit_f)}</span><br><span class="sub-text">Utilidad / MR</span></div>', unsafe_allow_html=True)
    with col_d3:
        st.markdown(f'<div class="highlight-box" style="border-left-color: #1d4ed8;"><span class="label-text">TOTAL SISTEMA</span><br><span class="big-number" style="color:#1d4ed8;">{fmt_money(profit_h+profit_f)}</span><br><span class="sub-text">Utilidad / MR</span></div>', unsafe_allow_html=True)

    # Tabla Final
    st.write("")
    df_detail = pd.DataFrame({
        "Concepto": ["Harvester", "Forwarder", "SISTEMA TOTAL"],
        "Costo Unitario": [fmt_money(cost_unit_h), fmt_money(cost_unit_f), fmt_money(cost_unit_sys)],
        "Tarifa Sugerida": [fmt_money(price_h_sim), fmt_money(price_f_sim), fmt_money(price_h_sim + price_f_sim)],
        "Margen %": [f"{margin_h}%", f"{margin_f}%", "-"]
    })
    st.dataframe(df_detail, use_container_width=True, hide_index=True)
