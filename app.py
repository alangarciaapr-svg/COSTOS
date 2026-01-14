import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import io
import requests

# --- 1. CONFIGURACIÓN Y ESTILO PRO ---
st.set_page_config(
    page_title="Forestal Costing Master", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    h1, h2, h3 {color: #1e293b; font-family: 'Segoe UI', sans-serif;}
    
    /* Estilo de Métricas Nativas Mejorado */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    /* Tarjetas Personalizadas */
    .machine-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-top: 4px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .card-h { border-top-color: #eab308; } /* Amarillo */
    .card-f { border-top-color: #22c55e; } /* Verde */
    
    .card-header {
        font-size: 1.1em;
        font-weight: 800;
        color: #334155;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px dashed #f1f5f9;
        font-size: 0.95em;
    }
    
    .total-row {
        display: flex;
        justify-content: space-between;
        padding-top: 12px;
        margin-top: 5px;
        font-weight: 800;
        font-size: 1.1em;
        color: #0f172a;
    }
    
    /* Badge de Estado */
    .badge-success { background-color: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }

    /* Ocultar índices */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v23_pro_goals.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
    df = df.fillna(0)
    total = 0
    total_hours = days * hours
    for _, row in df.iterrows():
        val = float(row.get('Valor', 0))
        tipo = row.get('Tipo') if machine_type == 'H' else row.get('Unidad')
        if not tipo: tipo = '$/Mes'
        frec = float(row.get('Frec', 1))
        
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf
        elif tipo == 'Litros/Día': cost = val * days * diesel
        elif tipo == '$/Ev': 
            if frec > 0 and total_hours > 0: cost = (val / frec) * total_hours
        total += cost
    return total

def calculate_system_costs(h_df, f_df, rrhh_df, flota_df, h_days, h_hours, f_days, f_hours, uf, diesel):
    total_h = calculate_single_machine_monthly_cost(h_df, h_days, h_hours, uf, diesel, 'H')
    total_f = calculate_single_machine_monthly_cost(f_df, f_days, f_hours, uf, diesel, 'F')
    
    rrhh_df = rrhh_df.fillna(0)
    flota_df = flota_df.fillna(0)
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect

# --- FUNCIÓN EXPORTAR EXCEL ---
def to_excel(state):
    output = io.BytesIO()
    # Usamos el motor por defecto o xlsxwriter si está disponible
    try:
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    except:
        writer = pd.ExcelWriter(output)
    
    params = [
        {"Parámetro": "Valor UF", "Valor": state['uf_manual']},
        {"Parámetro": "Precio Diesel", "Valor": state['fuel_price']},
        {"Parámetro": "Factor Conversión", "Valor": state['conv_factor']},
        {"Parámetro": "Meta Margen %", "Valor": state['target_company_margin']},
        {"Parámetro": "Tarifa H", "Valor": state['price_h']},
        {"Parámetro": "Tarifa F", "Valor": state['price_f']},
    ]
    pd.DataFrame(params).to_excel(writer, sheet_name='Configuracion', index=False)
    state['df_harvester'].to_excel(writer, sheet_name='Costos_Harvester', index=False)
    state['df_forwarder'].to_excel(writer, sheet_name='Costos_Forwarder', index=False)
    state['df_rrhh'].to_excel(writer, sheet_name='RRHH', index=False)
    state['df_flota'].to_excel(writer, sheet_name='Indirectos', index=False)
    
    writer.close()
    return output.getvalue()

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
            "alloc_pct", "price_h", "price_f", "conv_factor", "target_company_margin"]
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
init_key('price_h', 6500.0)
init_key('price_f', 5000.0)
init_key('alloc_pct', 0.5)
init_key('conv_factor', 2.44)
init_key('target_company_margin', 30.0) # NUEVO: Objetivo Gerencial
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
    
    excel_data = to_excel(st.session_state)
    st.download_button("📥 Descargar Reporte Excel", data=excel_data, file_name='Reporte_Gestion_Forestal.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type="primary")
    
    if st.button("♻️ Reset App"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    # NUEVO: OBJETIVOS
    with st.expander("🎯 Objetivos Estratégicos", expanded=True):
        st.session_state['target_company_margin'] = st.slider("Margen Objetivo Empresa (%)", 0, 60, int(st.session_state['target_company_margin']), on_change=save_config)
        st.caption("Este valor definirá si tus resultados están en Verde o Rojo.")

    with st.expander("🕒 Jornada Laboral", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state['h_days'] = c1.number_input("Días H", value=int(st.session_state['h_days']), on_change=save_config)
        st.session_state['h_hours'] = c2.number_input("Horas H", value=float(st.session_state['h_hours']), on_change=save_config)
        c3, c4 = st.columns(2)
        st.session_state['f_days'] = c3.number_input("Días F", value=int(st.session_state['f_days']), on_change=save_config)
        st.session_state['f_hours'] = c4.number_input("Horas F", value=float(st.session_state['f_hours']), on_change=save_config)

    with st.expander("💰 Mercado y Tarifas"):
        st.session_state['price_h'] = st.number_input("Tarifa H ($/MR)", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Tarifa F ($/MR)", value=float(st.session_state['price_f']), on_change=save_config)
        st.session_state['fuel_price'] = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config)
        
    with st.expander("📏 Conversión y Distribución"):
        st.session_state['conv_factor'] = st.number_input("Factor m³/MR", value=float(st.session_state['conv_factor']), step=0.01, on_change=save_config)
        alloc = st.slider("% Indirectos a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()

# --- 5. MOTOR DE CÁLCULO ---
h_dias = int(st.session_state['h_days'])
h_horas = float(st.session_state['h_hours'])
f_dias = int(st.session_state['f_days'])
f_horas = float(st.session_state['f_hours'])

tot_h_dir, tot_f_dir, tot_ind = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    h_dias, h_horas, f_dias, f_horas, 
    st.session_state.get('uf_manual', 0), st.session_state['fuel_price']
)

cost_h_total_mes = tot_h_dir + (tot_ind * st.session_state['alloc_pct'])
cost_f_total_mes = tot_f_dir + (tot_ind * (1 - st.session_state['alloc_pct']))

# --- 6. INTERFAZ ---
st.title("🌲 Dashboard de Control de Gestión Forestal")

tab_dash, tab_strat, tab_h, tab_f, tab_ind = st.tabs([
    "📊 Resultado Operacional", "🎯 Estrategia Precios", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos"
])

# --- TAB DASHBOARD ---
with tab_dash:
    # 1. INPUTS
    st.subheader("1. Variables de Producción")
    c1, c2, c3 = st.columns(3)
    with c1:
        prod_h_m3 = st.number_input("Prod. Harvester (m³/hr)", value=25.0, step=0.5)
        mr_h_hr = prod_h_m3 / st.session_state['conv_factor']
        st.info(f"H: {mr_h_hr:,.1f} MR/hr")
    with c2:
        prod_f_m3 = st.number_input("Prod. Forwarder (m³/hr)", value=28.0, step=0.5)
        mr_f_hr = prod_f_m3 / st.session_state['conv_factor']
        st.info(f"F: {mr_f_hr:,.1f} MR/hr")
    with c3:
        target_pct = st.session_state['target_company_margin']
        st.metric("Meta de Margen", f"{target_pct}%")

    st.divider()

    # 2. CÁLCULOS PROFUNDOS
    # Harvester
    inc_h_hr = mr_h_hr * st.session_state['price_h']
    inc_h_mes = inc_h_hr * h_horas * h_dias
    inc_h_day = inc_h_hr * h_horas
    
    cost_h_mes = cost_h_total_mes
    cost_h_day = cost_h_mes / h_dias if h_dias > 0 else 0
    cost_h_hr = cost_h_mes / (h_dias * h_horas) if (h_dias*h_horas) > 0 else 0
    
    prof_h_mes = inc_h_mes - cost_h_mes
    margin_h = (prof_h_mes / inc_h_mes * 100) if inc_h_mes > 0 else 0

    # Forwarder
    inc_f_hr = mr_f_hr * st.session_state['price_f']
    inc_f_mes = inc_f_hr * f_horas * f_dias
    inc_f_day = inc_f_hr * f_horas
    
    cost_f_mes = cost_f_total_mes
    cost_f_day = cost_f_mes / f_dias if f_dias > 0 else 0
    cost_f_hr = cost_f_mes / (f_dias * f_horas) if (f_dias*f_horas) > 0 else 0
    
    prof_f_mes = inc_f_mes - cost_f_mes
    margin_f = (prof_f_mes / inc_f_mes * 100) if inc_f_mes > 0 else 0

    # Consolidado
    prof_total_mes = prof_h_mes + prof_f_mes
    inc_total_mes = inc_h_mes + inc_f_mes
    margin_total = (prof_total_mes / inc_total_mes * 100) if inc_total_mes > 0 else 0
    
    # Delta vs Objetivo
    delta_prof_h = margin_h - target_pct
    delta_prof_f = margin_f - target_pct
    delta_prof_total = margin_total - target_pct

    # 3. VISUALIZACIÓN EJECUTIVA
    st.subheader("2. Estado de Resultados (Vs Meta)")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Utilidad Harvester (Mes)", fmt_money(prof_h_mes), f"{delta_prof_h:.1f}% vs Meta")
    col_kpi2.metric("Utilidad Forwarder (Mes)", fmt_money(prof_f_mes), f"{delta_prof_f:.1f}% vs Meta")
    col_kpi3.metric("UTILIDAD TOTAL (Mes)", fmt_money(prof_total_mes), f"{delta_prof_total:.1f}% vs Meta", delta_color="normal")

    c_card1, c_card2 = st.columns(2)
    
    # FUNCION DE RENDERIZADO DE TARJETAS
    def render_pro_card(title, df, margin, target, color_border):
        badge = "badge-success" if margin >= target else "badge-danger"
        badge_text = "CUMPLE META" if margin >= target else "BAJO META"
        
        # Convertir DF a HTML simple para control total
        html_table = "<table style='width:100%; border-collapse: collapse; margin-top:10px;'>"
        html_table += "<tr style='color:#64748b; border-bottom:1px solid #e2e8f0; font-size:0.9em;'><th style='text-align:left'>Periodo</th><th style='text-align:right'>Generado</th><th style='text-align:right'>Costo</th><th style='text-align:right'>Ganancia</th></tr>"
        
        for _, row in df.iterrows():
            html_table += f"<tr><td style='padding:6px 0; color:#0f172a;'>{row['Periodo']}</td><td style='text-align:right; font-weight:600;'>{row['Generado']}</td><td style='text-align:right; color:#ef4444;'>{row['Costo']}</td><td style='text-align:right; font-weight:700; color:{'#16a34a' if '$-' not in row['Ganancia'] else '#dc2626'}'>{row['Ganancia']}</td></tr>"
        html_table += "</table>"

        return f"""
        <div class="machine-card" style="border-top-color: {color_border}">
            <div class="card-header">
                <span>{title}</span>
                <span class="{badge}">{badge_text}</span>
            </div>
            {html_table}
            <div class="total-row">
                <span>Margen Real:</span>
                <span style="color: {'#16a34a' if margin >= target else '#dc2626'}">{margin:.1f}%</span>
            </div>
        </div>
        """

    # DATA FRAMES PARA DISPLAY
    df_h_display = pd.DataFrame([
        {"Periodo": "Hora", "Generado": fmt_money(inc_h_hr), "Costo": fmt_money(cost_h_hr), "Ganancia": fmt_money(inc_h_hr - cost_h_hr)},
        {"Periodo": "Día", "Generado": fmt_money(inc_h_day), "Costo": fmt_money(cost_h_day), "Ganancia": fmt_money(inc_h_day - cost_h_day)},
        {"Periodo": "Mes", "Generado": fmt_money(inc_h_mes), "Costo": fmt_money(cost_h_mes), "Ganancia": fmt_money(prof_h_mes)},
    ])
    
    df_f_display = pd.DataFrame([
        {"Periodo": "Hora", "Generado": fmt_money(inc_f_hr), "Costo": fmt_money(cost_f_hr), "Ganancia": fmt_money(inc_f_hr - cost_f_hr)},
        {"Periodo": "Día", "Generado": fmt_money(inc_f_day), "Costo": fmt_money(cost_f_day), "Ganancia": fmt_money(inc_f_day - cost_f_day)},
        {"Periodo": "Mes", "Generado": fmt_money(inc_f_mes), "Costo": fmt_money(cost_f_mes), "Ganancia": fmt_money(prof_f_mes)},
    ])

    with c_card1:
        st.markdown(render_pro_card("🚜 HARVESTER", df_h_display, margin_h, target_pct, "#eab308"), unsafe_allow_html=True)
        
    with c_card2:
        st.markdown(render_pro_card("🚜 FORWARDER", df_f_display, margin_f, target_pct, "#22c55e"), unsafe_allow_html=True)

    # GRÁFICO ESTRUCTURA COSTOS
    st.subheader("3. Estructura de Costos del Sistema")
    c_pie1, c_pie2 = st.columns([1, 2])
    with c_pie1:
        st.caption("Distribución del Gasto Mensual")
        labels = ['Harvester Directo', 'Forwarder Directo', 'Indirectos']
        values = [tot_h_dir, tot_f_dir, tot_ind]
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig_donut, use_container_width=True)
    with c_pie2:
        st.info("💡 **Análisis Rápido:** Si los costos indirectos superan el 25% del total, revisa la estructura de flota y administración.")

# --- TAB 2: ESTRATEGIA PRECIOS ---
with tab_strat:
    st.header("🎯 Simulador de Tarifas (30% vs 35%)")
    prod_cotiza = st.number_input("Productividad Estimada para Cotización (m³/hr)", value=25.0, step=0.5)
    mr_cotiza = prod_cotiza / st.session_state['conv_factor']
    
    # Costo Unitario del Sistema (Full Costing)
    cost_hr_sys_h = cost_h_total_mes / (h_dias * h_horas) if (h_dias*h_horas) > 0 else 0
    cost_hr_sys_f = cost_f_total_mes / (f_dias * f_horas) if (f_dias*f_horas) > 0 else 0
    
    cost_unit_h = cost_hr_sys_h / mr_cotiza if mr_cotiza > 0 else 0
    cost_unit_f = cost_hr_sys_f / mr_cotiza if mr_cotiza > 0 else 0
    
    # Cálculo Tarifas
    def get_tariffs(margin):
        th = calc_price(cost_unit_h, margin)
        tf = calc_price(cost_unit_f, margin)
        return th, tf, th+tf

    th30, tf30, sys30 = get_tariffs(30)
    th35, tf35, sys35 = get_tariffs(35)
    
    st.divider()
    c30, c35 = st.columns(2)
    
    with c30:
        st.markdown(f"""
        <div class="highlight-box" style="border-left-color:#fcd34d; background-color:#fffbeb;">
            <div style="color:#b45309; font-weight:bold">Escenario Base (30% Margen)</div>
            <div style="font-size:1.5em; font-weight:800; color:#b45309">{fmt_money(sys30)} / MR</div>
            <hr style="border-color:#fcd34d;">
            <div style="display:flex; justify-content:space-between; font-size:0.9em">
                <span><b>H:</b> {fmt_money(th30)}</span>
                <span><b>F:</b> {fmt_money(tf30)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c35:
        st.markdown(f"""
        <div class="highlight-box" style="border-left-color:#16a34a; background-color:#f0fdf4;">
            <div style="color:#15803d; font-weight:bold">Escenario Ideal (35% Margen)</div>
            <div style="font-size:1.5em; font-weight:800; color:#15803d">{fmt_money(sys35)} / MR</div>
            <hr style="border-color:#16a34a;">
            <div style="display:flex; justify-content:space-between; font-size:0.9em">
                <span><b>H:</b> {fmt_money(th35)}</span>
                <span><b>F:</b> {fmt_money(tf35)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- TABS EDICIÓN ---
with tab_h:
    st.subheader("Estructura de Costos Harvester")
    edited_h = st.data_editor(st.session_state['df_harvester'], num_rows="dynamic", use_container_width=True, column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_harvester'] = edited_h
    save_config()
    st.info(f"Total Directo: {fmt_money(calculate_single_machine_monthly_cost(edited_h, h_dias, h_horas, st.session_state.get('uf_manual',0), st.session_state['fuel_price'], 'H'))}")

with tab_f:
    st.subheader("Estructura de Costos Forwarder")
    edited_f = st.data_editor(st.session_state['df_forwarder'], num_rows="dynamic", use_container_width=True, column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_forwarder'] = edited_f
    save_config()
    st.info(f"Total Directo: {fmt_money(calculate_single_machine_monthly_cost(edited_f, f_dias, f_horas, st.session_state.get('uf_manual',0), st.session_state['fuel_price'], 'F'))}")

with tab_ind:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RRHH**")
        st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c2:
        st.markdown("**Flota**")
        st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()
    st.success(f"Total Indirectos: {fmt_money(st.session_state['df_rrhh']['Costo Empresa'].sum() + st.session_state['df_flota']['Monto'].sum())}")
