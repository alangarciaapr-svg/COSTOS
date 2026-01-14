import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
from datetime import datetime

# Intentamos importar FPDF
try:
    from fpdf import FPDF
except ImportError:
    st.error("⚠️ Librería fpdf no instalada. Agrega 'fpdf' a requirements.txt")

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(
    page_title="Forestal Costing Master", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f1f5f9;}
    h1, h2, h3 {color: #0f172a; font-family: 'Segoe UI', sans-serif;}
    
    /* Estilos de Métricas */
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; color: #0f172a; }
    div[data-testid="stMetricLabel"] { font-size: 1rem; font-weight: 600; color: #64748b; }
    
    /* Tarjetas Profesionales */
    .machine-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 5px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card-h { border-top-color: #eab308; } 
    .card-f { border-top-color: #22c55e; } 
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        font-weight: 800;
        font-size: 1.1em;
        color: #334155;
        text-transform: uppercase;
    }
    
    /* Badges de Estado */
    .badge-ok { background-color: #dcfce7; color: #15803d; padding: 4px 8px; border-radius: 6px; font-size: 0.75em; }
    .badge-bad { background-color: #fee2e2; color: #b91c1c; padding: 4px 8px; border-radius: 6px; font-size: 0.75em; }
    
    .total-row {
        margin-top: 15px;
        padding-top: 10px;
        border-top: 2px solid #f1f5f9;
        display: flex;
        justify-content: space-between;
        font-weight: 800;
        font-size: 1.1em;
        color: #0f172a;
    }
    
    .faena-card {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
    }

    /* Ocultar índices de tablas nativas */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v31_totals.json'

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

# --- 3. MOTOR PDF PROFESIONAL ---
class PDF_Pro(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 45, 'F')
        self.set_font('Arial', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 12)
        self.cell(0, 10, 'SOCIEDAD MADERERA GÁLVEZ Y DI GÉNOVA LDS', 0, 1, 'L')
        self.set_font('Arial', '', 11)
        self.set_text_color(203, 213, 225)
        self.cell(0, 5, 'INFORME DE GESTIÓN OPERACIONAL', 0, 1, 'L')
        self.set_xy(150, 15)
        self.set_font('Arial', 'B', 10)
        self.cell(50, 10, datetime.now().strftime('%d/%m/%Y'), 0, 1, 'R')
        self.ln(25)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_fill_color(241, 245, 249)
        self.set_text_color(30, 41, 59)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f"  {label}", 0, 1, 'L', 1)
        self.ln(5)

    def draw_financial_bar(self, label, income, cost, y_pos):
        self.set_xy(10, y_pos)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(50, 50, 50)
        self.cell(30, 6, label, 0, 0)
        
        max_width = 100
        scale = max_width / (income * 1.2) if income > 0 else 1
        w_inc = income * scale
        w_cost = cost * scale
        
        self.set_fill_color(59, 130, 246)
        self.rect(45, y_pos, w_inc, 5, 'F')
        self.set_fill_color(239, 68, 68)
        self.rect(45, y_pos + 6, w_cost, 5, 'F')
        
        self.set_xy(45 + w_inc + 2, y_pos - 1)
        self.set_font('Arial', '', 8)
        self.cell(30, 6, f"Venta: {fmt_money(income)}", 0, 1)
        
        self.set_xy(45 + w_cost + 2, y_pos + 5)
        self.cell(30, 6, f"Costo: {fmt_money(cost)}", 0, 1)
        
        profit = income - cost
        margin = (profit / income * 100) if income > 0 else 0
        self.set_xy(160, y_pos + 2)
        self.set_font('Arial', 'B', 10)
        if profit >= 0:
            self.set_text_color(22, 163, 74)
            self.cell(30, 6, f"+ {margin:.1f}%", 0, 0, 'R')
        else:
            self.set_text_color(220, 38, 38)
            self.cell(30, 6, f"{margin:.1f}%", 0, 0, 'R')

    def kpi_grid(self, data):
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        for row in data:
            self.cell(60, 8, row[0], 1)
            self.cell(40, 8, row[1], 1, 0, 'R')
            self.cell(40, 8, row[2], 1, 0, 'R')
            self.cell(40, 8, row[3], 1, 0, 'R')
            self.ln()

def create_pro_pdf(state, kpis):
    pdf = PDF_Pro()
    pdf.add_page()
    
    # 1. VISIÓN GENERAL
    pdf.chapter_title("1. RESUMEN DE COSTOS VS PRODUCCIÓN (SISTEMA)")
    pdf.ln(2)
    
    pdf.set_fill_color(226, 232, 240)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(60, 8, "Escala de Tiempo", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Ingreso Total", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Costo Total", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Utilidad", 1, 0, 'C', 1)
    pdf.ln()
    
    data_gen = [
        ["Por Hora Operativa", fmt_money(kpis['inc_sys_hr']), fmt_money(kpis['cost_sys_hr']), fmt_money(kpis['prof_sys_hr'])],
        ["Por Dia (Turno)", fmt_money(kpis['inc_sys_day']), fmt_money(kpis['cost_sys_day']), fmt_money(kpis['prof_sys_day'])],
        ["Proyeccion Mensual", fmt_money(kpis['inc_total']), fmt_money(kpis['cost_total']), fmt_money(kpis['prof_total'])]
    ]
    pdf.kpi_grid(data_gen)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, "Grafico de Rentabilidad Mensual:", 0, 1)
    pdf.draw_financial_bar("SISTEMA TOTAL", kpis['inc_total'], kpis['cost_total'], pdf.get_y())
    pdf.ln(20)
    
    # 2. DETALLE MAQUINAS
    pdf.chapter_title("2. DESGLOSE POR MAQUINA (Mensual)")
    pdf.ln(2)
    
    pdf.draw_financial_bar("HARVESTER", kpis['inc_h_mes'], kpis['cost_h_mes'], pdf.get_y())
    pdf.ln(15)
    pdf.draw_financial_bar("FORWARDER", kpis['inc_f_mes'], kpis['cost_f_mes'], pdf.get_y())
    pdf.ln(15)
    
    pdf.set_fill_color(226, 232, 240)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(60, 8, "Maquina", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Produccion (MR)", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Tarifa ($/MR)", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Utilidad Neta", 1, 0, 'C', 1)
    pdf.ln()
    
    pdf.kpi_grid([
        ["Harvester", f"{kpis['mr_h']:,.0f}", fmt_money(state['price_h']), fmt_money(kpis['prof_h_mes'])],
        ["Forwarder", f"{kpis['mr_f']:,.0f}", fmt_money(state['price_f']), fmt_money(kpis['prof_f_mes'])]
    ])
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

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
            "alloc_pct", "price_h", "price_f", "conv_factor", "target_company_margin",
            "prod_h_m3", "prod_f_m3"]
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame): state_to_save[k] = val.to_dict('records')
            else: state_to_save[k] = val
    with open(CONFIG_FILE, 'w') as f: json.dump(state_to_save, f, cls=NumpyEncoder)

# --- 4. INICIALIZACIÓN ---
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
init_key('target_company_margin', 30.0)
init_key('h_days', 28)
init_key('h_hours', 10.0)
init_key('f_days', 28)
init_key('f_hours', 10.0)
init_key('prod_h_m3', 25.0)
init_key('prod_f_m3', 28.0)

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

# --- 5. CÁLCULOS CENTRALES ---
h_dias = int(st.session_state['h_days'])
h_hours = float(st.session_state['h_hours'])
f_dias = int(st.session_state['f_days'])
f_hours = float(st.session_state['f_hours'])

# Costos
tot_h_dir, tot_f_dir, tot_ind = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    h_dias, h_hours, f_dias, f_hours, 
    st.session_state.get('uf_manual', 0), st.session_state['fuel_price']
)

cost_h_total_mes = tot_h_dir + (tot_ind * st.session_state['alloc_pct'])
cost_f_total_mes = tot_f_dir + (tot_ind * (1 - st.session_state['alloc_pct']))
cost_total_mes = cost_h_total_mes + cost_f_total_mes

# Producción
mr_h_hr = st.session_state['prod_h_m3'] / st.session_state['conv_factor']
mr_f_hr = st.session_state['prod_f_m3'] / st.session_state['conv_factor']

# Ingresos
inc_h_hr = mr_h_hr * st.session_state['price_h']
inc_h_day = inc_h_hr * h_hours
inc_h_mes = inc_h_day * h_dias
cost_h_hr = cost_h_total_mes / (h_dias * h_hours) if (h_dias*h_hours) > 0 else 0
prof_h_mes = inc_h_mes - cost_h_total_mes
margin_h = (prof_h_mes / inc_h_mes * 100) if inc_h_mes > 0 else 0

inc_f_hr = mr_f_hr * st.session_state['price_f']
inc_f_day = inc_f_hr * f_hours
inc_f_mes = inc_f_day * f_dias
cost_f_hr = cost_f_total_mes / (f_dias * f_hours) if (f_dias*f_hours) > 0 else 0
prof_f_mes = inc_f_mes - cost_f_total_mes
margin_f = (prof_f_mes / inc_f_mes * 100) if inc_f_mes > 0 else 0

inc_total = inc_h_mes + inc_f_mes
prof_total = prof_h_mes + prof_f_mes
margin_total = (prof_total / inc_total * 100) if inc_total > 0 else 0

# KPIs PDF
pdf_kpis = {
    'mr_h': mr_h_hr * h_hours * h_dias, 'mr_f': mr_f_hr * f_hours * f_dias,
    'inc_total': inc_total, 'cost_total': cost_total_mes, 'prof_total': prof_total,
    'margin_total': margin_total,
    'inc_h_mes': inc_h_mes, 'cost_h_mes': cost_h_total_mes, 'prof_h_mes': prof_h_mes,
    'inc_f_mes': inc_f_mes, 'cost_f_mes': cost_f_total_mes, 'prof_f_mes': prof_f_mes,
    'inc_sys_hr': inc_h_hr + inc_f_hr, 
    'cost_sys_hr': cost_h_hr + cost_f_hr,
    'prof_sys_hr': (inc_h_hr + inc_f_hr) - (cost_h_hr + cost_f_hr),
    'inc_sys_day': inc_h_day + inc_f_day,
    'cost_sys_day': (cost_h_total_mes/h_dias if h_dias>0 else 0) + (cost_f_total_mes/f_dias if f_dias>0 else 0),
    'prof_sys_day': (inc_h_day + inc_f_day) - ((cost_h_total_mes/h_dias if h_dias>0 else 0) + (cost_f_total_mes/f_dias if f_dias>0 else 0))
}

# --- 6. INTERFAZ ---
st.title("🌲 Sistema de Control de Gestión Forestal")

with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    try:
        pdf_bytes = create_pro_pdf(st.session_state, pdf_kpis)
        st.download_button("📄 Informe Ejecutivo (PDF)", data=pdf_bytes, file_name=f"Informe_Gestion_{datetime.now().strftime('%Y%m%d')}.pdf", mime='application/pdf', type="primary")
    except Exception as e: st.error(f"Error PDF: {e}")
    
    if st.button("♻️ Reset"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()
    
    with st.expander("🕒 Jornada Laboral", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state['h_days'] = c1.number_input("Días H", value=int(st.session_state['h_days']), on_change=save_config)
        st.session_state['h_hours'] = c2.number_input("Hrs H", value=float(st.session_state['h_hours']), on_change=save_config)
        st.session_state['f_days'] = st.number_input("Días F", value=int(st.session_state['f_days']), on_change=save_config)
        st.session_state['f_hours'] = st.number_input("Hrs F", value=float(st.session_state['f_hours']), on_change=save_config)
    
    with st.expander("💰 Tarifas"):
        st.session_state['price_h'] = st.number_input("Tarifa H", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Tarifa F", value=float(st.session_state['price_f']), on_change=save_config)
        st.session_state['fuel_price'] = st.number_input("Diesel", value=float(st.session_state['fuel_price']), on_change=save_config)
    
    with st.expander("📏 Conversión"):
        st.session_state['conv_factor'] = st.number_input("Factor", value=float(st.session_state['conv_factor']), on_change=save_config)
        st.session_state['alloc_pct'] = st.slider("% Ind. Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        st.session_state['target_company_margin'] = st.slider("Meta %", 0, 60, int(st.session_state['target_company_margin']), on_change=save_config)
        save_config()

tab_dash, tab_strat, tab_faena, tab_h, tab_f, tab_ind = st.tabs([
    "📊 Resultado Operacional", "🎯 Estrategia Precios", "🧮 Cierre de Faena", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos"
])

# --- TAB DASHBOARD ---
with tab_dash:
    st.subheader("1. Variables de Producción")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state['prod_h_m3'] = st.number_input("Prod. Harvester (m³/hr)", value=float(st.session_state['prod_h_m3']), step=0.5, on_change=save_config)
        st.info(f"H: {mr_h_hr:,.1f} MR/hr")
    with c2:
        st.session_state['prod_f_m3'] = st.number_input("Prod. Forwarder (m³/hr)", value=float(st.session_state['prod_f_m3']), step=0.5, on_change=save_config)
        st.info(f"F: {mr_f_hr:,.1f} MR/hr")
    with c3:
        target_pct = st.session_state['target_company_margin']
        st.metric("Meta de Margen", f"{target_pct}%")

    st.divider()

    st.subheader("2. Análisis Visual")
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        fig_money = go.Figure()
        cats = ['Harvester', 'Forwarder', 'Total']
        fig_money.add_trace(go.Bar(name='Venta', x=cats, y=[inc_h_mes, inc_f_mes, inc_total], marker_color='#3b82f6', texttemplate='$%{y:,.0f}'))
        fig_money.add_trace(go.Bar(name='Costo', x=cats, y=[cost_h_total_mes, cost_f_total_mes, cost_total_mes], marker_color='#ef4444', texttemplate='$%{y:,.0f}'))
        fig_money.add_trace(go.Bar(name='Utilidad', x=cats, y=[prof_h_mes, prof_f_mes, prof_total], marker_color='#22c55e', texttemplate='$%{y:,.0f}'))
        fig_money.update_layout(barmode='group', title="Resultado $ (Mensual)", height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_money, use_container_width=True)
        
    with c_g2:
        fig_pct = go.Figure()
        cols = ['#22c55e' if m >= target_pct else '#ef4444' for m in [margin_h, margin_f, margin_total]]
        fig_pct.add_trace(go.Bar(x=cats, y=[margin_h, margin_f, margin_total], marker_color=cols, text=[f"{m:.1f}%" for m in [margin_h, margin_f, margin_total]], textposition='auto'))
        fig_pct.add_shape(type="line", x0=-0.5, x1=2.5, y0=target_pct, y1=target_pct, line=dict(color="black", width=2, dash="dash"))
        fig_pct.update_layout(title=f"Margen % (Meta: {target_pct}%)", height=350)
        st.plotly_chart(fig_pct, use_container_width=True)

    st.subheader("3. Detalle por Máquina (Hora / Día / Mes)")
    
    def render_pro_card(title, df, margin, target, color_border):
        badge = "badge-ok" if margin >= target else "badge-bad"
        badge_text = "CUMPLE META" if margin >= target else "BAJO META"
        
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

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        df_h = pd.DataFrame([
            {"Periodo": "Hora", "Generado": fmt_money(inc_h_hr), "Costo": fmt_money(cost_h_hr), "Ganancia": fmt_money(inc_h_hr-cost_h_hr)},
            {"Periodo": "Día", "Generado": fmt_money(inc_h_day), "Costo": fmt_money(cost_h_total_mes/h_dias if h_dias>0 else 0), "Ganancia": fmt_money(inc_h_day-(cost_h_total_mes/h_dias if h_dias>0 else 0))},
            {"Periodo": "Mes", "Generado": fmt_money(inc_h_mes), "Costo": fmt_money(cost_h_total_mes), "Ganancia": fmt_money(prof_h_mes)},
        ])
        st.markdown(render_pro_card("🚜 HARVESTER", df_h, margin_h, target_pct, "#eab308"), unsafe_allow_html=True)
        
    with c_t2:
        df_f = pd.DataFrame([
            {"Periodo": "Hora", "Generado": fmt_money(inc_f_hr), "Costo": fmt_money(cost_f_hr), "Ganancia": fmt_money(inc_f_hr-cost_f_hr)},
            {"Periodo": "Día", "Generado": fmt_money(inc_f_day), "Costo": fmt_money(cost_f_total_mes/f_dias if f_dias>0 else 0), "Ganancia": fmt_money(inc_f_day-(cost_f_total_mes/f_dias if f_dias>0 else 0))},
            {"Periodo": "Mes", "Generado": fmt_money(inc_f_mes), "Costo": fmt_money(cost_f_total_mes), "Ganancia": fmt_money(prof_f_mes)},
        ])
        st.markdown(render_pro_card("🚜 FORWARDER", df_f, margin_f, target_pct, "#22c55e"), unsafe_allow_html=True)

# --- TAB FAENA ---
with tab_faena:
    st.header("🧮 Cierre de Faena")
    st.markdown("Ingresa el **Total de Metros Ruma (MR)** de una faena para ver su resultado específico.")
    mr_lote = st.number_input("Total MR Faena", value=1000.0, step=100.0)
    
    if mr_lote > 0:
        st.divider()
        hrs_req_h = mr_lote / mr_h_hr if mr_h_hr > 0 else 0
        hrs_req_f = mr_lote / mr_f_hr if mr_f_hr > 0 else 0
        
        inc_lote_h = mr_lote * st.session_state['price_h']
        inc_lote_f = mr_lote * st.session_state['price_f']
        
        cost_lote_h = hrs_req_h * cost_h_hr
        cost_lote_f = hrs_req_f * cost_f_hr
        
        prof_lote_h = inc_lote_h - cost_lote_h
        prof_lote_f = inc_lote_f - cost_lote_f
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown(f"""
            <div class="faena-card">
                <div class="card-header">🚜 Harvester</div>
                <div>Horas Requeridas: <b>{hrs_req_h:,.1f} hrs</b></div>
                <div style="margin-top:10px; font-size:1.1em">Utilidad: <span style="color:#16a34a; font-weight:bold">{fmt_money(prof_lote_h)}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c_res2:
            st.markdown(f"""
            <div class="faena-card">
                <div class="card-header">🚜 Forwarder</div>
                <div>Horas Requeridas: <b>{hrs_req_f:,.1f} hrs</b></div>
                <div style="margin-top:10px; font-size:1.1em">Utilidad: <span style="color:#16a34a; font-weight:bold">{fmt_money(prof_lote_f)}</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.success(f"💰 **UTILIDAD TOTAL DEL LOTE: {fmt_money(prof_lote_h + prof_lote_f)}**")

# --- TAB STRAT ---
with tab_strat:
    st.subheader("Simulador de Tarifas")
    prod_cotiza = st.number_input("Prod. Estimada (m³/hr)", value=25.0)
    mr_cotiza_sim = prod_cotiza / st.session_state['conv_factor']
    
    cost_unit_h_sim = (cost_h_total_mes / (h_dias*h_hours)) / mr_cotiza_sim if mr_cotiza_sim > 0 else 0
    cost_unit_f_sim = (cost_f_total_mes / (f_dias*f_hours)) / mr_cotiza_sim if mr_cotiza_sim > 0 else 0
    
    p30 = calc_price(cost_unit_h_sim+cost_unit_f_sim, 30)
    p35 = calc_price(cost_unit_h_sim+cost_unit_f_sim, 35)
    
    c1, c2 = st.columns(2)
    c1.metric("Tarifa Sugerida (30%)", fmt_money(p30))
    c2.metric("Tarifa Sugerida (35%)", fmt_money(p35))

# --- TABS EDICIÓN ---
with tab_h:
    st.subheader("Costos Harvester")
    # Mostrar Total GRANDE
    total_h_display = calculate_single_machine_monthly_cost(st.session_state['df_harvester'], h_dias, h_hours, st.session_state.get('uf_manual',0), st.session_state['fuel_price'], 'H')
    st.metric("💰 COSTO DIRECTO MENSUAL", fmt_money(total_h_display))
    
    edited_h = st.data_editor(st.session_state['df_harvester'], num_rows="dynamic", use_container_width=True, column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_harvester'] = edited_h
    save_config()

with tab_f:
    st.subheader("Costos Forwarder")
    total_f_display = calculate_single_machine_monthly_cost(st.session_state['df_forwarder'], f_dias, f_hours, st.session_state.get('uf_manual',0), st.session_state['fuel_price'], 'F')
    st.metric("💰 COSTO DIRECTO MENSUAL", fmt_money(total_f_display))
    
    edited_f = st.data_editor(st.session_state['df_forwarder'], num_rows="dynamic", use_container_width=True, column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_forwarder'] = edited_f
    save_config()

with tab_ind:
    c1, c2 = st.columns(2)
    with c1: st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c2: st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()
    
    total_ind_display = st.session_state['df_rrhh']['Costo Empresa'].sum() + st.session_state['df_flota']['Monto'].sum()
    st.metric("💰 TOTAL GASTOS INDIRECTOS", fmt_money(total_ind_display))
