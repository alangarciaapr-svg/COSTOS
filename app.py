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
    page_title="COSTOS / PRODUCCION SOCIEDAD MADERERA GALVEZ Y DI GENOVA LTDA.", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    h1 {color: #1e3a8a; font-family: 'Arial Black', sans-serif; font-size: 2rem;}
    h2, h3 {color: #334155; font-family: 'Segoe UI', sans-serif;}
    
    /* Input Gigante para Costos Totales */
    .big-input input {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #1e40af !important;
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        padding: 15px !important;
        border-radius: 8px !important;
    }
    .big-input label {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
    }

    /* Métricas */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    
    /* Tarjetas de Resultados */
    .machine-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 5px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 15px; font-weight: 800; font-size: 1.1em; color: #334155; text-transform: uppercase;
    }
    
    /* --- SIDEBAR BLANCO PROFESIONAL --- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    /* Textos oscuros para alto contraste */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p {
        color: #1e293b !important;
    }
    /* Inputs Sidebar */
    section[data-testid="stSidebar"] input {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Ocultar elementos nativos innecesarios */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v34_final_clean.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calc_price(cost, margin_pct):
    if margin_pct >= 100: return 0 
    factor = 1 - (margin_pct / 100.0)
    return cost / factor if factor > 0 else 0

# --- 3. MOTOR PDF GRÁFICO ---
class PDF_Pro(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 45, 'F')
        self.set_font('Arial', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 12)
        self.cell(0, 10, 'SOCIEDAD MADERERA GALVEZ Y DI GENOVA LTDA.', 0, 1, 'L')
        self.set_font('Arial', '', 10)
        self.set_text_color(203, 213, 225)
        self.cell(0, 5, 'INFORME DE GESTION: COSTOS V/S PRODUCCION', 0, 1, 'L')
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
    
    pdf.chapter_title("1. RESUMEN EJECUTIVO (SISTEMA)")
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
    
    pdf.chapter_title("2. DESGLOSE POR MAQUINA")
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
    keys = ["h_days", "h_hours", "f_days", "f_hours", 
            "cost_total_h", "cost_total_f", "cost_total_ind",
            "alloc_pct", "price_h", "price_f", "conv_factor", "target_company_margin",
            "prod_h_m3", "prod_f_m3"]
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            state_to_save[k] = st.session_state[k]
    with open(CONFIG_FILE, 'w') as f: json.dump(state_to_save, f, cls=NumpyEncoder)

# --- 4. INICIALIZACIÓN ---
saved = load_config()

def init_key(key, default_value):
    if key not in st.session_state:
        loaded_val = saved.get(key)
        if loaded_val is not None:
            st.session_state[key] = loaded_val
        else:
            st.session_state[key] = default_value

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

# INPUTS TOTALES (Valores por defecto)
init_key('cost_total_h', 15000000.0)
init_key('cost_total_f', 12000000.0)
init_key('cost_total_ind', 5000000.0)

# --- 5. CÁLCULO CENTRAL ---
h_dias = int(st.session_state['h_days'])
h_hours = float(st.session_state['h_hours'])
f_dias = int(st.session_state['f_days'])
f_hours = float(st.session_state['f_hours'])

# 1. Costos (Directo del input manual)
tot_h_dir = float(st.session_state['cost_total_h'])
tot_f_dir = float(st.session_state['cost_total_f'])
tot_ind = float(st.session_state['cost_total_ind'])

# 2. Distribución Indirectos
cost_h_total_mes = tot_h_dir + (tot_ind * st.session_state['alloc_pct'])
cost_f_total_mes = tot_f_dir + (tot_ind * (1 - st.session_state['alloc_pct']))
cost_total_mes = cost_h_total_mes + cost_f_total_mes

# 3. Producción (Real del Dashboard)
mr_h_hr = st.session_state['prod_h_m3'] / st.session_state['conv_factor']
mr_f_hr = st.session_state['prod_f_m3'] / st.session_state['conv_factor']

# 4. Ingresos y Utilidad (Harvester)
inc_h_hr = mr_h_hr * st.session_state['price_h']
inc_h_day = inc_h_hr * h_hours
inc_h_mes = inc_h_day * h_dias
cost_h_hr = cost_h_total_mes / (h_dias * h_hours) if (h_dias*h_hours) > 0 else 0
prof_h_mes = inc_h_mes - cost_h_total_mes
margin_h = (prof_h_mes / inc_h_mes * 100) if inc_h_mes > 0 else 0

# 5. Ingresos y Utilidad (Forwarder)
inc_f_hr = mr_f_hr * st.session_state['price_f']
inc_f_day = inc_f_hr * f_hours
inc_f_mes = inc_f_day * f_dias
cost_f_hr = cost_f_total_mes / (f_dias * f_hours) if (f_dias*f_hours) > 0 else 0
prof_f_mes = inc_f_mes - cost_f_total_mes
margin_f = (prof_f_mes / inc_f_mes * 100) if inc_f_mes > 0 else 0

# 6. Consolidado
inc_total = inc_h_mes + inc_f_mes
prof_total = prof_h_mes + prof_f_mes
margin_total = (prof_total / inc_total * 100) if inc_total > 0 else 0

# 7. Pack Datos PDF
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
st.title("COSTOS / PRODUCCION SOCIEDAD MADERERA GALVEZ Y DI GENOVA LTDA.")

with st.sidebar:
    st.markdown("### ⚙️ PANEL DE GESTIÓN")
    
    with st.expander("💵 Tarifas Venta ($/MR)", expanded=True):
        st.session_state['price_h'] = st.number_input("Harvester", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Forwarder", value=float(st.session_state['price_f']), on_change=save_config)

    with st.expander("🕒 Jornada Operativa", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state['h_days'] = c1.number_input("Días H", value=int(st.session_state['h_days']), on_change=save_config)
        st.session_state['h_hours'] = c2.number_input("Hrs H", value=float(st.session_state['h_hours']), on_change=save_config)
        
        c3, c4 = st.columns(2)
        st.session_state['f_days'] = c3.number_input("Días F", value=int(st.session_state['f_days']), on_change=save_config)
        st.session_state['f_hours'] = c4.number_input("Hrs F", value=float(st.session_state['f_hours']), on_change=save_config)

    with st.expander("📏 Técnica y Objetivos", expanded=False):
        st.session_state['conv_factor'] = st.number_input("Factor Conv. (m³/MR)", value=float(st.session_state['conv_factor']), on_change=save_config)
        st.session_state['alloc_pct'] = st.slider("% Indirectos a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        st.session_state['target_company_margin'] = st.slider("Meta Margen (%)", 0, 60, int(st.session_state['target_company_margin']), on_change=save_config)

    st.markdown("---")
    try:
        pdf_bytes = create_pro_pdf(st.session_state, pdf_kpis)
        st.download_button("📄 DESCARGAR INFORME PDF", data=pdf_bytes, file_name=f"Informe_Galvez_Genova_{datetime.now().strftime('%Y%m%d')}.pdf", mime='application/pdf', type="primary")
    except Exception as e: st.error(f"Error PDF: {e}")
    
    if st.button("♻️ Resetear Todo", type="secondary"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

tab_dash, tab_strat, tab_faena, tab_h, tab_f, tab_ind = st.tabs([
    "📊 Dashboard Gerencial", "🎯 Simulador Precios", "🧮 Cierre de Faena", "🚜 COSTOS HARVESTER", "🚜 COSTOS FORWARDER", "👷 COSTOS INDIRECTOS"
])

# --- DASHBOARD ---
with tab_dash:
    st.subheader("1. Variables de Producción (En Terreno)")
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

    st.subheader("2. Estado de Resultados Visual")
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

# --- TAB SIMULADOR CORREGIDO ---
with tab_strat:
    st.subheader("Simulador de Tarifas")
    st.markdown("Defina el rendimiento esperado para cotizar un nuevo trabajo.")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_mr = st.number_input("Productividad Objetivo (MR/Hr)", value=20.0, step=0.5)
    with col_sim2:
        sim_factor = st.number_input("Factor Faena (m³/MR)", value=2.44, step=0.01)
        
    equiv_m3 = sim_mr * sim_factor
    st.info(f"Equivale a una producción de: **{equiv_m3:,.1f} m³/Hr**")
    
    # Cálculo Seguro: Costo Hora Sistema / MR Objetivo
    # Evitamos errores usando variables ya calculadas globalmente
    hourly_cost_h_sys = cost_h_total_mes / (h_dias * h_hours) if (h_dias * h_hours) > 0 else 0
    hourly_cost_f_sys = cost_f_total_mes / (f_dias * f_hours) if (f_dias * f_hours) > 0 else 0
    
    # Costo por MR (Unitario)
    unit_cost_h = hourly_cost_h_sys / sim_mr if sim_mr > 0 else 0
    unit_cost_f = hourly_cost_f_sys / sim_mr if sim_mr > 0 else 0
    
    st.divider()
    st.markdown("#### Tarifas Sugeridas ($/MR)")
    
    c30, c35 = st.columns(2)
    
    with c30:
        p30_h = calc_price(unit_cost_h, 30)
        p30_f = calc_price(unit_cost_f, 30)
        st.markdown(f"""
        <div class="machine-card" style="border-top-color: #fcd34d">
            <div style="font-weight:bold; color:#b45309; margin-bottom:10px">META 30% MARGEN</div>
            <div style="display:flex; justify-content:space-between"><span>Harvester:</span><b>{fmt_money(p30_h)}</b></div>
            <div style="display:flex; justify-content:space-between"><span>Forwarder:</span><b>{fmt_money(p30_f)}</b></div>
            <hr>
            <div style="font-size:1.2em; font-weight:800; text-align:center; color:#b45309">{fmt_money(p30_h+p30_f)} / MR</div>
        </div>
        """, unsafe_allow_html=True)

    with c35:
        p35_h = calc_price(unit_cost_h, 35)
        p35_f = calc_price(unit_cost_f, 35)
        st.markdown(f"""
        <div class="machine-card" style="border-top-color: #22c55e">
            <div style="font-weight:bold; color:#15803d; margin-bottom:10px">META 35% MARGEN</div>
            <div style="display:flex; justify-content:space-between"><span>Harvester:</span><b>{fmt_money(p35_h)}</b></div>
            <div style="display:flex; justify-content:space-between"><span>Forwarder:</span><b>{fmt_money(p35_f)}</b></div>
            <hr>
            <div style="font-size:1.2em; font-weight:800; text-align:center; color:#15803d">{fmt_money(p35_h+p35_f)} / MR</div>
        </div>
        """, unsafe_allow_html=True)

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

# --- TABS COSTOS (INPUTS GIGANTES) ---
with tab_h:
    st.markdown("### Ingrese el Costo Total Mensual")
    st.markdown('<div class="big-input">', unsafe_allow_html=True)
    st.session_state['cost_total_h'] = st.number_input("Costo Harvester ($)", value=float(st.session_state['cost_total_h']), step=100000.0, format="%f", on_change=save_config, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"Costo por Hora Calculado: {fmt_money(tot_h_dir / (h_dias*h_hours) if (h_dias*h_hours)>0 else 0)}")

with tab_f:
    st.markdown("### Ingrese el Costo Total Mensual")
    st.markdown('<div class="big-input">', unsafe_allow_html=True)
    st.session_state['cost_total_f'] = st.number_input("Costo Forwarder ($)", value=float(st.session_state['cost_total_f']), step=100000.0, format="%f", on_change=save_config, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"Costo por Hora Calculado: {fmt_money(tot_f_dir / (f_dias*f_hours) if (f_dias*f_hours)>0 else 0)}")

with tab_ind:
    st.markdown("### Ingrese Total Gastos Indirectos")
    st.markdown('<div class="big-input">', unsafe_allow_html=True)
    st.session_state['cost_total_ind'] = st.number_input("Indirectos ($)", value=float(st.session_state['cost_total_ind']), step=100000.0, format="%f", on_change=save_config, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    st.info("Este monto se reparte entre H y F según el % definido en la barra lateral.")
