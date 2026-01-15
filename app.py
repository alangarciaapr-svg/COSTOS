import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

# Intentamos importar FPDF
try:
    from fpdf import FPDF
except ImportError:
    st.error("⚠️ Librería fpdf no instalada. Agrega 'fpdf' a requirements.txt")

# Nombre del archivo de logo esperado (debe estar en la misma carpeta)
LOGO_FILE = "logo.png" 

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
        padding: 15px;
        border-radius: 12px;
        border-top: 5px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 10px; font-weight: 800; font-size: 1.1em; color: #334155; text-transform: uppercase;
    }
    
    /* Estilos Faena */
    .faena-metric {
        font-size: 1.1em; color: #475569; margin-bottom: 5px;
        display: flex; justify-content: space-between; border-bottom: 1px dashed #e2e8f0; padding-bottom: 4px;
    }
    .faena-val { font-weight: 700; color: #0f172a; }
    .faena-result-box {
        background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
        padding: 15px; text-align: center; margin-top: 20px;
    }
    
    /* --- SIDEBAR BLANCO PROFESIONAL --- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] h1, h2, h3, label, p {
        color: #1e293b !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Ocultar elementos nativos */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v43_sim_final.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

# --- 3. MOTOR PDF GRÁFICO (CON LOGO Y DATOS REALES) ---
class PDF_Pro(FPDF):
    def header(self):
        # Fondo Azul Corporativo
        self.set_fill_color(30, 58, 138) # #1e3a8a
        self.rect(0, 0, 210, 40, 'F') # Aumentado un poco la altura para el logo
        
        # LOGO EMPRESA (Si existe)
        if os.path.exists(LOGO_FILE):
            # Insertar imagen (x, y, w). Ajusta 'w' según el tamaño de tu logo.
            try:
                self.image(LOGO_FILE, x=10, y=5, w=30)
                title_x_offset = 45 # Desplazar título si hay logo
            except:
                title_x_offset = 10
        else:
            title_x_offset = 10

        # Título
        self.set_font('Arial', 'B', 14) 
        self.set_text_color(255, 255, 255)
        self.set_xy(title_x_offset, 12)
        self.cell(0, 10, 'SOCIEDAD MADERERA GALVEZ Y DI GENOVA LTDA.', 0, 1, 'L')
        
        # Subtítulo
        self.set_font('Arial', '', 10)
        self.set_text_color(203, 213, 225)
        self.set_xy(title_x_offset, 19)
        self.cell(0, 5, 'REPORTE INTEGRAL: COSTOS, TARIFAS Y RESULTADOS', 0, 1, 'L')
        
        # Fecha
        self.set_xy(160, 15)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.cell(40, 10, datetime.now().strftime('%d/%m/%Y'), 0, 1, 'R')
        self.ln(25) # Espacio seguro después del header

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(30, 58, 138)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_draw_color(30, 58, 138)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def kp_card(self, label, value, sublabel, x, y, w=45, h=25, is_money=True):
        self.set_xy(x, y)
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(203, 213, 225)
        self.rect(x, y, w, h, 'DF')
        
        self.set_xy(x, y + 3)
        self.set_font('Arial', 'B', 8)
        self.set_text_color(100, 116, 139)
        self.cell(w, 5, label, 0, 0, 'C')
        
        self.set_xy(x, y + 9)
        self.set_font('Arial', 'B', 11) 
        self.set_text_color(15, 23, 42)
        val_str = fmt_money(value) if is_money else str(value)
        self.cell(w, 8, val_str, 0, 0, 'C')
        
        self.set_xy(x, y + 17)
        self.set_font('Arial', '', 7)
        self.set_text_color(22, 163, 74) # Green
        self.cell(w, 5, sublabel, 0, 0, 'C')

    def nice_table(self, header, data, col_widths):
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(226, 232, 240)
        self.set_text_color(30, 41, 59)
        self.set_draw_color(203, 213, 225)
        
        for i, h in enumerate(header):
            self.cell(col_widths[i], 8, h, 1, 0, 'C', 1)
        self.ln()
        
        self.set_font('Arial', '', 9)
        self.set_text_color(51, 65, 85)
        
        for row in data:
            for i, d in enumerate(row):
                align = 'L' if i == 0 else 'R'
                txt = str(d)
                self.cell(col_widths[i], 8, txt, 1, 0, align, 0)
            self.ln() 

def create_pro_pdf(state, kpis):
    pdf = PDF_Pro()
    pdf.add_page()
    
    # --- RECUPERAR DATOS EXACTOS DE LA APP ---
    mr_h_hr_real = kpis['mr_h_hr'] 
    mr_f_hr_real = kpis['mr_f_hr']
    # Recuperamos el valor del input del usuario
    lote_usuario = kpis.get('mr_lote_input', 1000.0) 
    
    # --- SECCIÓN 1: PARÁMETROS CONFIGURADOS ---
    pdf.section_title("1. PARAMETROS DE OPERACION")
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 5, "Resumen de las variables utilizadas para el calculo.", 0, 1)
    pdf.ln(4)

    params_header = ["Variable", "Harvester", "Forwarder", "Sistema Total"]
    params_data = [
        ["Dias Operativos", f"{state['h_days']}", f"{state['f_days']}", "-"],
        ["Horas por Turno", f"{state['h_hours']}", f"{state['f_hours']}", "-"],
        ["Productividad (MR/hr)", f"{mr_h_hr_real:.1f}", f"{mr_f_hr_real:.1f}", "-"],
        ["Costo Operativo Mensual", fmt_money(kpis['cost_h_mes']), fmt_money(kpis['cost_f_mes']), fmt_money(kpis['cost_total'])],
        ["Tarifa Venta ($/MR)", fmt_money(state['price_h']), fmt_money(state['price_f']), fmt_money(state['price_h']+state['price_f'])]
    ]
    pdf.nice_table(params_header, params_data, [50, 40, 40, 40])
    pdf.ln(12)

    # --- SECCIÓN 2: RESULTADOS FINANCIEROS (MENSUAL) ---
    pdf.section_title("2. PROYECCION MENSUAL")
    
    y_start = pdf.get_y()
    pdf.kp_card("INGRESO TOTAL", kpis['inc_total'], "Mensual Estimado", 10, y_start)
    pdf.kp_card("COSTO TOTAL", kpis['cost_total'], "Directo + Indirecto", 60, y_start)
    pdf.kp_card("UTILIDAD", kpis['prof_total'], f"Margen: {kpis['margin_total']:.1f}%", 110, y_start)
    
    pdf.set_y(y_start + 30) 
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, "Detalle del Estado de Resultados:", 0, 1)
    
    fin_header = ["Concepto", "Ingreso ($)", "Costo Total ($)", "Utilidad ($)", "Margen %"]
    fin_data = [
        ["HARVESTER", fmt_money(kpis['inc_h_mes']), fmt_money(kpis['cost_h_mes']), fmt_money(kpis['prof_h_mes']), f"{kpis['margin_h']:.1f}%"],
        ["FORWARDER", fmt_money(kpis['inc_f_mes']), fmt_money(kpis['cost_f_mes']), fmt_money(kpis['prof_f_mes']), f"{kpis['margin_f']:.1f}%"],
        ["TOTAL SISTEMA", fmt_money(kpis['inc_total']), fmt_money(kpis['cost_total']), fmt_money(kpis['prof_total']), f"{kpis['margin_total']:.1f}%"]
    ]
    pdf.nice_table(fin_header, fin_data, [50, 35, 35, 35, 25])
    pdf.ln(10)

    # --- SECCIÓN 3: CIERRE DE FAENA (Sincronizado con App) ---
    if pdf.get_y() > 180:
        pdf.add_page()
    else:
        pdf.ln(5)

    # Usamos el valor real del input
    pdf.section_title(f"3. ANALISIS DE CIERRE DE FAENA (Lote: {lote_usuario:,.0f} MR)")
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, f"Calculo especifico para un volumen de {lote_usuario:,.0f} MR, basado en los costos y tarifas actuales.")
    pdf.ln(4)
    
    hrs_h = lote_usuario / mr_h_hr_real if mr_h_hr_real > 0 else 0
    hrs_f = lote_usuario / mr_f_hr_real if mr_f_hr_real > 0 else 0
    
    cost_lote = (hrs_h * kpis['cost_sys_hr_h']) + (hrs_f * kpis['cost_sys_hr_f'])
    
    inc_lote = lote_usuario * (state['price_h'] + state['price_f'])
    prof_lote = inc_lote - cost_lote
    marg_lote = (prof_lote / inc_lote * 100) if inc_lote > 0 else 0

    faena_header = ["Metrica", "Valor Calculado"]
    faena_data = [
        ["Volumen Evaluado", f"{lote_usuario:,.0f} MR"],
        ["Horas Maquina Requeridas", f"H: {hrs_h:.1f} hrs | F: {hrs_f:.1f} hrs"],
        ["Facturacion Estimada", fmt_money(inc_lote)],
        ["Costo Operativo Real", fmt_money(cost_lote)],
        ["UTILIDAD FAENA", fmt_money(prof_lote)],
        ["MARGEN OPERACIONAL", f"{marg_lote:.1f}%"]
    ]
    pdf.nice_table(faena_header, faena_data, [80, 80])

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
            "pct_ind_h", "pct_ind_f", "price_h", "price_f", "conv_factor", "target_company_margin",
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
init_key('pct_ind_h', 50.0) 
init_key('pct_ind_f', 50.0) 
init_key('conv_factor', 2.44)
init_key('target_company_margin', 30.0)
init_key('h_days', 28)
init_key('h_hours', 10.0)
init_key('f_days', 28)
init_key('f_hours', 10.0)
init_key('prod_h_m3', 25.0)
init_key('prod_f_m3', 28.0)
init_key('cost_total_h', 15000000.0)
init_key('cost_total_f', 12000000.0)
init_key('cost_total_ind', 5000000.0)

# --- 5. RENDERIZADO DE INPUTS ---
with st.sidebar:
    # --- LOGO APP ---
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, use_column_width=True)
    
    st.markdown("### ⚙️ PANEL DE GESTIÓN")
    
    with st.expander("💵 Tarifas Venta (Sistema)", expanded=True):
        current_total_price = st.session_state['price_h'] + st.session_state['price_f']
        if current_total_price == 0: current_total_price = 11500.0
        
        total_price_val = st.number_input("Tarifa Total Sistema ($/MR)", value=float(current_total_price), step=100.0)
        
        current_h_pct_price = (st.session_state['price_h'] / total_price_val * 100) if total_price_val > 0 else 60.0
        h_split_price_pct = st.slider("Distribución Venta: % a Harvester", 0, 100, int(current_h_pct_price))
        
        st.session_state['price_h'] = total_price_val * (h_split_price_pct / 100)
        st.session_state['price_f'] = total_price_val * ((100 - h_split_price_pct) / 100)
        save_config()
        
        c_p1, c_p2 = st.columns(2)
        c_p1.metric("H", fmt_money(st.session_state['price_h']))
        c_p2.metric("F", fmt_money(st.session_state['price_f']))

    with st.expander("🕒 Jornada Operativa", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state['h_days'] = c1.number_input("Días H", value=int(st.session_state['h_days']))
        st.session_state['h_hours'] = c2.number_input("Hrs H", value=float(st.session_state['h_hours']))
        c3, c4 = st.columns(2)
        st.session_state['f_days'] = c3.number_input("Días F", value=int(st.session_state['f_days']))
        st.session_state['f_hours'] = c4.number_input("Hrs F", value=float(st.session_state['f_hours']))

    with st.expander("📏 Técnica y Objetivos", expanded=False):
        st.session_state['conv_factor'] = st.number_input("Factor Conv. (m³/MR)", value=float(st.session_state['conv_factor']))
        
        st.markdown("---")
        st.markdown("**Distribución Indirectos (%)**")
        h_pct_val = st.slider("Asignación Gasto a Harvester", 0, 100, int(st.session_state['pct_ind_h']))
        st.session_state['pct_ind_h'] = h_pct_val
        st.session_state['pct_ind_f'] = 100 - h_pct_val
        
        c_dist1, c_dist2 = st.columns(2)
        c_dist1.info(f"🚜 H: {st.session_state['pct_ind_h']}%")
        c_dist2.info(f"🚜 F: {st.session_state['pct_ind_f']}%")
        
        st.session_state['target_company_margin'] = st.slider("Meta Margen (%)", 0, 60, int(st.session_state['target_company_margin']))

    st.markdown("---")
    if st.button("♻️ Resetear Todo", type="secondary"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

# --- DEFINICIÓN DE PESTAÑAS (SIN SIMULADOR) ---
tab_dash, tab_faena, tab_h, tab_f, tab_ind = st.tabs([
    "📊 Dashboard Gerencial", "🧮 Cierre de Faena", "🚜 COSTOS HARVESTER", "🚜 COSTOS FORWARDER", "👷 COSTOS INDIRECTOS"
])

# --- INPUTS COSTOS ---
with tab_h:
    st.markdown("### Ingrese el Costo Total Mensual")
    st.markdown('<div class="big-input">', unsafe_allow_html=True)
    st.session_state['cost_total_h'] = st.number_input("Costo Harvester ($)", value=float(st.session_state['cost_total_h']), step=100000.0, format="%f", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_f:
    st.markdown("### Ingrese el Costo Total Mensual")
    st.markdown('<div class="big-input">', unsafe_allow_html=True)
    st.session_state['cost_total_f'] = st.number_input("Costo Forwarder ($)", value=float(st.session_state['cost_total_f']), step=100000.0, format="%f", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_ind:
    st.markdown("### Ingrese Total Gastos Indirectos (Mensual)")
    st.markdown('<div class="big-input">', unsafe_allow_html=True)
    st.session_state['cost_total_ind'] = st.number_input("Indirectos ($)", value=float(st.session_state['cost_total_ind']), step=100000.0, format="%f", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. MOTOR DE CÁLCULO ---
h_dias = int(st.session_state['h_days'])
h_hours = float(st.session_state['h_hours'])
f_dias = int(st.session_state['f_days'])
f_hours = float(st.session_state['f_hours'])

tot_h_dir = float(st.session_state['cost_total_h'])
tot_f_dir = float(st.session_state['cost_total_f'])
tot_ind = float(st.session_state['cost_total_ind'])

# --- LOGICA COSTOS HORA CORRECTA ---
hourly_ind_base_24_7 = tot_ind / 720 # Base cronologica

burden_h_ind = hourly_ind_base_24_7 * (st.session_state['pct_ind_h'] / 100)
burden_f_ind = hourly_ind_base_24_7 * (st.session_state['pct_ind_f'] / 100)

burden_h_dir = tot_h_dir / (h_dias * h_hours) if (h_dias * h_hours) > 0 else 0
burden_f_dir = tot_f_dir / (f_dias * f_hours) if (f_dias * f_hours) > 0 else 0

cost_h_hr = burden_h_dir + burden_h_ind
cost_f_hr = burden_f_dir + burden_f_ind

# Totales P&L ajustados
cost_h_total_mes_real = cost_h_hr * (h_dias * h_hours)
cost_f_total_mes_real = cost_f_hr * (f_dias * f_hours)
cost_total_mes_real = cost_h_total_mes_real + cost_f_total_mes_real

# Display en Pestañas
with tab_h:
    st.success(f"💰 **COSTO TOTAL HORA:** {fmt_money(cost_h_hr)}")
    st.caption(f"Desglose: {fmt_money(burden_h_dir)} (Directo) + {fmt_money(burden_h_ind)} (Indirecto)")

with tab_f:
    st.success(f"💰 **COSTO TOTAL HORA:** {fmt_money(cost_f_hr)}")
    st.caption(f"Desglose: {fmt_money(burden_f_dir)} (Directo) + {fmt_money(burden_f_ind)} (Indirecto)")

with tab_ind:
    st.info(f"ℹ️ **Valor Hora Cronológica (Base 720 hrs):** {fmt_money(hourly_ind_base_24_7)}")
    st.caption(f"Asignación: {fmt_money(burden_h_ind)} a H | {fmt_money(burden_f_ind)} a F (por hora)")

# Producción
with tab_dash: 
    st.subheader("1. Variables de Producción (En Terreno)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state['prod_h_m3'] = st.number_input("Prod. Harvester (m³/hr)", value=float(st.session_state['prod_h_m3']), step=0.5)
        mr_h_hr = st.session_state['prod_h_m3'] / st.session_state['conv_factor']
        st.info(f"H: {mr_h_hr:,.1f} MR/hr")
    with c2:
        st.session_state['prod_f_m3'] = st.number_input("Prod. Forwarder (m³/hr)", value=float(st.session_state['prod_f_m3']), step=0.5)
        mr_f_hr = st.session_state['prod_f_m3'] / st.session_state['conv_factor']
        st.info(f"F: {mr_f_hr:,.1f} MR/hr")
    with c3:
        target_pct = st.session_state['target_company_margin']
        st.metric("Meta de Margen", f"{target_pct}%")

# Ingresos y Utilidad
inc_h_hr = mr_h_hr * st.session_state['price_h']
inc_h_day = inc_h_hr * h_hours
inc_h_mes = inc_h_day * h_dias
prof_h_mes = inc_h_mes - cost_h_total_mes_real
margin_h = (prof_h_mes / inc_h_mes * 100) if inc_h_mes > 0 else 0

inc_f_hr = mr_f_hr * st.session_state['price_f']
inc_f_day = inc_f_hr * f_hours
inc_f_mes = inc_f_day * f_dias
prof_f_mes = inc_f_mes - cost_f_total_mes_real
margin_f = (prof_f_mes / inc_f_mes * 100) if inc_f_mes > 0 else 0

inc_total = inc_h_mes + inc_f_mes
prof_total = prof_h_mes + prof_f_mes
margin_total = (prof_total / inc_total * 100) if inc_total > 0 else 0

# --- 7. VISUALIZACIÓN DASHBOARD (RESULTADOS) ---
with tab_dash:
    st.divider()
    st.subheader("2. Estado de Resultados Visual")
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        fig_money = go.Figure()
        cats = ['Harvester', 'Forwarder', 'Total']
        fig_money.add_trace(go.Bar(name='Venta', x=cats, y=[inc_h_mes, inc_f_mes, inc_total], marker_color='#3b82f6', texttemplate='$%{y:,.0f}'))
        fig_money.add_trace(go.Bar(name='Costo', x=cats, y=[cost_h_total_mes_real, cost_f_total_mes_real, cost_total_mes_real], marker_color='#ef4444', texttemplate='$%{y:,.0f}'))
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
            {"Periodo": "Día", "Generado": fmt_money(inc_h_day), "Costo": fmt_money(cost_h_hr*h_hours), "Ganancia": fmt_money(inc_h_day-(cost_h_hr*h_hours))},
            {"Periodo": "Mes", "Generado": fmt_money(inc_h_mes), "Costo": fmt_money(cost_h_total_mes_real), "Ganancia": fmt_money(prof_h_mes)},
        ])
        st.markdown(render_pro_card("🚜 HARVESTER", df_h, margin_h, target_pct, "#eab308"), unsafe_allow_html=True)
        
    with c_t2:
        df_f = pd.DataFrame([
            {"Periodo": "Hora", "Generado": fmt_money(inc_f_hr), "Costo": fmt_money(cost_f_hr), "Ganancia": fmt_money(inc_f_hr-cost_f_hr)},
            {"Periodo": "Día", "Generado": fmt_money(inc_f_day), "Costo": fmt_money(cost_f_hr*f_hours), "Ganancia": fmt_money(inc_f_day-(cost_f_hr*f_hours))},
            {"Periodo": "Mes", "Generado": fmt_money(inc_f_mes), "Costo": fmt_money(cost_f_total_mes_real), "Ganancia": fmt_money(prof_f_mes)},
        ])
        st.markdown(render_pro_card("🚜 FORWARDER", df_f, margin_f, target_pct, "#22c55e"), unsafe_allow_html=True)

# --- TAB FAENA (ACTUALIZADA) ---
with tab_faena:
    st.header("🧮 Cierre de Faena")
    st.markdown("Ingresa el **Total de Metros Ruma (MR)** de una faena para ver su resultado específico.")
    
    # IMPORTANTE: Capturamos el valor aquí
    mr_lote = st.number_input("Total MR Faena", value=1000.0, step=100.0)
    
    if mr_lote > 0:
        st.divider()
        
        # 1. Calculo Tiempo
        req_hrs_h = mr_lote / mr_h_hr if mr_h_hr > 0 else 0
        req_hrs_f = mr_lote / mr_f_hr if mr_f_hr > 0 else 0
        
        req_days_h = req_hrs_h / h_hours if h_hours > 0 else 0
        req_days_f = req_hrs_f / f_hours if f_hours > 0 else 0
        
        # 2. Generado (Venta)
        rev_lote_h = mr_lote * st.session_state['price_h']
        rev_lote_f = mr_lote * st.session_state['price_f']
        
        # 3. Costo Real (Costo Hora Total * Horas Requeridas)
        cost_lote_h = req_hrs_h * cost_h_hr
        cost_lote_f = req_hrs_f * cost_f_hr
        
        # 4. Utilidad
        util_lote_h = rev_lote_h - cost_lote_h
        util_lote_f = rev_lote_f - cost_lote_f
        
        c_res1, c_res2 = st.columns(2)
        
        with c_res1:
            st.markdown(f"""
            <div class="machine-card" style="border-top-color:#eab308">
                <div class="card-header">🚜 Harvester</div>
                <div class="faena-metric"><span>Prod. Horaria:</span><span class="faena-val">{mr_h_hr:,.1f} MR/hr</span></div>
                <div class="faena-metric"><span>Tiempo Total:</span><span class="faena-val">{req_hrs_h:,.1f} hrs ({req_days_h:,.1f} días)</span></div>
                <div class="faena-metric"><span>Generado:</span><span class="faena-val" style="color:#2563eb">{fmt_money(rev_lote_h)}</span></div>
                <div class="faena-metric"><span>Costo Real:</span><span class="faena-val" style="color:#dc2626">{fmt_money(cost_lote_h)}</span></div>
                <div class="faena-result-box"><span style="color:#475569; font-weight:600">UTILIDAD:</span><br><span style="font-size:1.4em; font-weight:800; color:{'#166534' if util_lote_h>0 else '#b91c1c'}">{fmt_money(util_lote_h)}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_res2:
            st.markdown(f"""
            <div class="machine-card" style="border-top-color:#22c55e">
                <div class="card-header">🚜 Forwarder</div>
                <div class="faena-metric"><span>Prod. Horaria:</span><span class="faena-val">{mr_f_hr:,.1f} MR/hr</span></div>
                <div class="faena-metric"><span>Tiempo Total:</span><span class="faena-val">{req_hrs_f:,.1f} hrs ({req_days_f:,.1f} días)</span></div>
                <div class="faena-metric"><span>Generado:</span><span class="faena-val" style="color:#2563eb">{fmt_money(rev_lote_f)}</span></div>
                <div class="faena-metric"><span>Costo Real:</span><span class="faena-val" style="color:#dc2626">{fmt_money(cost_lote_f)}</span></div>
                <div class="faena-result-box"><span style="color:#475569; font-weight:600">UTILIDAD:</span><br><span style="font-size:1.4em; font-weight:800; color:{'#166534' if util_lote_f>0 else '#b91c1c'}">{fmt_money(util_lote_f)}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        total_util_faena = util_lote_h + util_lote_f
        total_generated = rev_lote_h + rev_lote_f
        margin_faena = (total_util_faena / total_generated * 100) if total_generated > 0 else 0
        
        st.markdown(f"""
        <div style='text-align:center; padding:25px; background:#1e293b; border-radius:12px; margin-top:20px; color:white'>
            <h3 style='color:#94a3b8; margin:0; font-size:1.2em'>RESULTADO CONSOLIDADO FAENA</h3>
            <h1 style='color:#4ade80; margin:10px 0; font-size:2.5em'>{fmt_money(total_util_faena)}</h1>
            <div style='font-size:1.1em; font-weight:600; color:#cbd5e1'>Margen Operacional: {margin_faena:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# --- GENERACIÓN PDF (Data Prep - AHORA CON DATOS EXACTOS) ---
pdf_kpis = {
    'mr_h': mr_h_hr * h_hours * h_dias, 'mr_f': mr_f_hr * f_hours * f_dias,
    'inc_total': inc_total, 'cost_total': cost_total_mes_real, 'prof_total': prof_total,
    'margin_total': margin_total,
    'margin_h': margin_h, 'margin_f': margin_f,
    'inc_h_mes': inc_h_mes, 'cost_h_mes': cost_h_total_mes_real, 'prof_h_mes': prof_h_mes,
    'inc_f_mes': inc_f_mes, 'cost_f_mes': cost_f_total_mes_real, 'prof_f_mes': prof_f_mes,
    'cost_sys_hr_h': cost_h_hr,
    'cost_sys_hr_f': cost_f_hr,
    # Datos para sincronización exacta
    'mr_h_hr': mr_h_hr,
    'mr_f_hr': mr_f_hr,
    'mr_lote_input': mr_lote # Valor ingresado por el usuario
}

with st.sidebar:
    try:
        pdf_bytes = create_pro_pdf(st.session_state, pdf_kpis)
        st.download_button("📄 DESCARGAR INFORME PDF", data=pdf_bytes, file_name=f"Reporte_Galvez_{datetime.now().strftime('%Y%m%d')}.pdf", mime='application/pdf', type="primary")
    except Exception as e: st.error(f"Error PDF: {e}")

# Guardado automático al final
save_config()
