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
    st.error("Librería fpdf no instalada. Por favor agrega 'fpdf' a requirements.txt")

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
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    
    .machine-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-top: 4px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .card-header { font-size: 1.1em; font-weight: 800; color: #334155; display: flex; justify-content: space-between; margin-bottom: 15px; }
    .badge-success { background-color: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v26_fixed_graphs.json'

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
        # Fondo Encabezado
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 40, 'F')
        
        self.set_font('Arial', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 10)
        self.cell(0, 10, 'SOCIEDAD MADERERA GÁLVEZ Y DI GÉNOVA LDS', 0, 1, 'L')
        
        self.set_font('Arial', '', 11)
        self.set_text_color(203, 213, 225)
        self.cell(0, 5, 'INFORME DE GESTIÓN: COSTOS V/S PRODUCCIÓN', 0, 1, 'L')
        
        self.set_xy(150, 15)
        self.set_font('Arial', 'B', 10)
        self.cell(50, 10, datetime.now().strftime('%d/%m/%Y'), 0, 1, 'R')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(30, 41, 59)
        self.set_fill_color(226, 232, 240)
        self.cell(0, 8, f"  {label}", 0, 1, 'L', 1)
        self.ln(4)

    def kpi_box(self, label, value, x, y, width, color_header):
        self.set_xy(x, y)
        self.set_draw_color(200, 200, 200)
        self.rect(x, y, width, 25)
        
        self.set_fill_color(*color_header)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 9)
        self.cell(width, 7, label, 1, 1, 'C', 1)
        
        self.set_xy(x, y + 8)
        self.set_text_color(30, 41, 59)
        self.set_font('Arial', 'B', 12)
        self.cell(width, 17, value, 0, 0, 'C')

    def table_simple(self, header, data, col_widths):
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(241, 245, 249)
        self.set_text_color(30, 41, 59)
        for i, h in enumerate(header):
            self.cell(col_widths[i], 7, h, 1, 0, 'C', 1)
        self.ln()
        
        self.set_font('Arial', '', 9)
        fill = False
        for row in data:
            self.set_fill_color(250, 250, 250)
            for i, datum in enumerate(row):
                align = 'L' if i == 0 else 'R'
                self.cell(col_widths[i], 6, str(datum), 1, 0, align, fill)
            self.ln()
            fill = not fill

def create_pro_pdf(state, kpis):
    pdf = PDF_Pro()
    pdf.add_page()
    
    pdf.chapter_title('RESUMEN EJECUTIVO MENSUAL')
    start_y = pdf.get_y()
    
    pdf.kpi_box("VENTA TOTAL", fmt_money(kpis['inc_total']), 10, start_y, 60, (59, 130, 246))
    pdf.kpi_box("COSTO TOTAL", fmt_money(kpis['cost_total']), 75, start_y, 60, (239, 68, 68))
    pdf.kpi_box("UTILIDAD NETA", fmt_money(kpis['prof_total']), 140, start_y, 60, (34, 197, 94))
    
    pdf.set_y(start_y + 30)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, f"MARGEN OPERACIONAL: {kpis['margin_total']:.1f}%", 0, 1, 'R')
    pdf.ln(5)
    
    pdf.chapter_title('DETALLE POR MÁQUINA')
    
    # Harvester
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(95, 8, "HARVESTER", 0, 1)
    pdf.set_font('Arial', '', 9)
    pdf.cell(95, 6, f"Producción Mes: {kpis['mr_h']:,.1f} MR", 0, 1)
    pdf.cell(95, 6, f"Tarifa: {fmt_money(state['price_h'])} / MR", 0, 1)
    pdf.cell(95, 6, f"Ingresos: {fmt_money(kpis['inc_h'])}", 0, 1)
    pdf.cell(95, 6, f"Costos: {fmt_money(kpis['cost_h'])}", 0, 1)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(95, 6, f"Utilidad H: {fmt_money(kpis['inc_h'] - kpis['cost_h'])}", 0, 1)
    
    # Forwarder
    pdf.set_xy(110, pdf.get_y() - 30)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(95, 8, "FORWARDER", 0, 1)
    pdf.set_xy(110, pdf.get_y())
    pdf.set_font('Arial', '', 9)
    pdf.cell(95, 6, f"Producción Mes: {kpis['mr_f']:,.1f} MR", 0, 1)
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(95, 6, f"Tarifa: {fmt_money(state['price_f'])} / MR", 0, 1)
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(95, 6, f"Ingresos: {fmt_money(kpis['inc_f'])}", 0, 1)
    pdf.set_xy(110, pdf.get_y())
    pdf.cell(95, 6, f"Costos: {fmt_money(kpis['cost_f'])}", 0, 1)
    pdf.set_xy(110, pdf.get_y())
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(95, 6, f"Utilidad F: {fmt_money(kpis['inc_f'] - kpis['cost_f'])}", 0, 1)
    
    pdf.ln(10)
    
    pdf.chapter_title('ESTRUCTURA DE COSTOS')
    
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, "Costos Directos Harvester", 0, 1)
    data_h = []
    for _, row in state['df_harvester'].iterrows():
        data_h.append([row['Ítem'], row.get('Tipo','$/Mes'), fmt_money(row['Valor'])])
    pdf.table_simple(['Item', 'Unidad', 'Valor'], data_h, [90, 40, 60])
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, "Costos Directos Forwarder", 0, 1)
    data_f = []
    for _, row in state['df_forwarder'].iterrows():
        data_f.append([row['Ítem'], row.get('Unidad','$/Mes'), fmt_money(row['Valor'])])
    pdf.table_simple(['Item', 'Unidad', 'Valor'], data_f, [90, 40, 60])
    
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
            "alloc_pct", "price_h", "price_f", "conv_factor", "target_company_margin"]
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
st.title("🌲 Sistema de Control de Gestión Forestal")

tab_dash, tab_strat, tab_h, tab_f, tab_ind = st.tabs([
    "📊 Resultado Operacional", "🎯 Estrategia Precios", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos"
])

# --- TAB DASHBOARD ---
with tab_dash:
    # 1. INPUTS
    st.subheader("1. Variables de Producción (En Terreno)")
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
    inc_h_hr = mr_h_hr * st.session_state['price_h']
    inc_h_mes = inc_h_hr * h_horas * h_dias
    inc_h_day = inc_h_hr * h_horas
    cost_h_mes = cost_h_total_mes
    cost_h_day = cost_h_mes / h_dias if h_dias > 0 else 0
    cost_h_hr = cost_h_mes / (h_dias * h_horas) if (h_dias*h_horas) > 0 else 0
    prof_h_mes = inc_h_mes - cost_h_mes
    margin_h = (prof_h_mes / inc_h_mes * 100) if inc_h_mes > 0 else 0

    inc_f_hr = mr_f_hr * st.session_state['price_f']
    inc_f_mes = inc_f_hr * f_horas * f_dias
    inc_f_day = inc_f_hr * f_horas
    cost_f_mes = cost_f_total_mes
    cost_f_day = cost_f_mes / f_dias if f_dias > 0 else 0
    cost_f_hr = cost_f_mes / (f_dias * f_horas) if (f_dias*f_horas) > 0 else 0
    prof_f_mes = inc_f_mes - cost_f_mes
    margin_f = (prof_f_mes / inc_f_mes * 100) if inc_f_mes > 0 else 0

    prof_total_mes = prof_h_mes + prof_f_mes
    inc_total_mes = inc_h_mes + inc_f_mes
    cost_total_mes = cost_h_mes + cost_f_mes
    margin_total = (prof_total_mes / inc_total_mes * 100) if inc_total_mes > 0 else 0
    
    # Datos para PDF
    pdf_kpis = {
        'mr_h': mr_h_hr * h_horas * h_dias, 'mr_f': mr_f_hr * f_horas * f_dias,
        'inc_total': inc_total_mes, 'cost_total': cost_total_mes, 'prof_total': prof_total_mes,
        'margin_total': margin_total,
        'inc_h': inc_h_mes, 'cost_h': cost_h_mes, 'inc_f': inc_f_mes, 'cost_f': cost_f_mes
    }

    # 3. GRÁFICOS (CORREGIDOS)
    st.subheader("2. Análisis Visual: Ganancia vs Costo")
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        # Gráfico $ (CORREGIDO: Usando texttemplate en lugar de text_auto)
        fig_money = go.Figure()
        categories = ['Harvester', 'Forwarder', 'Total Sistema']
        
        fig_money.add_trace(go.Bar(name='Venta ($)', x=categories, y=[inc_h_mes, inc_f_mes, inc_total_mes], 
                                   marker_color='#3b82f6', texttemplate='$%{y:,.0f}', textposition='auto'))
        
        fig_money.add_trace(go.Bar(name='Costo ($)', x=categories, y=[cost_h_mes, cost_f_mes, cost_total_mes], 
                                   marker_color='#ef4444', texttemplate='$%{y:,.0f}', textposition='auto'))
        
        fig_money.add_trace(go.Bar(name='Ganancia ($)', x=categories, y=[prof_h_mes, prof_f_mes, prof_total_mes], 
                                   marker_color='#22c55e', texttemplate='$%{y:,.0f}', textposition='auto'))
        
        fig_money.update_layout(barmode='group', title="Resultado Operacional (Dinero)", height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_money, use_container_width=True)
        
    with col_chart_2:
        # Gráfico %
        fig_pct = go.Figure()
        colors_pct = ['#22c55e' if m >= target_pct else '#ef4444' for m in [margin_h, margin_f, margin_total]]
        
        fig_pct.add_trace(go.Bar(
            x=categories, y=[margin_h, margin_f, margin_total],
            marker_color=colors_pct, text=[f"{m:.1f}%" for m in [margin_h, margin_f, margin_total]],
            textposition='auto'
        ))
        
        fig_pct.add_shape(type="line", x0=-0.5, x1=2.5, y0=target_pct, y1=target_pct, line=dict(color="black", width=2, dash="dash"))
        fig_pct.update_layout(title=f"Margen % (Meta: {target_pct}%)", height=350)
        st.plotly_chart(fig_pct, use_container_width=True)

    # 4. TABLAS DETALLADAS
    st.subheader("3. Detalle Financiero por Periodo")
    
    def render_pro_card(title, df, margin, target, color_border):
        badge = "badge-success" if margin >= target else "badge-danger"
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

    c_card1, c_card2 = st.columns(2)
    with c_card1:
        st.markdown(render_pro_card("🚜 HARVESTER", df_h_display, margin_h, target_pct, "#eab308"), unsafe_allow_html=True)
    with c_card2:
        st.markdown(render_pro_card("🚜 FORWARDER", df_f_display, margin_f, target_pct, "#22c55e"), unsafe_allow_html=True)

# --- SIDEBAR (CON PDF) ---
with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    
    try:
        pdf_bytes = create_pro_pdf(st.session_state, pdf_kpis)
        st.download_button(
            "📄 Generar Informe Ejecutivo (PDF)", 
            data=pdf_bytes, 
            file_name=f"Informe_Galvez_Genova_{datetime.now().strftime('%Y%m%d')}.pdf", 
            mime='application/pdf', 
            type="primary"
        )
    except Exception as e:
        st.warning(f"Instala 'fpdf' para el informe: {e}")
    
    if st.button("♻️ Reset App"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    with st.expander("🎯 Objetivos Estratégicos", expanded=True):
        st.session_state['target_company_margin'] = st.slider("Margen Objetivo Empresa (%)", 0, 60, int(st.session_state['target_company_margin']), on_change=save_config)

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
