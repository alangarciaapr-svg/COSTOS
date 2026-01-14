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
    
    /* Ajuste de Texto para que no se corten cifras */
    .sim-val {
        font-size: 1.1em;
        font-weight: 700;
        color: #0f172a;
    }
    .sim-total {
        font-size: 1.4em; 
        font-weight: 900;
        text-align: center;
        margin-top: 5px;
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

CONFIG_FILE = 'forest_config_v36_hourly_fix.json'

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
    if os.path.exists(
