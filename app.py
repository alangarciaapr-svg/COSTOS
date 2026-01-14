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
        self.cell(0, 10, f'Pagina {self.page_no
