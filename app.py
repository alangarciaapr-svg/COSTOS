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
    .machine-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 5px solid #3b82f6; /* Azul por defecto */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card-h { border-top-color: #eab308; } /* Amarillo Harvester */
    .card-f { border-top-color: #22c55e; } /* Verde Forwarder */
    
    .card-title { font-size: 1.2em; font-weight: 800; color: #334155; margin-bottom: 15px; text-transform: uppercase; }
    
    .res-row { display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 4px; }
    .res-label { color: #64748b; font-weight: 500; }
    .res-val { color: #0f172a; font-weight: 700; }
    .res-net { font-size: 1.3em; font-weight: 800; color: #16a34a; margin-top: 10px; text-align: right; }
    .res-loss { font-size: 1.3em; font-weight: 800; color: #dc2626; margin-top: 10px; text-align: right; }

    .mr-badge { background-color: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_v18_machine_split.json'

# --- 2. FUNCIONES GLOBALES ---

def fmt_money(x): 
    if x is None: return "$ 0"
    return f"$ {x:,.0f}".replace(",", ".")

def calculate_single_machine_monthly_cost(df, days, hours, uf, diesel, machine_type='H'):
    df = df.fillna(0)
    total = 0
    total_hours = days * hours
    for _, row in df.iterrows():
        val = float(row.get('Valor', 0))
        tipo = row.get('Tipo') if machine_type == 'H' else row.get('Unidad')
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
            "alloc_pct", "price_h", "price_f", "conv_factor"] # Changed sales_price to price_h/f
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
init_key('price_h', 6500.0) # Tarifa Harvester
init_key('price_f', 5000.0) # Tarifa Forwarder
init_key('alloc_pct', 0.5)
init_key('conv_factor', 2.44)
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
    if st.button("♻️ Reset App"):
        if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
        st.session_state.clear()
        st.rerun()

    with st.expander("💰 Mercado", expanded=True):
        use_api = st.checkbox("UF Auto", value=True)
        uf_api, _ = get_uf_api()
        val_uf = uf_api if (use_api and uf_api) else st.session_state['uf_manual']
        
        curr_uf = st.number_input("Valor UF", value=float(val_uf), disabled=bool(use_api and uf_api))
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()
        
        st.session_state['fuel_price'] = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config)

    with st.expander("💵 Tarifas por Máquina", expanded=True):
        st.info("Define cuánto cobras por el servicio de cada máquina.")
        st.session_state['price_h'] = st.number_input("Tarifa Harvester ($/MR)", value=float(st.session_state['price_h']), on_change=save_config)
        st.session_state['price_f'] = st.number_input("Tarifa Forwarder ($/MR)", value=float(st.session_state['price_f']), on_change=save_config)
        st.caption(f"**Sistema Total:** {fmt_money(st.session_state['price_h'] + st.session_state['price_f'])} / MR")

    with st.expander("📏 Conversión y Distribución"):
        st.session_state['conv_factor'] = st.number_input("Factor m³/MR", value=float(st.session_state['conv_factor']), step=0.01, on_change=save_config)
        alloc = st.slider("% Indirectos a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()

# --- 5. CÁLCULOS GLOBALES ---
tot_h_dir, tot_f_dir, tot_ind, hrs_h, hrs_f = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    int(st.session_state['h_days']), float(st.session_state['h_hours']), 
    int(st.session_state['f_days']), float(st.session_state['f_hours']), 
    curr_uf, st.session_state['fuel_price']
)

# Asignación de Indirectos
ind_h = tot_ind * st.session_state['alloc_pct']
ind_f = tot_ind * (1 - st.session_state['alloc_pct'])

# Costos Mensuales Totales por Máquina
cost_total_h_mes = tot_h_dir + ind_h
cost_total_f_mes = tot_f_dir + ind_f

# --- 6. INTERFAZ ---
st.title("🌲 Resultado Operacional por Máquina")

tab_dash, tab_h, tab_f, tab_ind = st.tabs(["📊 Dashboard Gerencial", "🚜 Harvester", "🚜 Forwarder", "👷 Indirectos"])

# --- TAB DASHBOARD ---
with tab_dash:
    # 1. Inputs Producción
    st.markdown("### 1. Ingreso de Producción Real")
    c1, c2 = st.columns(2)
    with c1:
        prod_h_hr = st.number_input("Productividad Harvester (m³/hr)", value=25.0, step=0.5)
        mr_h_hr = prod_h_hr / st.session_state['conv_factor']
        st.caption(f"Equivale a: **{mr_h_hr:,.1f} MR/hr**")
    with c2:
        prod_f_hr = st.number_input("Productividad Forwarder (m³/hr)", value=28.0, step=0.5)
        mr_f_hr = prod_f_hr / st.session_state['conv_factor']
        st.caption(f"Equivale a: **{mr_f_hr:,.1f} MR/hr**")

    # 2. Cálculos de Negocio
    # Proyección Mensual (Prod Hora * Horas Mes Configuradas)
    mr_h_mes = mr_h_hr * hrs_h
    mr_f_mes = mr_f_hr * hrs_f
    
    # Ingresos (Producción MR * Tarifa Especifica)
    ingreso_h_mes = mr_h_mes * st.session_state['price_h']
    ingreso_f_mes = mr_f_mes * st.session_state['price_f']
    
    # Utilidad
    utilidad_h_mes = ingreso_h_mes - cost_total_h_mes
    utilidad_f_mes = ingreso_f_mes - cost_total_f_mes
    
    # Márgenes
    margen_h = (utilidad_h_mes / ingreso_h_mes * 100) if ingreso_h_mes > 0 else 0
    margen_f = (utilidad_f_mes / ingreso_f_mes * 100) if ingreso_f_mes > 0 else 0

    st.divider()
    st.markdown("### 2. Estado de Resultados por Máquina")

    # 3. VISUALIZACIÓN SPLIT (Harvester vs Forwarder)
    col_h, col_f = st.columns(2)
    
    def render_machine_card(title, prod_mr, inc, cost, prof, margin, style_class):
        color_prof = "#16a34a" if prof >= 0 else "#dc2626"
        return f"""
        <div class="machine-card {style_class}">
            <div class="card-title">{title}</div>
            <div class="res-row"><span class="res-label">Producción Mes</span><span class="res-val">{prod_mr:,.0f} MR</span></div>
            <div class="res-row"><span class="res-label">Ingresos</span><span class="res-val">{fmt_money(inc)}</span></div>
            <div class="res-row"><span class="res-label">Costo Total</span><span class="res-val" style="color:#ef4444">{fmt_money(cost)}</span></div>
            <div class="{ 'res-net' if prof >=0 else 'res-loss' }">{fmt_money(prof)}</div>
            <div style="text-align:right; font-size:0.9em; color:{color_prof}">Margen: {margin:.1f}%</div>
        </div>
        """

    with col_h:
        st.markdown(render_machine_card("🚜 HARVESTER", mr_h_mes, ingreso_h_mes, cost_total_h_mes, utilidad_h_mes, margen_h, "card-h"), unsafe_allow_html=True)
        # Desglose H
        with st.expander("Ver detalle Harvester (Hr/Sem/Mes)"):
            st.dataframe(pd.DataFrame({
                "Periodo": ["Hora", "Semana", "Mes"],
                "Generado ($)": [fmt_money(ingreso_h_mes/hrs_h), fmt_money(ingreso_h_mes/4), fmt_money(ingreso_h_mes)],
                "Costo ($)": [fmt_money(cost_total_h_mes/hrs_h), fmt_money(cost_total_h_mes/4), fmt_money(cost_total_h_mes)],
                "Utilidad ($)": [fmt_money(utilidad_h_mes/hrs_h), fmt_money(utilidad_h_mes/4), fmt_money(utilidad_h_mes)]
            }), hide_index=True)

    with col_f:
        st.markdown(render_machine_card("🚜 FORWARDER", mr_f_mes, ingreso_f_mes, cost_total_f_mes, utilidad_f_mes, margen_f, "card-f"), unsafe_allow_html=True)
        # Desglose F
        with st.expander("Ver detalle Forwarder (Hr/Sem/Mes)"):
            st.dataframe(pd.DataFrame({
                "Periodo": ["Hora", "Semana", "Mes"],
                "Generado ($)": [fmt_money(ingreso_f_mes/hrs_f), fmt_money(ingreso_f_mes/4), fmt_money(ingreso_f_mes)],
                "Costo ($)": [fmt_money(cost_total_f_mes/hrs_f), fmt_money(cost_total_f_mes/4), fmt_money(cost_total_f_mes)],
                "Utilidad ($)": [fmt_money(utilidad_f_mes/hrs_f), fmt_money(utilidad_f_mes/4), fmt_money(utilidad_f_mes)]
            }), hide_index=True)

    # 4. GRÁFICO COMPARATIVO
    st.subheader("Comparativa Visual de Rentabilidad")
    
    # Preparar datos para gráfico
    data_plot = [
        {"Máquina": "Harvester", "Tipo": "Ingreso", "Monto": ingreso_h_mes},
        {"Máquina": "Harvester", "Tipo": "Costo", "Monto": cost_total_h_mes},
        {"Máquina": "Forwarder", "Tipo": "Ingreso", "Monto": ingreso_f_mes},
        {"Máquina": "Forwarder", "Tipo": "Costo", "Monto": cost_total_f_mes},
    ]
    
    fig = px.bar(data_plot, x="Máquina", y="Monto", color="Tipo", barmode="group",
                 color_discrete_map={"Ingreso": "#3b82f6", "Costo": "#ef4444"}, text_auto='.2s')
    
    fig.add_trace(go.Scatter(
        x=["Harvester", "Forwarder"], y=[ingreso_h_mes, ingreso_f_mes],
        text=[f"{margen_h:.1f}% Margen", f"{margen_f:.1f}% Margen"],
        mode="text", textposition="top center", name="Margen %"
    ))
    
    st.plotly_chart(fig, use_container_width=True)

# --- TABS EDICIÓN ---
with tab_h:
    st.header("Costos Harvester")
    edited_h = st.data_editor(st.session_state['df_harvester'], num_rows="dynamic", use_container_width=True,
                              column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_harvester'] = edited_h
    save_config()
    # Total Vivo
    live_h = calculate_single_machine_monthly_cost(edited_h, int(st.session_state['h_days']), float(st.session_state['h_hours']), curr_uf, st.session_state['fuel_price'], 'H')
    st.success(f"💰 Costo Mensual Directo H: **{fmt_money(live_h)}**")

with tab_f:
    st.header("Costos Forwarder")
    edited_f = st.data_editor(st.session_state['df_forwarder'], num_rows="dynamic", use_container_width=True,
                              column_config={"Valor": st.column_config.NumberColumn(format="$ %d")})
    st.session_state['df_forwarder'] = edited_f
    save_config()
    live_f = calculate_single_machine_monthly_cost(edited_f, int(st.session_state['f_days']), float(st.session_state['f_hours']), curr_uf, st.session_state['fuel_price'], 'F')
    st.success(f"💰 Costo Mensual Directo F: **{fmt_money(live_f)}**")

with tab_ind:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RRHH")
        st.session_state['df_rrhh'] = st.data_editor(st.session_state['df_rrhh'], num_rows="dynamic", column_config={"Costo Empresa": st.column_config.NumberColumn(format="$ %d")})
    with c2:
        st.subheader("Flota y Varios")
        st.session_state['df_flota'] = st.data_editor(st.session_state['df_flota'], num_rows="dynamic", column_config={"Monto": st.column_config.NumberColumn(format="$ %d")})
    save_config()
    live_ind = st.session_state['df_rrhh']['Costo Empresa'].sum() + st.session_state['df_flota']['Monto'].sum()
    st.info(f"💰 Total Indirectos Mensual: **{fmt_money(live_ind)}**")
