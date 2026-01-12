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
    page_title="Master Forestal Costing", 
    layout="wide", 
    page_icon="🌲",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f4f6f9;}
    h1, h2, h3 {color: #1e293b; font-family: 'Helvetica Neue', sans-serif;}
    .stMetric {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

CONFIG_FILE = 'forest_config_master_v2.json'

# --- 2. FUNCIONES BACKEND ---

@st.cache_data(ttl=3600) 
def get_uf_api():
    try:
        url = "https://mindicador.cl/api/uf"
        response = requests.get(url, timeout=3)
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
            "df_harvester", "df_forwarder", "df_rrhh", "df_flota", "alloc_pct", "sales_price", "target_margin"]
    
    state_to_save = {}
    for k in keys:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame): state_to_save[k] = val.to_dict('records')
            else: state_to_save[k] = val
                
    with open(CONFIG_FILE, 'w') as f:
        json.dump(state_to_save, f, cls=NumpyEncoder)

def safe_float(val, default_val):
    try: return float(val) if val is not None else default_val
    except: return default_val

def fmt_money(x): 
    return f"$ {x:,.0f}".replace(",", ".")

# --- 3. INICIALIZACIÓN ROBUSTA (FIX PARA KEYERROR) ---
# Cargamos configuración guardada o diccionarios vacíos
saved = load_config()

# Función para inicializar una key si no existe
def init_key(key, default_value):
    if key not in st.session_state:
        # Intentar cargar del archivo, si no, usar default
        loaded_val = saved.get(key)
        if loaded_val is not None:
            if isinstance(default_value, pd.DataFrame):
                st.session_state[key] = pd.DataFrame(loaded_val)
            else:
                st.session_state[key] = loaded_val
        else:
            st.session_state[key] = default_value

# Inicializamos una por una para evitar errores
init_key('uf_manual', 39755.0)
init_key('fuel_price', 774.0)
init_key('sales_price', 11500.0)
init_key('alloc_pct', 0.5)
init_key('target_margin', 15.0)
init_key('h_days', 28)
init_key('h_hours', 10.0)
init_key('f_days', 28)
init_key('f_hours', 10.0)

# DataFrames (Aquí fallaba antes, ahora está protegido)
init_key('df_harvester', pd.DataFrame([
    {"Cat": "Fijos", "Ítem": "Arriendo Base", "Tipo": "$/Mes", "Frec": 1, "Valor": 10900000},
    {"Cat": "Fijos", "Ítem": "Operador T1", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Fijos", "Ítem": "Operador T2", "Tipo": "$/Mes", "Frec": 1, "Valor": 1923721},
    {"Cat": "Variable", "Ítem": "Petróleo T1", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Variable", "Ítem": "Petróleo T2", "Tipo": "Litros/Día", "Frec": 1, "Valor": 200.0},
    {"Cat": "Insumos", "Ítem": "Cadenas/Espadas", "Tipo": "$/Mes", "Frec": 1, "Valor": 450000},
    {"Cat": "Insumos", "Ítem": "Aceite Hidráulico", "Tipo": "$/Mes", "Frec": 1, "Valor": 180000},
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

# Tablas Nuevas (RRHH y Flota) - Protegidas
init_key('df_rrhh', pd.DataFrame([
    {"Cargo": "Jefe de Faena", "Sueldo Líquido": 1800000, "Costo Empresa": 2300000},
    {"Cargo": "Mecánico", "Sueldo Líquido": 1200000, "Costo Empresa": 1600000},
    {"Cargo": "Prevencionista", "Sueldo Líquido": 900000, "Costo Empresa": 1200000},
    {"Cargo": "Estrobero/Cancha", "Sueldo Líquido": 600000, "Costo Empresa": 850000},
]))

init_key('df_flota', pd.DataFrame([
    {"Ítem": "Camioneta 1 (Arriendo)", "Monto": 800000},
    {"Ítem": "Camioneta 2 (Arriendo)", "Monto": 800000},
    {"Ítem": "Combustible Camionetas", "Monto": 600000},
    {"Ítem": "Pensión/Alojamiento", "Monto": 1500000},
    {"Ítem": "Gastos Adm. Central", "Monto": 500000},
]))

# --- 4. CÁLCULOS CENTRALIZADOS ---
def calculate_system_costs(h_df, f_df, rrhh_df, flota_df, days_h, hrs_h, days_f, hrs_f, uf, diesel):
    # 1. Harvester
    total_h = 0
    total_h_hrs = days_h * hrs_h
    for _, row in h_df.iterrows():
        val = row['Valor'] or 0
        tipo = row['Tipo']
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'UF/Mes': cost = val * uf
        elif tipo == 'Litros/Día': cost = val * days_h * diesel
        elif tipo == '$/Ev': 
            frec = row.get('Frec', 1)
            if frec > 0 and total_h_hrs > 0: cost = (val / frec) * total_h_hrs
        total_h += cost

    # 2. Forwarder
    total_f = 0
    total_f_hrs = days_f * hrs_f
    for _, row in f_df.iterrows():
        val = row['Valor'] or 0
        tipo = row['Unidad']
        cost = 0
        if tipo == '$/Mes': cost = val
        elif tipo == 'Litros/Día': cost = val * days_f * diesel
        total_f += cost

    # 3. Indirectos (RRHH + Flota)
    total_indirect = rrhh_df['Costo Empresa'].sum() + flota_df['Monto'].sum()

    return total_h, total_f, total_indirect, total_h_hrs, total_f_hrs

# --- 5. INTERFAZ: SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2823/2823538.png", width=50)
    st.title("Configuración Maestra")
    
    with st.expander("💰 Mercado y Tarifas", expanded=True):
        # UF
        use_api = st.checkbox("UF Online", value=True)
        uf_api_val, _ = get_uf_api()
        if use_api and uf_api_val:
            st.success(f"UF: ${uf_api_val:,.2f}")
            val_uf = uf_api_val
        else:
            val_uf = st.session_state['uf_manual']
        
        curr_uf = st.number_input("Valor UF", value=float(val_uf), disabled=use_api and uf_api_val is not None)
        if curr_uf != st.session_state['uf_manual']:
            st.session_state['uf_manual'] = curr_uf
            save_config()

        # Diesel
        curr_fuel = st.number_input("Diesel ($/Lt)", value=float(st.session_state['fuel_price']), on_change=save_config, key="fuel_price")
        
        # Tarifa Venta
        curr_sales = st.number_input("Tarifa Venta ($/MR)", value=float(st.session_state['sales_price']), on_change=save_config, key="sales_price")

    with st.expander("⚙️ Distribución Costos Indirectos"):
        alloc = st.slider("% Asignado a Harvester", 0, 100, int(st.session_state['alloc_pct']*100)) / 100.0
        if alloc != st.session_state['alloc_pct']:
            st.session_state['alloc_pct'] = alloc
            save_config()
        st.caption(f"Harvester: {alloc*100:.0f}% | Forwarder: {(1-alloc)*100:.0f}%")

    if st.button("💾 Guardar Cambios"):
        save_config()
        st.toast("Configuración guardada exitosamente")

# --- 6. INTERFAZ: CUERPO PRINCIPAL ---
st.title("🌲 Sistema de Costos Forestales Profesional")
st.markdown(f"**Tarifa Actual:** {fmt_money(st.session_state['sales_price'])} / MR | **UF:** {fmt_money(curr_uf)}")

# PESTAÑAS
tab_dash, tab_h, tab_f, tab_ind, tab_sim = st.tabs([
    "📊 Dashboard & Margen", 
    "🚜 Harvester", 
    "🚜 Forwarder", 
    "👷 Indirectos (RRHH/Flota)",
    "📈 Matriz Sensibilidad"
])

# --- TAB HARVESTER ---
with tab_h:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Jornada")
        h_days = st.number_input("Días/Mes H", value=int(st.session_state.get('h_days', 28)), key="h_days", on_change=save_config)
        h_hours = st.number_input("Horas/Día H", value=float(st.session_state.get('h_hours', 10.0)), key="h_hours", on_change=save_config)
        st.info(f"Horas Mes: **{h_days * h_hours:,.0f}**")
    with c2:
        st.subheader("Estructura de Costos Directos")
        st.session_state['df_harvester'] = st.data_editor(
            st.session_state['df_harvester'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d"),
                "Tipo": st.column_config.SelectboxColumn(options=["$/Mes", "UF/Mes", "Litros/Día", "$/Ev"], required=True),
                "Cat": st.column_config.SelectboxColumn(options=["Fijos", "Variable", "Insumos", "Mantención", "Mayor"], required=True)
            }
        )
        save_config()

# --- TAB FORWARDER ---
with tab_f:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Jornada")
        f_days = st.number_input("Días/Mes F", value=int(st.session_state.get('f_days', 28)), key="f_days", on_change=save_config)
        f_hours = st.number_input("Horas/Día F", value=float(st.session_state.get('f_hours', 10.0)), key="f_hours", on_change=save_config)
        st.info(f"Horas Mes: **{f_days * f_hours:,.0f}**")
    with c2:
        st.subheader("Estructura Simplificada")
        st.session_state['df_forwarder'] = st.data_editor(
            st.session_state['df_forwarder'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Valor": st.column_config.NumberColumn(format="$ %d"),
                "Unidad": st.column_config.SelectboxColumn(options=["$/Mes", "Litros/Día"], required=True)
            }
        )
        save_config()

# --- TAB INDIRECTOS (MEJORADO SEGÚN EXCEL) ---
with tab_ind:
    c_rrhh, c_flota = st.columns(2)
    
    with c_rrhh:
        st.markdown("### 👷 RRHH Indirecto")
        st.markdown("*Jefatura, Mecánicos, Prevención*")
        st.session_state['df_rrhh'] = st.data_editor(
            st.session_state['df_rrhh'], use_container_width=True, num_rows="dynamic",
            column_config={
                "Sueldo Líquido": st.column_config.NumberColumn(format="$ %d"),
                "Costo Empresa": st.column_config.NumberColumn(format="$ %d", help="Incluye Leyes Sociales + Bonos")
            }
        )
        total_rrhh = st.session_state['df_rrhh']['Costo Empresa'].sum()
        st.metric("Total RRHH Mes", fmt_money(total_rrhh))

    with c_flota:
        st.markdown("### 🛻 Flota y Gastos Generales")
        st.markdown("*Camionetas, Combustible Apoyo, Pensión*")
        st.session_state['df_flota'] = st.data_editor(
            st.session_state['df_flota'], use_container_width=True, num_rows="dynamic",
            column_config={"Monto": st.column_config.NumberColumn(format="$ %d")}
        )
        total_flota = st.session_state['df_flota']['Monto'].sum()
        st.metric("Total Flota/Admin Mes", fmt_money(total_flota))
    
    save_config()

# --- CÁLCULO FINAL ---
tot_h_dir, tot_f_dir, tot_ind, hrs_h, hrs_f = calculate_system_costs(
    st.session_state['df_harvester'], st.session_state['df_forwarder'], 
    st.session_state['df_rrhh'], st.session_state['df_flota'],
    h_days, h_hours, f_days, f_hours, curr_uf, curr_fuel
)

# Asignación
ind_h = tot_ind * st.session_state['alloc_pct']
ind_f = tot_ind * (1 - st.session_state['alloc_pct'])

final_h_mes = tot_h_dir + ind_h
final_f_mes = tot_f_dir + ind_f
final_sys_mes = final_h_mes + final_f_mes

# Costo Hora
cost_hr_h = final_h_mes / hrs_h if hrs_h > 0 else 0
cost_hr_f = final_f_mes / hrs_f if hrs_f > 0 else 0
cost_hr_sys = cost_hr_h + cost_hr_f

# --- TAB DASHBOARD ---
with tab_dash:
    # KPIs Top
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Costo Sistema Hora", fmt_money(cost_hr_sys), "Total Operacional")
    k2.metric("Costo Harvester/Hr", fmt_money(cost_hr_h), "Directo + Indirecto")
    k3.metric("Costo Forwarder/Hr", fmt_money(cost_hr_f), "Directo + Indirecto")
    
    # Producción de Equilibrio
    mr_eq = cost_hr_sys / curr_sales if curr_sales > 0 else 0
    k4.metric("Pto. Equilibrio", f"{mr_eq:.1f} MR/Hr", "Para cubrir costos")

    st.divider()

    # Visualización Costos
    c_chart, c_table = st.columns([1, 1])
    
    with c_chart:
        st.subheader("Composición del Costo Mensual")
        # Datos para gráfico Sunburst
        df_sun = pd.DataFrame([
            ['Sistema', 'Harvester', final_h_mes],
            ['Sistema', 'Forwarder', final_f_mes],
            ['Harvester', 'Directo H', tot_h_dir],
            ['Harvester', 'Indirecto H', ind_h],
            ['Forwarder', 'Directo F', tot_f_dir],
            ['Forwarder', 'Indirecto F', ind_f],
        ], columns=['Parent', 'Label', 'Value'])
        
        fig_sun = px.sunburst(df_sun, names='Label', parents='Parent', values='Value', branchvalues='total')
        st.plotly_chart(fig_sun, use_container_width=True)

    with c_table:
        st.subheader("Resumen Financiero Mensual")
        res_df = pd.DataFrame({
            "Concepto": ["Directo Maquinaria", "RRHH Indirecto", "Flota y Admin", "TOTAL MENSUAL"],
            "Monto": [
                fmt_money(tot_h_dir + tot_f_dir),
                fmt_money(total_rrhh),
                fmt_money(total_flota),
                fmt_money(final_sys_mes)
            ]
        })
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        st.info(f"Días Trabajados: {h_days} | Horas Totales: {hrs_h}")

# --- TAB SIMULACIÓN / SENSIBILIDAD ---
with tab_sim:
    st.markdown("### 📈 Análisis de Rentabilidad y Sensibilidad")
    st.markdown("Analiza cómo cambia tu **Margen (%)** según la **Productividad (MR/Hr)** y el **Precio de Venta**.")

    # Controles Simulación
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        sim_prod = st.slider("Productividad Real (MR/Hr)", 10.0, 40.0, 22.0, 0.5)
    with col_ctrl2:
        st.metric("Margen Actual Estimado", 
                  f"{((sim_prod * curr_sales - cost_hr_sys)/(sim_prod * curr_sales)*100):.1f}%",
                  f"Utilidad: {fmt_money((sim_prod * curr_sales) - cost_hr_sys)} / hr")

    st.divider()
    
    # MATRIZ DE CALOR (HEATMAP)
    # Generar rangos
    rango_precios = np.linspace(curr_sales * 0.8, curr_sales * 1.2, 10)
    rango_prod = np.linspace(10, 40, 10)
    
    z_data = []
    for p in rango_prod:
        row = []
        for price in rango_precios:
            ingreso = p * price
            margen = ((ingreso - cost_hr_sys) / ingreso) * 100
            row.append(margen)
        z_data.append(row)
        
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f"${p:,.0f}" for p in rango_precios],
        y=[f"{pr:.1f}" for pr in rango_prod],
        colorscale='RdYlGn',
        zmin=-20, zmax=40,
        texttemplate="%{z:.0f}%",
        hoverongaps=False
    ))
    
    fig_heat.update_layout(
        title="Matriz de Rentabilidad (Margen %)",
        xaxis_title="Precio de Venta ($/MR)",
        yaxis_title="Productividad (MR/Hr)",
        height=500
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Rojo: Pérdida | Verde: Ganancia. El eje Y muestra cuántos metros ruma haces por hora.")
